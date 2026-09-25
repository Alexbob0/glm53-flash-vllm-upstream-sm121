#!/usr/bin/env python3
"""Real-vLLM replay test for the MAMBA_FREE overlay (runs inside glm53-upstream:b4, no GPU).

Drives vLLM's own MambaManager + BlockPool through a cold prefill the way
kv_cache_manager.allocate_slots does (remove_skipped_blocks on the processed
prefix = computed - in flight, then allocation, then cache_blocks), with our
prod geometry: Mamba block 4608, hash block 64, 7 speculative blocks, one batch
in flight (async scheduling), chunk = MNBT 7168 aligned down to the Mamba block.

    docker run --rm --entrypoint python3 \
      -v $PWD:/t:ro [-v $PWD/single_type_kv_cache_manager.py:<vllm>/v1/core/single_type_kv_cache_manager.py:ro] \
      glm53-upstream:b4 /t/test_mamba_free_nightly.py [stock|patched]

"stock" expects the leak (proves the test sees it); "patched" expects the bound.
"""
import sys
import types

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec
import torch

MODE = sys.argv[1] if len(sys.argv) > 1 else "patched"
BLOCK, HASH, SPEC = 4608, 64, 7
BATCHES = 2  # async scheduling
FAIL = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAIL.append(label)


def make():
    spec = MambaSpec(block_size=BLOCK, shapes=((4,),), dtypes=(torch.float32,), mamba_cache_mode="align",
                     num_speculative_blocks=SPEC)
    pool = BlockPool(num_gpu_blocks=4096, enable_caching=True, hash_block_size=HASH)
    mgr = MambaManager(spec, block_pool=pool, enable_caching=True, kv_cache_group_id=1, scheduler_block_size=BLOCK)
    return mgr, pool


def referenced(pool):
    return sum(1 for b in pool.blocks if b.ref_cnt > 0 and not b.is_null)


def replay(mgr, rid, tokens, chunk, in_flight, retention):
    req = types.SimpleNamespace(request_id=rid, num_prompt_tokens=tokens, num_tokens=tokens,
                                block_hashes=[BlockHash(f"{rid}-{i}".encode()) for i in range(tokens // HASH)],
                                shared_prefix_boundary=0)
    computed, last, held, live_ok = 0, 0, [], True
    while computed < tokens:
        new = min(chunk, tokens - computed)
        processed = max(0, computed - (last if in_flight else 0))
        mgr.new_step_starts()
        mgr.remove_skipped_blocks(rid, processed, num_prompt_tokens=tokens)
        blocks = mgr.req_to_blocks[rid]
        # the state at the processed boundary feeds the in-flight step; the newest state feeds the next
        for t in (processed, computed):
            if t:
                live_ok &= not blocks[-(-t // BLOCK) - 1].is_null
        mgr.get_num_blocks_to_allocate(rid, computed + new, [], computed, computed, computed + new)
        mgr.allocate_new_blocks(rid, computed + new, computed + new)
        mgr.cache_blocks(req, computed + new, retention_interval=retention)
        computed += new
        last = new
        held.append(sum(not b.is_null for b in mgr.req_to_blocks[rid]))
    return held, live_ok


print(f"mode={MODE}")
bound = 1 + BATCHES + SPEC
tokens = 26 * BLOCK  # ~120K, the prod mean prompt
for label, chunk, in_flight, retention in (
    ("prod: chunk 4608 (MNBT 7168 aligned), async, retention 4608", 4608, True, 4608),
    ("chunk 4608, async, no retention", 4608, True, None),
    ("chunk 4608, synchronous", 4608, False, 4608),
    ("sub-block chunk 1152, async", 1152, True, 4608),
):
    mgr, pool = make()
    held, live_ok = replay(mgr, "r", tokens, chunk, in_flight, retention)
    peak = max(held)
    check(live_ok, f"[{label}] live states never released")
    if MODE == "stock" and chunk >= BLOCK and in_flight:
        check(peak > bound + 10, f"[{label}] stock leaks superseded state blocks (peak {peak} > {bound}+10)")
    elif MODE == "patched":
        check(peak <= bound, f"[{label}] resident state blocks <= 1+batches+spec = {bound} (peak {peak})")
    else:
        print(f"  info [{label}] peak {peak}")
    mgr.free("r")
    check(referenced(pool) == 0, f"[{label}] free returns every block")
    held2, live2 = replay(mgr, "r", 6 * BLOCK, chunk, in_flight, retention)
    if MODE == "patched":
        check(max(held2) <= bound and live2, f"[{label}] reused request id stays bounded")
    mgr.free("r")
    check(referenced(pool) == 0, f"[{label}] reused request frees everything")

if MODE == "patched":
    from vllm.v1.kv_cache_interface import MambaSpec as S
    s = S(block_size=BLOCK, shapes=((4,),), dtypes=(torch.float32,), mamba_cache_mode="align", num_speculative_blocks=SPEC)
    cfg = lambda b: types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode="align"),
                                          max_concurrent_batches=b, model_config=types.SimpleNamespace(max_model_len=1 << 20))
    check(s.max_memory_usage_bytes(cfg(2)) == s.page_size_bytes * 10 and s.max_memory_usage_bytes(cfg(1)) == s.page_size_bytes * 9,
          "reservation = 1 + batches + spec (+checkpoint) pages")

print(f"{len(FAIL)} failure(s)" if FAIL else "OK")
sys.exit(1 if FAIL else 0)
