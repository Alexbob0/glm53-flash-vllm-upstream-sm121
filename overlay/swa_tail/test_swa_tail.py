#!/usr/bin/env python3
"""Real-vLLM test for the SWA_TAIL overlay (runs inside glm53-upstream:b4, no GPU).

Turn 1 prefills a prompt through vLLM's SlidingWindowManager + BlockPool (drafter geometry: block 1152,
window 2048, hash unit 64, retention 4608), decodes a few tokens, finishes. Turn 2 = same prefix + more.
  stock   (GLM53_SWA_TAIL=0): hit is 1152-aligned  -> expect floor(tail/1152)*1152
  patched (GLM53_SWA_TAIL=1): hit reaches the 64-aligned prompt tail, then the partial block is CoW'd.

    docker run --rm --entrypoint python3 -e GLM53_SWA_TAIL=1 -v $PWD:/t:ro \
      -v $PWD/single_type_kv_cache_manager.py:<vllm>/v1/core/single_type_kv_cache_manager.py:ro \
      glm53-upstream:b4 /t/test_swa_tail.py
"""
import os
import sys
import types

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

PATCHED = os.environ.get("GLM53_SWA_TAIL", "0") == "1"
BS, WIN, HASH, RET = 1152, 2048, 64, 4608
FAIL = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAIL.append(label)


def req(rid, n, prefix="p"):
    return types.SimpleNamespace(request_id=rid, num_prompt_tokens=n, num_tokens=n, shared_prefix_boundary=0,
                                 block_hashes=[BlockHash(f"{prefix}-{i}".encode()) for i in range(n // HASH)])


def make():
    spec = SlidingWindowSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=WIN)
    pool = BlockPool(num_gpu_blocks=512, enable_caching=True, hash_block_size=HASH)
    mgr = SlidingWindowManager(spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0,
                               scheduler_block_size=4608)
    mgr.cache_hit_alignment_tokens = HASH
    return mgr, pool, spec


def run_turn(mgr, r, hit_blocks, hit, total, chunk=4608, decode=0):
    mgr.new_step_starts() if hasattr(mgr, "new_step_starts") else None
    mgr.get_num_blocks_to_allocate(r.request_id, min(total, max(hit, 1) + chunk), hit_blocks, hit, hit, total)
    mgr.add_local_computed_blocks(r.request_id, hit_blocks, hit, 0)
    done = hit
    while done < r.num_prompt_tokens:
        new = min(chunk - done % chunk if done % chunk else chunk, r.num_prompt_tokens - done)
        mgr.remove_skipped_blocks(r.request_id, done)
        mgr.get_num_blocks_to_allocate(r.request_id, done + new, [], done, done, done + new)
        mgr.allocate_new_blocks(r.request_id, done + new, done + new)
        mgr.cache_blocks(r, done + new, retention_interval=RET)
        done += new
    for _ in range(decode):
        mgr.remove_skipped_blocks(r.request_id, done)
        mgr.allocate_new_blocks(r.request_id, done + 1, done + 1)
        done += 1
        r.num_tokens = done
        mgr.cache_blocks(r, min(done, r.num_prompt_tokens), retention_interval=RET)


def lookup(pool, spec, r, max_len):
    return SlidingWindowManager.find_longest_cache_hit(
        block_hashes=r.block_hashes, max_length=max_len, kv_cache_group_ids=[0], block_pool=pool,
        kv_cache_spec=spec, drop_eagle_block=False, alignment_tokens=HASH)


print(f"patched={PATCHED}")
for n1, n2 in ((68252, 68390), (20409, 20490), (31514, 31700), (9216 + 100, 9216 + 300), (4700, 5000)):
    mgr, pool, spec = make()
    r1 = req("t1", n1)
    run_turn(mgr, r1, [], 0, n1, decode=40)
    mgr.free("t1")
    r2 = req("t2", n2)
    blocks, hit = lookup(pool, spec, r2, n2 - 1)
    tail = n1 // HASH * HASH
    expect = tail if PATCHED else tail // BS * BS
    # stock: 1152-aligned hit also needs its window blocks retained (4608 retention) -> may be lower
    if PATCHED:
        check(hit == expect, f"[{n1}->{n2}] hit {hit} == prompt tail {expect}")
    else:
        check(hit <= expect and hit % BS == 0, f"[{n1}->{n2}] stock hit {hit} is block-aligned <= {expect}")
    blk = blocks[0]
    check(len(blk) == -(-hit // BS), f"[{n1}->{n2}] {len(blk)} blocks for hit {hit}")
    need_first = max(0, (hit - WIN + 1) // BS)
    check(all(not b.is_null for b in blk[need_first:]), f"[{n1}->{n2}] window blocks {need_first}..{len(blk)-1} real")
    if PATCHED and hit % BS:
        src = blk[-1]
        run_turn(mgr, r2, blk, hit, n2, decode=5)
        cow = mgr.take_pending_cow_copies()
        check(len(cow) == 1 and cow[0][0] is src and cow[0][1] is not src,
              f"[{n1}->{n2}] partial tail block CoW'd ({len(cow)} copy)")
        check(mgr.req_to_blocks["t2"][hit // BS] is not src, f"[{n1}->{n2}] turn 2 writes a private block")
        pool.free_blocks([b for pair in cow for b in pair])  # what Scheduler._free_cow_retained_blocks does after the step
        mgr.free("t2")
        # the source partial block survives and still serves the same turn-1 prefix
        _, hit3 = lookup(pool, spec, req("t3", n2), n2 - 1)
        check(hit3 >= tail, f"[{n1}->{n2}] prefix still reusable after turn 2 ({hit3})")
    for rid in list(mgr.req_to_blocks):
        mgr.free(rid)
    check(sum(1 for b in pool.blocks if b.ref_cnt > 0 and not b.is_null) == 0, f"[{n1}->{n2}] all blocks released")

# unrelated prompt never hits
mgr, pool, spec = make()
run_turn(mgr, req("a", 30000, "A"), [], 0, 30000)
mgr.free("a")
_, h = lookup(pool, spec, req("b", 30100, "B"), 30099)
check(h == 0, f"unrelated prefix -> no hit ({h})")
print(f"{len(FAIL)} failure(s)" if FAIL else "OK")
sys.exit(1 if FAIL else 0)
