#!/usr/bin/env python3
"""[swa-tail] (24/09) Fine-grained prompt-tail prefix hits for the DFlash2 drafter's SlidingWindow group.

Why: multi-turn hits were capped at a 4608 multiple. Per-group lookups (APC debug, 06/09) give MLA and KDA a
hit at the prompt tail (64-aligned: MLA via FullAttentionManager partial tails, KDA via the mamba "align"
partial-tail state), but SlidingWindowManager only matches whole 1152-token blocks. The reconciled hit is the
largest length every group supports -> the last 4608 boundary (e.g. 68 390-token turn: tail 68 224, drafter
67 968, result 64 512 -> ~3.9K tokens re-prefilled per turn, ~2.5 s).

Fix (gated by GLM53_SWA_TAIL=1): (1) cache_blocks registers the drafter's partial block at the prompt's last
hash boundary, exactly like FullAttentionManager._cache_partial_tail_block; (2) find_longest_cache_hit, after
the stock block scan, probes fine-grained boundaries above it for such a partial entry whose preceding window
blocks are all cached. The base-class partial-hit CoW (reserve + copy_kv_cache_blocks_inplace) is generic.
Correctness: the drafter only proposes; the target verifies, so a stale drafter KV can only cost acceptance.
Usage: patch_swa_tail.py <in.py> <out.py>
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
MARK = "# [swa-tail]"
if MARK in src:
    Path(sys.argv[2]).write_text(src)
    print("already patched")
    sys.exit(0)

a0 = "logger = init_logger(__name__)\n"
assert src.count(a0) == 1, "anchor0 drift"
src = src.replace(a0, a0 + 'import os as _swa_os  # [swa-tail]\n_SWA_TAIL = _swa_os.environ.get("GLM53_SWA_TAIL", "0") == "1"\n')

a1 = """        assert pcp_world_size == 1, "PCP not support sliding window attn now."
        # Sliding-window cache hits must stay at the group's physical block
        # granularity."""
assert src.count(a1) == 1, "anchor1 drift"
src = src.replace(a1, a1.replace("        # Sliding-window cache hits",
                                 "        fine_hashes = block_hashes  # [swa-tail] fine-grained view, before resolve\n"
                                 "        # Sliding-window cache hits"))

a2 = """        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
"""
assert src.count(a2) == 1, "anchor2 drift"
r2 = """        hit_length = len(computed_blocks[0]) * block_size
        # [swa-tail] extend into a cached prompt-tail partial block when finer hashes exist.
        if (
            _SWA_TAIL
            and not drop_eagle_block
            and hit_length < max_length
            and alignment_tokens == block_pool.hash_block_size
            and alignment_tokens < block_size
            and block_size % alignment_tokens == 0
        ):
            ext = cls._swa_tail_extend(
                fine_hashes, block_hashes, max_length, hit_length,
                kv_cache_group_ids, block_pool, kv_cache_spec, alignment_tokens,
            )
            if ext is not None:
                return ext
        return computed_blocks, hit_length

    @classmethod
    def _swa_tail_extend(
        cls,
        fine_hashes,
        block_view_hashes,
        max_length: int,
        floor_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        unit: int,
    ):
        \"\"\"[swa-tail] Longest hit in (floor_length, max_length] ending on a cached partial tail.\"\"\"
        block_size = kv_cache_spec.block_size
        window = kv_cache_spec.sliding_window
        hi = min(max_length // unit, len(fine_hashes))
        lo = floor_length // unit
        for fine_idx in range(hi - 1, lo - 1, -1):
            num_tokens = (fine_idx + 1) * unit
            if num_tokens % block_size == 0:
                continue  # whole-block boundaries belong to the stock scan
            tail = block_pool.get_cached_block(fine_hashes[fine_idx], kv_cache_group_ids)
            if not tail:
                continue
            tail_idx = num_tokens // block_size
            # Blocks covering the window [num_tokens - window + 1, num_tokens) must be cached.
            first = max(0, (num_tokens - window + 1) // block_size)
            window_blocks = []
            for j in range(first, tail_idx):
                cached = block_pool.get_cached_block(block_view_hashes[j], kv_cache_group_ids)
                if not cached:
                    window_blocks = None
                    break
                window_blocks.append(cached)
            if window_blocks is None:
                continue
            computed = tuple(
                [block_pool.null_block] * first
                + [cached[g] for cached in window_blocks]
                + [tail[g]]
                for g in range(len(kv_cache_group_ids))
            )
            return computed, num_tokens
        return None

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
"""
src = src.replace(a2, r2)

a3 = """    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        \"\"\"
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.
"""
assert src.count(a3) == 1, "anchor3 drift"
r3 = """    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
        # [swa-tail] also register the prompt-tail partial block (same rule as full attention).
        if _SWA_TAIL and self.block_size != self.block_pool.hash_block_size:
            FullAttentionManager._cache_partial_tail_block(self, request, num_tokens)

""" + a3
src = src.replace(a3, r3)
Path(sys.argv[2]).write_text(src)
print("swa-tail patched")
