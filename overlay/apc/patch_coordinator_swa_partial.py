#!/usr/bin/env python3
"""APC (06/09): keep fine-grained prefix-cache hits when the only block-aligned manager is
SlidingWindowManager (the DFlash2 drafter group). That manager already resolves fine hashes
to its block view and skips unaligned candidates, so the hybrid hit loop converges on a
1152-aligned length — exactly where the single KDA prefill state is cached. Usage: patch <in> <out>."""
import sys
from pathlib import Path
src = Path(sys.argv[1]).read_text()
anchor = """                if group.kv_cache_spec.prefix_cacheable
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }"""
repl = """                if group.kv_cache_spec.prefix_cacheable
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
                # [apc-swa] SlidingWindowManager resolves fine hashes to its block
                # view and enforces alignment itself; do not let it veto partial hits.
                and type(manager).__name__ != "SlidingWindowManager"
            }"""
assert src.count(anchor) == 1, "anchor drift"
Path(sys.argv[2]).write_text(src.replace(anchor, repl))
print("patched")

# --- [apc-dbg] per-request hit diagnostics (env GLM53_APC_DEBUG=1) ---
src2 = Path(sys.argv[2]).read_text()
anchor2 = "        num_uncached_common_prefix_tokens = longest_hit_length - hit_length\n"
dbg = '''        if _APC_DEBUG:
            indep = {}
            for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
                try:
                    indep[group_ids[0]] = manager_cls.find_longest_cache_hit(
                        block_hashes=block_hashes, max_length=max_cache_hit_length,
                        kv_cache_group_ids=group_ids, block_pool=self.block_pool,
                        kv_cache_spec=spec, drop_eagle_block=use_eagle,
                        alignment_tokens=self._cache_hit_alignment_tokens,
                        dcp_world_size=self.single_type_managers[group_ids[0]].dcp_world_size,
                        pcp_world_size=self.single_type_managers[group_ids[0]].pcp_world_size,
                    )[1]
                except Exception as exc:  # noqa: BLE001
                    indep[group_ids[0]] = f"ERR {type(exc).__name__}: {exc}"
            logger.info(
                "[apc-dbg] max=%d hit=%d by_group=%s indep=%s groups=%s align=%d hash_bs=%d partial=%s eagle_ids=%s",
                max_cache_hit_length, hit_length, hit_length_by_group, indep,
                [(g.group_ids, type(g.spec).__name__, self.single_type_managers[g.group_ids[0]].block_size, g.use_eagle) for g in self.attention_groups],
                self._cache_hit_alignment_tokens, self.hash_block_size, self.enable_partial_hash_hits, sorted(self.eagle_group_ids),
            )
'''
assert src2.count(anchor2) == 1, "anchor2 drift"
src2 = src2.replace(anchor2, dbg + anchor2)
head_anchor = "logger = init_logger(__name__)\n"
assert src2.count(head_anchor) == 1, "logger anchor drift"
src2 = src2.replace(head_anchor, head_anchor + 'import os as _os\n_APC_DEBUG = _os.environ.get("GLM53_APC_DEBUG", "0") == "1"\n')
Path(sys.argv[2]).write_text(src2)
print("debug instrumentation added")

# --- [apc-dbg2] per-iteration trace of the hybrid reconciliation loop ---
src3 = Path(sys.argv[2]).read_text()
a3 = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:"""
r3 = """                if _APC_DEBUG:
                    logger.info("[apc-dbg2] it grp=%s %s max=%d drop=%s -> %d (curr=%d hit=%d)",
                                group_ids, type(spec).__name__, _max_length, drop_eagle_block,
                                _new_hit_length, curr_hit_length, hit_length)
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:"""
assert src3.count(a3) == 1, "anchor3 drift"
Path(sys.argv[2]).write_text(src3.replace(a3, r3))
print("loop trace added")
