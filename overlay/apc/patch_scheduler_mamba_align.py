#!/usr/bin/env python3
"""APC (06/09): _mamba_block_aligned_split used cache_config.block_size (1152 here, set by the
drafter group) while the KDA/mamba groups have block 4608 (= scheduler lcm block). Chunk ends were
therefore not aligned on mamba blocks and KDA states were never cacheable. Align on the real mamba
block size. Usage: patch <in> <out>."""
import sys
from pathlib import Path
src = Path(sys.argv[1]).read_text()
a1 = "        block_size = self.cache_config.block_size\n        # The last block-aligned position whose state can be cached. With\n"
r1 = "        # [apc-align] align on the mamba group block (may exceed cache_config.block_size)\n        block_size = self._mamba_align_block_size\n        # The last block-aligned position whose state can be cached. With\n"
assert src.count(a1) == 1, "anchor1 drift"
src = src.replace(a1, r1)
a2 = "        self.need_mamba_block_aligned_split = (\n            self.has_mamba_layers and self.cache_config.mamba_cache_mode == \"align\"\n        )\n"
r2 = a2 + "        # [apc-align] real mamba block size for chunk alignment (cache_config.block_size\n        # can be the smaller drafter block here).\n        self._mamba_align_block_size = max(\n            (\n                group.kv_cache_spec.block_size\n                for group in kv_cache_config.kv_cache_groups\n                if isinstance(group.kv_cache_spec, MambaSpec)\n            ),\n            default=self.cache_config.block_size,\n        )\n        if self._mamba_align_block_size != self.cache_config.block_size:\n            logger.info(\n                \"[apc-align] mamba chunk alignment %d (cache_config.block_size=%d)\",\n                self._mamba_align_block_size,\n                self.cache_config.block_size,\n            )\n"
assert src.count(a2) == 1, "anchor2 drift"
src = src.replace(a2, r2)
Path(sys.argv[2]).write_text(src)
print("scheduler patched")
