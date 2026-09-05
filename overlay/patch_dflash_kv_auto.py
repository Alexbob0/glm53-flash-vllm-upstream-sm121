"""Give the DFlash/DFlash2 draft's dense attention a bf16 KV cache config.

The MLA canonicalization writes fp8_ds_mla into the global cache_config; the draft's
Qwen3-style attention (head 128) inherits it and no dense backend supports that
dtype ("No valid attention backend found"). --kv-cache-dtype-skip-layers by index
cannot be used: DFlash2 draft layers are indexed 0-4 in the base class and would
collide with target layer 3 (MLA). So the draft gets a copy of cache_config with
cache_dtype="auto".
"""
import pathlib
import sys

p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py")
s = p.read_text()
if "[draft-kv-auto]" in s:
    print("qwen3_dflash.py: deja patche")
    sys.exit(0)
old = (
    "        self.sliding_window = sliding_window\n"
    "        self.attn = Attention(\n"
)
new = (
    "        self.sliding_window = sliding_window\n"
    "        # [draft-kv-auto] ds_mla packed dtypes are MLA-only; draft uses bf16 KV.\n"
    "        if cache_config is not None and str(cache_config.cache_dtype).endswith(\"_ds_mla\"):\n"
    "            import copy as _copy\n"
    "            cache_config = _copy.copy(cache_config)\n"
    "            cache_config.cache_dtype = \"auto\"\n"
    "        self.attn = Attention(\n"
)
assert s.count(old) == 1, "qwen3_dflash.py: site Attention introuvable/ambigu"
p.write_text(s.replace(old, new))
print("qwen3_dflash.py: patche (draft-kv-auto)")
