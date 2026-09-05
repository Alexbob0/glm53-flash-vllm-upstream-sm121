"""Wrapper-level alternative for NoPE on the SM120 packed backend (vllm#53963).

Off by default (GLM53_PAD_ROPE=0): the shipped flashinfer_mla_sparse_sm120.py pads
the 64 RoPE dims inside the impl instead. When enabled, the MLA wrapper pads q with
64 zero dims and feeds a zero k_pe so only the MLAAttention layer sees rope=64
(cache 656 B/token, stock kernels); the config, the EXL3 projections, the indexer
and the softmax scaling (256^-0.5) stay NoPE. Zero q_pe/k_pe leave the scores
unchanged; cost is +12 % KV bytes on the 11 MLA layers.
"""
import pathlib
import sys

p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla.py")
s = p.read_text()
if "[pad-rope]" in s:
    print("mla.py: deja patche")
    sys.exit(0)

edits = [
    (
        "        self.qk_rope_head_dim = qk_rope_head_dim\n"
        "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n",
        "        self.qk_rope_head_dim = qk_rope_head_dim\n"
        "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n"
        "        # [pad-rope] NoPE -> DeepSeek shape (rope 64) for the MLA layer only.\n"
        "        import os as _os\n"
        "        self._rope_pad = (\n"
        "            64\n"
        "            if qk_rope_head_dim == 0 and _os.environ.get(\"GLM53_PAD_ROPE\", \"1\") == \"1\"\n"
        "            else 0\n"
        "        )\n",
    ),
    (
        "            qk_rope_head_dim=self.qk_rope_head_dim,\n"
        "            v_head_dim=self.v_head_dim,\n"
        "            q_lora_rank=self.q_lora_rank,\n"
        "            kv_lora_rank=self.kv_lora_rank,\n"
        "            cache_config=cache_config,\n",
        "            qk_rope_head_dim=self.qk_rope_head_dim + self._rope_pad,  # [pad-rope]\n"
        "            v_head_dim=self.v_head_dim,\n"
        "            q_lora_rank=self.q_lora_rank,\n"
        "            kv_lora_rank=self.kv_lora_rank,\n"
        "            cache_config=cache_config,\n",
    ),
    (
        "        attn_out = self.mla_attn(\n"
        "            q,\n"
        "            kv_c_normed,\n"
        "            k_pe,\n",
        "        if self._rope_pad:  # [pad-rope] q_pe = 0, k_pe = 0 (scores unchanged)\n"
        "            q = torch.nn.functional.pad(q, (0, self._rope_pad))\n"
        "            k_pe = q.new_zeros((k_pe.shape[0], 1, self._rope_pad))\n"
        "        attn_out = self.mla_attn(\n"
        "            q,\n"
        "            kv_c_normed,\n"
        "            k_pe,\n",
    ),
]
for old, new in edits:
    assert s.count(old) == 1, f"mla.py: contexte introuvable/ambigu:\n{old}"
    s = s.replace(old, new)
p.write_text(s)
print("mla.py: patche (pad-rope)")
