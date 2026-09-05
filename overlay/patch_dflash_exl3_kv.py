"""EXL3-quantized DFlash2 draft on the nightly: fused context-KV reconstruction.

precompute_and_store_context_kv runs one fused GEMM over the K/V rows of every
draft layer's qkv_proj (`qkv_proj.weight[q_size:]`). With EXL3 there is no BF16
.weight: reconstruct those rows once with identity forwards on the k and v shards
(<=128-row slices: fast trellis path; >144 rows hits reconstruct_hgemm with a CPU
unpack that deadlocked both ranks during dummy_run). At load time the LinearEXL3
objects do not exist yet (built by process_weights_after_loading): defer, and let
the lazy path call _build_fused_kv_buffers on first use.
"""
import pathlib
import sys

p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py")
s = p.read_text()
if "[draft-exl3-kv]" in s:
    print("qwen3_dflash.py: deja patche (exl3 kv)")
    sys.exit(0)

edits = [
    (
        "        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]\n"
        "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
        "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n",
        "        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]\n"
        "        # [draft-exl3-kv] EXL3 qkv: no BF16 .weight — rebuild the K/V rows\n"
        "        # once (identity forward on the k and v shards). The draft's decode\n"
        "        # path stays quantized.\n"
        "        def _kv_rows(a):\n"
        "            w = getattr(a.qkv_proj, \"weight\", None)\n"
        "            if w is not None and w.shape[0]:\n"
        "                return w[a.q_size :]\n"
        "            linears = getattr(a.qkv_proj, \"_exl3_linears\", None)\n"
        "            if not linears:\n"
        "                return None  # LinearEXL3 not built yet: defer\n"
        "            dev = (\n"
        "                a.qkv_proj.mask_dtype_device[2]\n"
        "                if hasattr(a.qkv_proj, \"mask_dtype_device\")\n"
        "                else self.hidden_norm.weight.device\n"
        "            )\n"
        "            in_f = linears[1].in_features\n"
        "            eye = torch.eye(in_f, dtype=torch.float16, device=dev)\n"
        "            rows = []\n"
        "            for li in (1, 2):  # k, v shards — same order as weight[q_size:]\n"
        "                outs = [\n"
        "                    linears[li].forward(eye[i : i + 128], {}, out_dtype=torch.float16)\n"
        "                    for i in range(0, in_f, 128)\n"
        "                ]\n"
        "                rows.append(torch.cat(outs, dim=0).t().contiguous())\n"
        "            return torch.cat(rows, dim=0).to(self.hidden_norm.weight.dtype)\n"
        "\n"
        "        kv_weights = [_kv_rows(a) for a in layers_attn]\n"
        "        if any(w is None for w in kv_weights):\n"
        "            self._kv_buffers_deferred = True\n"
        "            return\n"
        "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n",
    ),
    (
        "        self._k_norm_weights = torch.stack(\n"
        "            [a.k_norm.weight.data for a in layers_attn], dim=0\n"
        "        ).contiguous()\n"
        "\n"
        "    def _build_fused_kv_buffers(self) -> None:\n",
        "        self._k_norm_weights = torch.stack(\n"
        "            [a.k_norm.weight.data for a in layers_attn], dim=0\n"
        "        ).contiguous()\n"
        "        self._kv_buffers_deferred = False\n"
        "\n"
        "    def _build_fused_kv_buffers(self) -> None:\n",
    ),
    (
        "        self._build_context_kv_buffers(layers_attn, has_bias)\n"
        "\n"
        "        # RoPE parameters\n",
        "        self._build_context_kv_buffers(layers_attn, has_bias)\n"
        "        if getattr(self, \"_kv_buffers_deferred\", False):\n"
        "            # [draft-exl3-kv] set nothing (especially not _num_attn_layers, the\n"
        "            # lazy-path sentinel) while the LinearEXL3 objects are missing.\n"
        "            for attr in (\"_num_attn_layers\", \"_fused_kv_weight\"):\n"
        "                if hasattr(self, attr):\n"
        "                    delattr(self, attr)\n"
        "            return\n"
        "\n"
        "        # RoPE parameters\n",
    ),
]
for old, new in edits:
    assert s.count(old) == 1, f"qwen3_dflash.py: contexte introuvable/ambigu ({s.count(old)}):\n{old}"
    s = s.replace(old, new)
p.write_text(s)
print("qwen3_dflash.py: patche (draft-exl3-kv)")
