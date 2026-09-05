"""Indexer top-k fallback beyond ~600K of context on GB10.

persistent_topk (csrc topk.cu) oversubscribes the 48 SMs when max_seq_len
approaches 1M (85 CTAs) and its FilteredTopK fallback needs 128 KB of smem (GB10:
99 KB) -> RuntimeError during profiling. top_k_per_row_decode sits right below in
the kpool indexer: use it when max_model_len exceeds
GLM53_PERSISTENT_TOPK_MAX_LEN (default 600000; 524288 verified OK, 1000000 fails).
Static threshold (config-level), not per batch, to stay consistent under CUDA graphs.
Measured cost: -1 % decode.
"""
import pathlib
import sys

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
)
s = p.read_text()
if "[topk-fallback]" in s:
    print("sparse_attn_indexer_kpool.py: deja patche")
    sys.exit(0)
old = (
    "            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n"
    "\n"
    "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):\n"
)
new = (
    "            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]\n"
    "\n"
    "        # [topk-fallback] persistent_topk oversubscribes GB10 (48 SMs, 99 KB smem)\n"
    "        # past ~600K context; use top_k_per_row_decode beyond that.\n"
    "        _persistent_ok = max_model_len <= int(\n"
    "            __import__(\"os\").environ.get(\"GLM53_PERSISTENT_TOPK_MAX_LEN\", \"600000\")\n"
    "        )\n"
    "        if current_platform.is_cuda() and _persistent_ok and select_k in (512, 1024, 2048):\n"
)
assert s.count(old) == 1, f"kpool: site persistent_topk introuvable/ambigu ({s.count(old)})"
p.write_text(s.replace(old, new))
print("sparse_attn_indexer_kpool.py: patche (topk-fallback)")
