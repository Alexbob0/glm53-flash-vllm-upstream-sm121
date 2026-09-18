#!/usr/bin/env python3
"""Bucket the kernel table of an analyze_trace.py summary into model components."""
import json, re, sys
from collections import defaultdict
RULES = [
    ('nccl', r'nccl|AllReduce|all_reduce|allgather|ReduceScatter'),
    ('mhc_tilelang', r'mhc_|tilelang|sinkhorn|hc_prenorm|hc_post'),
    ('experts_moe', r'exl3_fat|exl3_moe|cooperative|coop_|moe_|fused_moe|topk_softmax|grouped_topk|moe_align|expert'),
    ('mla_indexer', r'sparse_mla|flashinfer|mla_|fmha|paged|top_k_per_row|topKPerRow|persistent_topk|indexer|fp8_mqa|deep_gemm|deepgemm|fp8_gemm|rotary|rope|kpool|_fwht_|pools_and|_tail_'),
    ('kda_fla', r'chunk_|kda|gated_delta|delta_rule|l2norm|solve_tril|_wy_|recompute_w_u|cumsum|causal_conv|conv1d|gather_initial|merge_16x16|layer_norm_gated|ssm|mamba'),
    ('dense_gemm', r'nvjet|cutlass|gemm|hgemm|Gemm|exl3_|had_|hadamard|reconstruct|cublas|sm90_|sm120_|sm121_'),
    ('norm_act', r'rms_norm|layer_norm|layernorm|norm_|fused_add|act_and_mul|silu|gelu|softmax|sigmoid'),
    ('elementwise_copy', r'elementwise_kernel|direct_copy|vectorized|gather_kernel|index_|scatter|fill_|Fill|cat_|CatArray|copy_|memcpy|Memcpy|Memset|reduce_kernel|unrolled|arange|where|cast|masked|compare|sort|cub::|radix|scan|triton_poi|triton_red'),
]
def bucket(name):
    for tag, rx in RULES:
        if re.search(rx, name): return tag
    return 'other'
for path in sys.argv[1:]:
    s = json.load(open(path))
    tot = s['kernel_sum_ms']; b = defaultdict(lambda: [0, 0.0]); other = []
    for k in s['kernels']:
        t = bucket(k['name']); b[t][0] += k['calls']; b[t][1] += k['sum_us'] / 1000
        if t == 'other': other.append(k)
    print(f"\n== {path}\n   kernel_sum {tot:.0f} ms | union {s['kernel_union_ms']:.0f} ms | span {s['kernel_span_ms']:.0f} ms | calls {s['kernel_calls']}")
    for t, (c, ms) in sorted(b.items(), key=lambda x: -x[1][1]):
        print(f"   {t:18s} {ms:8.0f} ms  {100*ms/tot:5.1f} %  calls {c}")
    print("   -- top 45 kernels:")
    for k in s['kernels'][:45]:
        n = re.sub(r'\(.*', '', k['name'])[:110]
        print(f"   {k['sum_us']/1000:8.1f} ms {k['calls']:6d}x  [{bucket(k['name'])}] {n}")
    if other:
        print("   -- 'other' (top 15):")
        for k in other[:15]: print(f"   {k['sum_us']/1000:8.1f} ms {k['calls']:6d}x  {k['name'][:120]}")
