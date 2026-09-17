#!/usr/bin/env python3
"""Fixed-input CUDA-event microbenchmark of stock vs cooperative fused MoE.

Captures separate CUDA graphs for stock and candidate on identical tensors.
Reports kernel-cycle time, not serving tok/s. Requires a maintenance window
and GLM53_COOP_MAINTENANCE_TEST=1. Does not modify serving.
"""

import json
import os
import statistics
import sys
import time

assert os.environ.get("GLM53_COOP_MAINTENANCE_TEST") == "1"
os.environ.update(
    EXL3_FUSED_MOE="1",
    EXL3_FAT_GROUPED="1",
    EXL3_FAT_KERNEL="0",
    EXL3_TEMP_ROWS_FUSED="32",
)
sys.path.insert(0, "/opt/glm53")
import torch
import test_exl3_overlay as tests
from vllm.model_executor.layers.quantization import exl3


def time_graph(graph, warmup=10, iters=50):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "p10_ms": samples[max(len(samples) // 10 - 1, 0)],
        "p90_ms": samples[min(int(len(samples) * 0.9), len(samples) - 1)],
        "iters": iters,
    }


def main():
    rows = int(os.environ.get("GLM53_COOP_MICRO_ROWS", "8"))
    n_exp = int(os.environ.get("GLM53_COOP_MICRO_EXPERTS", "32"))
    _, layer = tests._tiny_layer(
        torch.device("cuda"), n_exp=n_exp, hidden=4096, inter=1024
    )
    native = getattr(layer, "_glm53_coop_native", None)
    assert native is not None
    stock = exl3._glm53_coop_original_apply
    wrapper = exl3.apply_exl3_fused_moe
    torch.manual_seed(20260916)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    concentrated = os.environ.get("GLM53_COOP_MICRO_CONC", "0") == "1"
    if concentrated:
        ids = torch.randperm(n_exp, device="cuda")[:8].repeat(rows, 1)
    else:
        ids = torch.stack(
            [torch.randperm(n_exp, device="cuda")[:8] for _ in range(rows)]
        )
    weights = torch.rand(rows, 8, device="cuda").softmax(-1)

    def run():
        return exl3.apply_exl3_experts(
            x.float(), ids, weights, layer, fused=True, limit=10.0
        )

    results = {
        "rows": rows,
        "experts": n_exp,
        "geometry": getattr(exl3, "_glm53_coop_geometry", None),
        "concentrated": concentrated,
        "ts": time.time(),
    }
    # Allocate fused temps and scratch outside capture; CUDA graphs cannot
    # recover from a failed kernel in the same context.
    for impl in (stock, wrapper):
        exl3.apply_exl3_fused_moe = impl
        run()
    torch.cuda.synchronize()
    for name, impl in (("stock", stock), ("cooperative", wrapper)):
        exl3.apply_exl3_fused_moe = impl
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        results[name] = time_graph(graph)
        del graph
    stock_ms = results["stock"]["median_ms"]
    coop_ms = results["cooperative"]["median_ms"]
    results["speedup"] = stock_ms / coop_ms if coop_ms else None
    results["delta_ms"] = stock_ms - coop_ms
    print(json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
