# Cooperative MoE for GLM-5.3-Flash TP2

A fixed-shape EXL3 MoE extension for dual-Spark GLM-5.3-Flash serving. It uses
the upstream two-stage cooperative kernel for **decode-sized** fused `exl3_moe`
only (tiny M). Prefill keeps the shipped E3 grouped path. Activation is
explicit; this change does not modify the default image, launcher, or overlay.

**To try it:** follow the [complete two-node opt-in guide](../../docs/cooperative-moe-quickstart.md).
A standalone `GLM53_COOPERATIVE_MOE=1` does not activate or install the
extension.

Do **not** copy the DS4.1 `cooperative_moe.so`. That binary is compile-time
locked to DeepSeek-V4.1 (H=5120, I=1152, top-k 6, K2/K3 mul1) and will never
fire here.

Current README decode baselines (stock fused EXL3, sparkDash): prose ×1 **32.1**
tok/s / ×2 agg **41.2**; structured ×1 **62.9** tok/s. Cooperative geometry 1
(A-wide/B-wide) measured **37.4** prose / **80.0** structured on this kit; see
[the report](../../docs/cooperative-moe.md) and the
[operator handoff](../../docs/cooperative-moe-handoff.md). The packaged GPU gate
passed on this head (peak-normalized vs stock fused, E3 not replaced).

## Supported configuration

| Component | Requirement |
|---|---|
| Hardware | Two SM121a Sparks, tensor parallelism 2 (not EP) |
| Expert shape | Hidden 4096, local intermediate **1024** (packed 2048 / TP2), top-k 8 |
| Quantization | Uniform K4 MCG (codebook cb=1), no mul1 |
| Physical decode batch | 1–32 rows (DFlash2 k=7 → 8 rows/seq; ×4 seqs → 32) |
| Prefill / fat experts | Stock E3 grouped (`EXL3_FAT_GROUPED=1`); this path is not replaced |

TP3/EP (local I=2048, 96 experts) and unsupported shapes use stock dispatch.
Scratch is allocated after weight loading and before graph capture. The adapter
never retries stock after a partially launched CUDA operation.

## Build

Requirements: an ExLlamaV3 checkout containing commit
`02aef45cd681b960a00afcd0749a4ab99e6c1bfe` (headers; kernel origin is
`58d4d732`), the ARM64 GLM recipe image with CUDA 13 / nvcc, and an empty
output directory. The recipe image has nvcc but not git: archive the pin on
the host, then compile in the container.

```sh
bash extensions/cooperative_moe/build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY
```

Outputs are `cooperative_moe.so`, `runtime.py`, and the build log. The adapter
and `prepare_profile.py` pin the binary digest. Different toolchains or
compiler paths may change it. Do not bypass the check.

## Select a profile

Stage `cooperative_moe.so` and `runtime.py` at the same container-visible path
on **both ranks**. Generate a separate overlay:

```sh
python3 extensions/cooperative_moe/prepare_profile.py \
  --stock overlay/exl3.py \
  --artifacts VERIFIED_ARTIFACT_DIRECTORY \
  --runtime-directory /root/.cache/vllm/cooperative_moe \
  --output NEW_COOPERATIVE_OVERLAY.py
```

Select the generated file with `EXL3_OVERLAY_HOST`. Keep E3 on. The generated
profile calls `install(..., enabled=True)`. Direct integrations may use
`GLM53_COOPERATIVE_MOE=1` when calling `install()`.

## Tests

```sh
python3 extensions/cooperative_moe/test_dispatch.py
python3 extensions/cooperative_moe/test_profile.py
python3 -O extensions/cooperative_moe/test_optimized_init.py
bash -n extensions/cooperative_moe/build.sh
```

`test_cuda_integration.py` needs the selected overlay, `test_exl3_overlay.py`
from `/opt/glm53`, and `GLM53_COOP_MAINTENANCE_TEST=1`. Numerical screening is
peak-normalized vs stock fused, not bit-exact.

## Attribution

The native implementation derives from Turboderp's
[two-stage cooperative MoE kernel](https://github.com/turboderp-org/exllamav3/commit/58d4d7322a1b3bd70aae8412487b21cc5e205cf4)
(MIT, retained in `native/LICENSE.exllamav3`). GLM specialization is H/I/topk/rows,
K4+MCG (cb=1), and SM121 at the kernel entry points; device arithmetic is
otherwise unchanged.
