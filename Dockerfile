# GLM-5.3-Flash (EXL3 weights) on stock upstream vLLM nightly, SM121 (DGX Spark / GB10).
#
# Everything here is a Python overlay on the official image plus one CUDA extension
# build (ExLlamaV3 + the E2 fat-expert GEMM from MiaAI Lab, extended with a two-source variant and an
# atomicAdd scatter) plus MiaAI's E3 grouped-MoE module. No vLLM C++ is rebuilt.
#
# Pins (what was benchmarked in RESULTS.md):
#   VLLM_IMAGE       vllm/vllm-openai:nightly @ the digest below (vLLM 0.28.1rc1.dev388+g8a728663c,
#                    FlashInfer 0.6.18). A newer nightly may or may not still need every patch.
#   EXLLAMAV3_COMMIT MiaAI-Lab/exllamav3 (MIT fork of turboderp's ExLlamaV3, version 1.4.2).
ARG VLLM_IMAGE=vllm/vllm-openai@sha256:f5df5cc3302b5f404848c4eca88d7bf7ed5226e151c056da22816d7734644d67
FROM ${VLLM_IMAGE}

ARG EXLLAMAV3_REPO=https://github.com/MiaAI-Lab/exllamav3
ARG EXLLAMAV3_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
ENV TORCH_CUDA_ARCH_LIST=12.1a
ARG MAX_JOBS=3

# --- ExLlamaV3 extension (+ E2 fat-expert kernel graft) ---------------------------------
COPY exl3-fat-kernel /opt/exl3-fat-kernel
RUN set -eux; \
    mkdir -p /opt/exllamav3-src; \
    curl -fsSL "${EXLLAMAV3_REPO}/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /opt/exllamav3-src --strip-components=1; \
    python3 /opt/exl3-fat-kernel/patch_exl3_fat_kernel.py \
      /opt/exllamav3-src/exllamav3/exllamav3_ext /opt/exl3-fat-kernel; \
    CPATH=$(python3 -c "import glob;print(':'.join(glob.glob('/usr/local/lib/python3.12/dist-packages/nvidia/*/include')))") \
      MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation --no-deps /opt/exllamav3-src; \
    pip install -q marisa-trie; \
    python3 -c "import torch, exllamav3_ext as e; assert hasattr(e, 'exl3_fat_gemm') and hasattr(e, 'exl3_fat_gemm_scatter') and hasattr(e, 'exl3_fat_gemm2') and e.exl3_fat_scatter_atomic(), dir(e); print('exllamav3_ext OK (fat_gemm=yes gemm2=yes atomic=yes)')"

# --- E3 grouped fat-expert MoE (MiaAI Lab, AGPL-3.0-or-later, 2026-09-07) as an additive module ------
# Built by their own script (WITHOUT --use_fast_math) inside a copy of the installed extension tree;
# exllamav3_ext itself is not recompiled. EXL3_FAT_GROUPED=1 in run.sh selects it (fails closed if absent).
COPY overlay/e3 /opt/glm53/e3
RUN set -eux; MAX_JOBS=2 python3 /opt/glm53/e3/build_exl3_fat_moe_ext.py --src /opt/glm53/e3 --out /tmp/e3build \
      --install /usr/local/lib/python3.12/dist-packages; \
    python3 -c "import torch, exl3_fat_moe_ext as m; print('exl3_fat_moe_ext OK tile_gu', m.exl3_fat_moe_tile_rows_gateup(), 'tile_dn', m.exl3_fat_moe_tile_rows_down())"; \
    rm -rf /tmp/e3build

# --- Python overlay -------------------------------------------------------------------
# exl3.py: the EXL3 quantization plugin (MiaAI Lab kit + our dense-overlay/TP work), registered
# below as a first-class vLLM quantization method.
COPY overlay/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
# SM120 sparse-MLA backend from the MiaAI kit (NoPE handled inside the impl) + our top-k split.
COPY overlay/flashinfer_mla_sparse_sm120.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
COPY overlay/patch_*.py /opt/patches/
RUN set -eux; \
    for p in patch_quant_registry patch_model_overrides patch_glm5next patch_mla_pad_rope \
             patch_glm5next_eagle3 patch_dflash_kv_auto patch_kv_drafter_group \
             patch_kpool_topk_fallback patch_dflash_exl3_kv; do python3 /opt/patches/$p.py; done; \
    python3 -c "from vllm.model_executor.layers.quantization import get_quantization_config as g; print('registry:', g('exl3').__name__)"

# --- Prefix-cache fixes + indexer workspace right-sizing (2026-09-06/07, see README "Prefix caching") -----
# Applied in place; patched files are the same ones run.sh used to bind-mount during the campaign.
COPY overlay/apc /opt/patches/apc
COPY overlay/rightsize /opt/patches/rightsize
RUN set -eux; V=/usr/local/lib/python3.12/dist-packages/vllm; \
    python3 /opt/patches/apc/patch_scheduler_mamba_align.py $V/v1/core/sched/scheduler.py $V/v1/core/sched/scheduler.py; \
    python3 /opt/patches/apc/patch_coordinator_swa_partial.py $V/v1/core/kv_cache_coordinator.py $V/v1/core/kv_cache_coordinator.py; \
    GLM53_INDEXER_BACKEND_PY=$V/v1/attention/backends/mla/indexer.py python3 /opt/patches/rightsize/patch_indexer_workspace.py; \
    find $V/v1 -name "__pycache__" -type d -exec rm -rf {} + ; \
    python3 -c "import ast,pathlib; [ast.parse(pathlib.Path(f).read_text()) for f in ['$V/v1/core/sched/scheduler.py','$V/v1/core/kv_cache_coordinator.py','$V/v1/attention/backends/mla/indexer.py']]; print('apc/rightsize patches OK')"
COPY chat_template.jinja /opt/chat_template.jinja
