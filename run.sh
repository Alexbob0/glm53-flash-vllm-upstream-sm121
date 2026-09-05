#!/bin/bash
# GLM-5.3-Flash EXL3 on upstream vLLM nightly — TP=2 across two DGX Sparks (mp backend).
#
# Usage:  ./run.sh worker   (on the second node, BEFORE the head)
#         ./run.sh head     (on the first node)
#         ./warmup.sh       (on the head, after /health — see PITFALLS.md "first-request hang")
#
# Required environment (export or edit below):
#   HF_CACHE      host path of the HuggingFace cache holding the two snapshots (both nodes)
#   MODEL_SNAP    snapshot dir of the EXL3 target pack, relative to $HF_CACHE/hub
#   DRAFT_SNAP    snapshot dir of the DFlash2 draft (BF16 incoai, or an EXL3 quant of it)
#   HEAD_IP       fabric IP of the head (RoCE /30)      NCCL_IF  socket interface   NCCL_HCA  IB HCA list
set -euo pipefail
ROLE="${1:?role: head|worker}"
IMG="${IMG:-glm53-upstream:latest}"
HF_CACHE="${HF_CACHE:-$HOME/hf}"
MODEL_SNAP="${MODEL_SNAP:?e.g. models--local--glm53-dense-K6/snapshots/<rev>}"
DRAFT_SNAP="${DRAFT_SNAP:?e.g. models--incoai--GLM-5.3-Flash-DFlash2/snapshots/<rev>}"
HEAD_IP="${HEAD_IP:?fabric IP of the head, e.g. 10.0.0.1}"
NCCL_IF="${NCCL_IF:?e.g. enp1s0f0np0}"
NCCL_HCA="${NCCL_HCA:?e.g. rocep1s0f0,roceP2p1s0f0}"
PORT="${PORT:-8888}"
SPEC="${SPEC:-dflash}"          # dflash (DFlash2 draft) | none
K="${K:-7}"
MAX_LEN="${MAX_LEN:-1000000}"   # 1M needs GMU>=0.87 on 2x GB10; 0.80 is enough up to ~640K
GMU="${GMU:-0.87}"
MODEL=/root/.cache/huggingface/hub/${MODEL_SNAP}
DRAFT=/root/.cache/huggingface/hub/${DRAFT_SNAP}

ENVS=(-e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF"
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_NO_USAGE_STATS=1
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
  # E2 fat-expert tier (needs the grafted exl3_fat_gemm): +10% prefill vs the legacy tier.
  -e EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}" -e EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"
  -e EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-128}" -e EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
  # Wrapper-level RoPE pad (alternative to the backend-level pad; keep 0 with the shipped backend).
  -e GLM53_PAD_ROPE="${PAD_ROPE:-0}"
  # Persistent top-k oversubscribes GB10 past ~600K context (see patch_kpool_topk_fallback.py).
  -e GLM53_PERSISTENT_TOPK_MAX_LEN="${PERSISTENT_TOPK_MAX_LEN:-600000}"
  # Breakable CUDA graphs (nightly default on) deadlocked 3 boots/10 inside KDA eager-break capture
  # (py-spy: gather_initial_states / l2norm_fwd launch on both ranks). Off unless BREAKABLE=1.
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${BREAKABLE:-0}"
  -e TORCH_CUDA_ARCH_LIST=12.1a)
[ -n "${NCCL_IB_GID_INDEX:-}" ] && ENVS+=(-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX")

ARGS=(serve "$MODEL"
  --served-model-name GLM-5.3-Flash-EXL3 ${EXTRA_ALIAS:-}
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size 2 --nnodes 2 --master-addr "$HEAD_IP" --master-port "${MASTER_PORT:-29811}"
  --distributed-executor-backend mp
  --max-model-len "$MAX_LEN" --max-num-seqs "${SEQS:-6}"
  --max-num-batched-tokens "${MNBT:-7168}"
  --gpu-memory-utilization "$GMU"
  --kv-cache-dtype fp8                      # canonicalized to fp8_ds_mla by the SM120 backend
  --attention-config "{\"sparse_mla_force_mqa\":true}"   # dense MHA prefill is unavailable at head 320
  --trust-remote-code --enable-prefix-caching
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template /opt/chat_template.jinja
  --limit-mm-per-prompt "{\"image\":4,\"video\":1}" --skip-mm-profiling)
# NEVER add --language-model-only: it switches to Glm5NextForCausalLM and the module prefixes
# no longer match the EXL3 non_routed keys (language_model.model.layers.*).
[ "$SPEC" = dflash ] && ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K}")
if [ "$ROLE" = head ]; then
  ARGS+=(--node-rank 0); NAME=glm53-up-head
else
  ARGS+=(--node-rank 1 --headless); NAME=glm53-up-worker
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --name "$NAME" --gpus all --network host --ipc host \
  --device /dev/infiniband --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$HF_CACHE":/root/.cache/huggingface \
  "${ENVS[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}"
