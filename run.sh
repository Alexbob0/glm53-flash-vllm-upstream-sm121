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
PMU="${PMU-64}"                 # --prefix-match-unit (empty = off)
RETENTION="${RETENTION-4608}"   # --prefix-cache-retention-interval (empty = upstream default 0)
ADAPTIVE_K="${ADAPTIVE_K:-0}"   # 1 = adaptive verification length (opt-in, 2026-09-12); 0 = byte-for-byte the baked scheduler
# Optional: mount a generated exl3.py over the baked one (e.g. the cooperative-MoE overlay from
# extensions/cooperative_moe/prepare_profile.py). Empty = the baked plugin. Its runtime.py + .so
# live in /root/.cache/vllm/cooperative_moe, i.e. inside the JIT cache mount (needs JIT_CACHE=1).
EXL3_OVERLAY_HOST="${EXL3_OVERLAY_HOST:-}"
HERE=$(dirname "$(readlink -f "$0")")
EXTRA_MOUNTS=()
MODEL=/root/.cache/huggingface/hub/${MODEL_SNAP}
DRAFT=/root/.cache/huggingface/hub/${DRAFT_SNAP}

ENVS=(-e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF"
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_NO_USAGE_STATS=1
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
  # E2 fat-expert tier (needs the grafted exl3_fat_gemm): +10% prefill vs the legacy tier.
  -e EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}" -e EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"
  # E3 grouped fat-expert MoE (MiaAI Lab, 2026-09-07): 3 launches per MoE layer from device-side tables.
  # Measured here on real code: 8K/32K/100K cold prefill 1,260 / 1,343 / 1,345 tok/s (E2: 890 / 940 / 960).
  # Fused cap 64 with E3 (must stay >= SEQS x (K+1) so decode is one graph-safe launch), 128 with E2.
  -e EXL3_FAT_GROUPED="${FAT_GROUPED:-1}"
  -e MAX_NUM_BATCHED_TOKENS="${MNBT:-7168}"   # E3 sizes its persistent fat-row scratch from this (x top-k)
  -e EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-$([ "${FAT_GROUPED:-1}" = 1 ] && echo 64 || echo 128)}" -e EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
  # E2 fallback tier only: fat experts round-robined over n CUDA streams (atomicAdd scatter). 4 = +14-19 %
  # cold prefill over the host loop (the fat GEMM is a flat ~200 us per expert at <=512 rows, 16-32 CTAs / 48 SMs);
  # 8 oversubscribes and loses the gain. Irrelevant when E3 is active.
  -e EXL3_FAT_STREAMS="${FAT_STREAMS:-4}"
  # Sparse-indexer prefill workspace: stock = max_model_len x 40 entries (5 GiB at 1M, charged to the KV pool)
  # while the splitter is fed pool-compressed lengths (kpool 4). rightsize = legal per-step max: +18-27 % KV pool
  # here (1.71-1.84M -> 2.18M tokens at 1M), chunking and speed unchanged. Port of MiaAI's patch (vllm#55222).
  -e GLM53_INDEXER_WORKSPACE="${INDEXER_WORKSPACE:-rightsize}"
  # Wrapper-level RoPE pad (alternative to the backend-level pad; keep 0 with the shipped backend).
  -e GLM53_PAD_ROPE="${PAD_ROPE:-0}"
  # Persistent top-k oversubscribes GB10 past ~600K context (see patch_kpool_topk_fallback.py).
  -e GLM53_PERSISTENT_TOPK_MAX_LEN="${PERSISTENT_TOPK_MAX_LEN:-600000}"
  # Breakable CUDA graphs (nightly default on) deadlocked 3 boots/10 inside KDA eager-break capture
  # (py-spy: gather_initial_states / l2norm_fwd launch on both ranks). =0 is not viable on this build (no
  # piecewise graphs without torch.compile): keep 1 and boot through supervise.sh (auto-retry + warmup).
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${BREAKABLE:-1}"
  -e GLM53_COOP_GEOMETRY="${GLM53_COOP_GEOMETRY:-}"   # cooperative-MoE tile geometry (0/1/2); only read by the coop overlay
  # Fused Triton LSE merge of the 2048+128 top-k split (2026-09-18): cold prefill +6-7 %. FUSED_MERGE=0 = the eager
  # merge; hot toggle = touch <JIT cache>/vllm/glm53_fused_merge.off on both nodes (see the SM120 backend).
  -e GLM53_FUSED_LSE_MERGE="${FUSED_MERGE:-1}"
  # Dense EXL3 forward writes each shard into a pre-allocated output instead of torch.cat (2026-09-18): prefill +2 %.
  -e GLM53_DENSE_NOCAT="${DENSE_NOCAT:-1}"
  # MiaAI's SM121 thin-decode fast path (opt-in; needs the image built with patch_exl3_decode_pipeline_ours.py, fails
  # closed otherwise). Measured here: alone = the cooperative MoE gain (not additive); with coop on it only serves the
  # thin tier of prefill (+1-2 %). 0 = stock kernels, byte for byte.
  -e GLM53_EXL3_MOE_FAST="${MOE_FAST:-0}"
  -e TORCH_CUDA_ARCH_LIST=12.1a)
[ -n "${NCCL_IB_GID_INDEX:-}" ] && ENVS+=(-e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX")
# Opt-in NCCL knobs (NCCL_PROTO=Simple measured neutral on prefill here; NCHANNELS not measured).
[ -n "${NCCL_PROTO:-}" ] && ENVS+=(-e NCCL_PROTO="$NCCL_PROTO")
[ -n "${NCCL_NCHANNELS:-}" ] && ENVS+=(-e NCCL_MIN_NCHANNELS="$NCCL_NCHANNELS" -e NCCL_MAX_NCHANNELS="$NCCL_NCHANNELS")

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
  --enable-prompt-tokens-details   # usage.prompt_tokens_details.cached_tokens for clients (prefix-cache hits visible per request)
  # Prefix caching on this hybrid (KDA states + MLA pages + DFlash2 drafter group): see README "Prefix caching".
  # PMU=64: hash unit finer than the drafter block so the KDA tail state at the exact prompt end is cacheable.
  # RETENTION=4608: one KDA state per 4608-token block (upstream default 0 keeps only the prompt-tail state,
  # which the next turn can only reach when n mod 4608 >~ 2304). Empty values disable either flag.
  ${PMU:+--prefix-match-unit $PMU} ${RETENTION:+--prefix-cache-retention-interval $RETENTION}
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --chat-template /opt/chat_template.jinja
  # image cap is PER CONVERSATION (clients resend the whole history): 4 blocked on the 2nd turn.
  # --mm-processor-cache-gb 1 caps the processor cache (vLLM default 4 GiB of UMA on this host).
  --limit-mm-per-prompt "{\"image\":16,\"video\":1}" --mm-processor-cache-gb 1 --skip-mm-profiling)

# --- Adaptive verification length (opt-in, 2026-09-12) -----------------------------------
# ADAPTIVE_K=1 mounts two runtime overlays (default 0 = byte-for-byte the scheduler baked in the
# image, no mount and no extra arg). DFlash2 still drafts k=7; the scheduler verifies only a per-step
# prefix chosen from a CPU-side EMA of accepted drafts, uniform over the batch, so every decode step
# still lands on a FULL CUDA graph captured for each candidate length + 1. Measured here: +19 % prose,
# +31 % prose @131K, +7 % code long, -4 % short FR code; prefill cost zero. Regenerate the two
# overlay files with overlay/adaptive_k/patch_adaptive_k_nightly.py whenever the scheduler moves.
[ -n "$EXL3_OVERLAY_HOST" ] && EXTRA_MOUNTS+=(-v "$EXL3_OVERLAY_HOST:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro")

_ak_sizes() {
  # Uniform decode of r requests at draft length kk is r*(kk+1) tokens; capture the union of those
  # with the list vLLM would build on its own so max_cudagraph_capture_size and mixed coverage stay.
  local seqs=$1 k=$2 kset=$3 maxcg n kk r
  maxcg=$(( seqs * (k + 1) * 2 )); if [ "$maxcg" -gt 512 ]; then maxcg=512; fi
  { echo 1; echo 2; echo 4
    n=8; while [ "$n" -le "$maxcg" ]; do echo "$n"; n=$(( n + 8 )); done
    for kk in ${kset//,/ }; do
      r=1; while [ "$r" -le "$seqs" ]; do echo $(( r * (kk + 1) )); r=$(( r + 1 )); done
    done
  } | sort -n -u | tr '\n' ' '
}
if [ "$ADAPTIVE_K" = 1 ]; then
  for _f in scheduler.py cudagraph_utils.py; do
    [ -f "$HERE/overlay/adaptive_k/$_f" ] || { echo "[run.sh] missing overlay/adaptive_k/$_f (run patch_adaptive_k_nightly.py)" >&2; exit 2; }
  done
  V=/usr/local/lib/python3.12/dist-packages/vllm
  EXTRA_MOUNTS+=(-v "$HERE/overlay/adaptive_k/scheduler.py:$V/v1/core/sched/scheduler.py:ro")
  EXTRA_MOUNTS+=(-v "$HERE/overlay/adaptive_k/cudagraph_utils.py:$V/v1/worker/gpu/cudagraph_utils.py:ro")
  AK_SET="${GLM53_ADAPTIVE_K_SET:-2,4,7}"
  ENVS+=(-e GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-ema}" -e GLM53_ADAPTIVE_K_SET="$AK_SET")
  for _v in ALPHA MARGIN MIN_STEPS SATURATE HIST FILE; do   # passthrough only when the caller set them
    eval "_val=\${GLM53_ADAPTIVE_K_$_v-}"
    [ -n "$_val" ] && ENVS+=(-e "GLM53_ADAPTIVE_K_$_v=$_val")
  done
  read -r -a AK_SIZES <<< "$(_ak_sizes "${SEQS:-6}" "$K" "$AK_SET")"
  ARGS+=(--cudagraph-capture-sizes "${AK_SIZES[@]}")
fi
# NEVER add --language-model-only: it switches to Glm5NextForCausalLM and the module prefixes
# no longer match the EXL3 non_routed keys (language_model.model.layers.*).
# EAGLE_DROP=0 (default): keep the last matching block for the eagle-style drafter instead of dropping it.
# Measured: exact 32K repeat TTFT 5.3 -> 0.4 s, multi-turn hits / decode / acceptance unchanged.
SPEC_EXTRA=$([ "${EAGLE_DROP:-0}" = 0 ] && echo ',"disable_eagle_block_drop":true' || echo '')
[ "$SPEC" = dflash ] && ARGS+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K$SPEC_EXTRA}")
if [ "$ROLE" = head ]; then
  ARGS+=(--node-rank 0); NAME=glm53-up-head
else
  ARGS+=(--node-rank 1 --headless); NAME=glm53-up-worker
fi
# Persistent JIT caches per node (Triton, FlashInfer autotune / deep_gemm, NVRTC, inductor), keyed by image tag.
JIT_MOUNTS=()
if [ "${JIT_CACHE:-1}" = 1 ]; then
  C="${JIT_CACHE_DIR:-$HOME/glm53-upstream-cache}/${IMG##*:}"; mkdir -p "$C"/{triton,vllm,nv,inductor}
  JIT_MOUNTS=(-v "$C/triton:/root/.triton" -v "$C/vllm:/root/.cache/vllm" -v "$C/nv:/root/.nv" -v "$C/inductor:/tmp/torchinductor_root")
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --name "$NAME" --gpus all --network host --ipc host \
  --device /dev/infiniband --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$HF_CACHE":/root/.cache/huggingface ${JIT_MOUNTS[@]+"${JIT_MOUNTS[@]}"} ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"} \
  "${ENVS[@]}" --entrypoint vllm "$IMG" "${ARGS[@]}"
