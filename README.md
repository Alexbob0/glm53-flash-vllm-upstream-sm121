# GLM-5.3-Flash (EXL3) on stock upstream vLLM — 2× DGX Spark, SM121, CUDA graphs on

**TL;DR** — GLM-5.3-Flash runs on the *official* `vllm/vllm-openai:nightly` image on two DGX
Sparks (GB10, sm_121) at **80 tok/s structured / 33 prose / 50 code, prefill 1 000–1 165 tok/s**,
with CUDA graphs, DFlash2 speculative decoding and a 1M context. That is on par with the
best fork-based stack for this hardware (MiaAI Lab's) and ~3× the ~26 tok/s reported by the
`--enforce-eager` SM90-route workarounds ([vllm#53963](https://github.com/vllm-project/vllm/issues/53963)).
Everything is a **Python overlay** (nine small patch scripts) plus one extension build — **no vLLM
C++ is rebuilt**.

| | structured | prose | code | prefill 8K / 32K / 100K | KV pool @1M |
|---|---|---|---|---|---|
| this repo (nightly + EXL3 draft + E2) | **80.0** | 33.3 | **49.6** | **1 013 / 1 080 / 1 165** | 20 GiB (GMU 0.87) |
| MiaAI fork stack, same night, same protocol | 79.2 | 36.1 | 43.7 | ~1 000–1 100 / ~1 150 | 1.54M tokens |
| community SM121 recipes (SM90 route, eager) | ~26 (MTP k=3) | | | ~1 400 | |

Protocol: `bench/bench_decode.py` (MiaAI's `bench_decode.py`: streaming, temp 0, thinking off,
TTFT excluded, median of 3). Quality: teacher-forced top-20 logprobs on code panels, KL median
**0.00030 nats** vs the reference fork build (top-1 agreement 97.6 %). Details in
[RESULTS.md](RESULTS.md). Hardware: 2× ASUS Ascent GX10 (GB10, 128 GB unified), ConnectX-7 RoCE.

## Why this exists

GLM-5.3-Flash is a hybrid (34 KDA linear-attention + 11 sparse-MLA layers) with **rope-free MLA**
(`qk_rope_head_dim = 0`). On SM120/121 the only sparse-MLA backend in stock vLLM requires the
packed `fp8_ds_mla` layout whose kernels assume DeepSeek's 64-dim RoPE, and FlashInfer 0.6.18 only
ships GLM decode kernels for `top_k ∈ {128, 512, 1024, 2048}` while the model's kpool indexer
produces a 2048 + 128 table. Every public workaround so far routes attention through the SM90
backend and gives up CUDA graphs. This repo keeps the SM120 packed path and CUDA graphs.

## What is patched (and why)

All patches live in `overlay/` and are applied at image build time. Each file has a docstring with
the failure it fixes.

| # | file | problem on the stock nightly | fix |
|---|---|---|---|
| 1 | `patch_quant_registry.py`, `patch_model_overrides.py` | no `exl3` quantization method | register `Exl3Config` (lazy) + ModelConfig override list |
| 2 | `patch_glm5next.py` | KDA/MLA layers are built with `quant_config=None` (BF16 assumption) | let them see the real config so dense EXL3 layers match |
| 3 | `flashinfer_mla_sparse_sm120.py` | `pe_dim must be 64 for fp8_ds_mla` (NoPE) | MiaAI's backend: 64 zero RoPE dims padded *inside the impl* (q_pe = k_pe = 0, scores unchanged, scaling untouched) |
| 4 | same file, `forward_mqa` | no GLM kernel for `top_k = 2176`; prefill orchestrator refuses ≤ 64 tokens | **split main 2048 + extra 128** (64-token slices on the decode kernel), exact **LSE merge in log2** (validated to 5e-4 = bf16, +0.09 ms per MLA layer) |
| 5 | `patch_glm5next_eagle3.py` | DFlash speculator needs `SupportsEagle3`; glm5next has no aux-hidden-state plumbing | port of the fork's block (4-stream mHC contraction) |
| 6 | `patch_dflash_kv_auto.py` | draft's dense attention inherits `fp8_ds_mla` → "No valid attention backend" | draft attention gets a bf16 copy of the cache config |
| 7 | `patch_kv_drafter_group.py` | KV page unification fails (drafter page vs MLA page, prime factor 41); the generic path pads KDA states to the drafter page | **drafter KV group**: draft layers slot-share the MLA tensors (block 64, page padded to the MLA page), plus correct accounting |
| 8 | `patch_kpool_topk_fallback.py` | `persistent_topk` oversubscribes the 48 SMs at 1M context | `top_k_per_row_decode` when `max_model_len > 600K` (−1 % decode) |
| 9 | `patch_dflash_exl3_kv.py` | an EXL3-quantized DFlash2 draft has no `.weight` for the fused context-KV GEMM | reconstruct K/V rows once via identity forwards on the EXL3 shards |
| — | `exl3-fat-kernel/` | the E2 fat-expert GEMM (MiaAI PR #77, +10 % prefill) lives outside upstream ExLlamaV3 | grafted into the extension at build time |

Also required: exllamav3 ≥ 1.4 instantiates `NullConfig` inside `LinearEXL3` — the plugin imports
the real `exllamav3.model.config` under its namespace stub instead of a hand-made stub.

## What upstream would need to make this unnecessary

1. A GLM_NSA decode kernel (and prefill orchestrator path) for `top_k = 2048 + kpool tail` in
   FlashInfer, or a `sparse_mla_segments` path that the DSV3.2/GLM branch honours.
2. NoPE support in `concat_and_cache_mla` / the SM120 backend (in-impl padding is a fine interim).
3. `SupportsEagle3` on `Glm5NextForCausalLM` / `ForConditionalGeneration`.
4. A drafter KV group in the GLM-5.3-Flash hybrid grouping (`_get_kv_cache_groups_glm5_next`).
5. A bounded fallback for `persistent_topk` when `total_ctas > num_sms × occupancy`.

Related upstream work: vllm#53963 (issue), #55219 (generic packed KV layout for GLM-5.3-Flash),
#55222 (indexer workspace right-sizing), ZJY0516/vllm#15 (rope-free MLA on SM12x, C++).

## Build & run

```bash
docker build -t glm53-upstream:latest .        # ~6 min of CUDA compile per node (MAX_JOBS=3)
# on the worker node, then on the head:
export HF_CACHE=$HOME/hf MODEL_SNAP=models--<your-exl3-pack>/snapshots/<rev> \
       DRAFT_SNAP=models--incoai--GLM-5.3-Flash-DFlash2/snapshots/<rev> \
       HEAD_IP=10.0.0.1 NCCL_IF=enp1s0f0np0 NCCL_HCA=rocep1s0f0,roceP2p1s0f0
./run.sh worker      # node 2
./run.sh head        # node 1
./supervise.sh       # node 1: boots both ranks, warms up, restarts both on the capture-race hang (PITFALLS.md)
BENCH_URL=http://127.0.0.1:8888/v1/chat/completions ./bench/bench_decode.py
```

Weights: the target is an EXL3 pack with `quantization_config.non_routed_exl3` keys for the dense
layers (routed experts from Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw, dense/attention/lm_head from
turboderp/GLM-5.3-Flash-exl3, see the companion repo
[glm53-flash-dense-exl3-tp2](https://github.com/Alexbob0/glm53-flash-dense-exl3-tp2)); the draft is
incoai/GLM-5.3-Flash-DFlash2 (BF16, works as-is) or an EXL3 quant of it (faster; cannot be
redistributed, CC BY-NC-ND). Knobs: `SPEC=none` (no draft), `MAX_LEN`, `GMU`, `K`,
`EXTRA_ALIAS=<name>` (extra served-model alias), `EXL3_FAT_KERNEL=0` (legacy MoE tier).

Never pass `--language-model-only`: it selects `Glm5NextForCausalLM`, whose module prefixes
(`model.layers.*`) no longer match the pack's `language_model.model.layers.*` keys.

## Layout

```
Dockerfile              official nightly (digest-pinned) + ExLlamaV3 build + overlay
overlay/                the plugin, the SM120 backend, nine patch scripts (docstrings = rationale)
exl3-fat-kernel/        E2 fat-expert GEMM (.cu/.cuh) + graft script       (MiaAI Lab, MIT)
run.sh / warmup.sh / supervise.sh   two-node launcher, post-boot warmup, hang-tolerant boot supervisor
bench/                  decode protocol, prefill probe (8K/32K), ~100K probe
tests/                  numerical validation of the top-k split/merge, kernel shape probes
RESULTS.md              measurements; PITFALLS.md: what bit us
```

## License

MIT (this repo). Bundled MiaAI Lab and turboderp code is MIT — see [NOTICE](NOTICE). No weights.
