# GLM-5.3-Flash (EXL3) on stock upstream vLLM — 2× DGX Spark, SM121, CUDA graphs on

**TL;DR (2026-09-25)** — GLM-5.3-Flash runs on the *official* `vllm/vllm-openai:nightly` image on two
DGX Sparks (GB10, sm_121) with CUDA graphs, DFlash2 speculative decoding and MiaAI's E3 / cooperative-MoE /
thin-decode kernels. Everything is a **Python overlay** (small patch scripts) plus one extension build —
**no vLLM C++ is rebuilt**. Latest changes: [2026-09-22 → 09-25](#2026-09-22--09-25-agent-latency-the-wi-fi-bug-and-what-we-did-not-adopt).

### Current numbers (2026-09-25 morning, production config, one boot)

| | tok/s |
|---|---:|
| decode, structured (count 1-200) | **91.0** |
| decode, prose (en, hash map) | **40.8** |
| decode, code (fr, BST) / code (en, BST) | **53.2** / **59.3** |
| decode, code (sparkDash "clamp" prompt) | **83.7** |
| cold prefill 8K / 32K / 100K | **1 498 / 1 578 / 1 582** |
| multi-turn agent at 68K context, turns 2-4 (TTFT) | **0.88-0.95 s** |
| KV pool at 1M context | 1.81 M tokens (1.81×) with the fast loader · 2.14 M with `LOAD_CLONE=0` |
| boot (weights 48 s on the head rank) | ~4 min 30 |
| quality | code_eval 8/8 · tool calling pass · HumanEval+fr 97.0 % (2026-09-22, below) |

Protocol: `bench/bench_decode.py` (MiaAI's `bench_decode.py`: streaming, temp 0, thinking off, TTFT excluded, median of 3);
cold prefill = prompt tokens / TTFT, best of 2 salted prompts after a discarded warm-up pass.

### Head-to-head with the MiaAI Lab kit, same machines, same evening (2026-09-24)

Kit `0f49cfd` in its **maximum config** (image built from the repo, 850K, fair mixed-prefill, adaptive-k ema,
`DENSE_FP8=dense,kda`, cooperative MoE geometry 1, `MOE_FAST`, `KDA_BF16_LARGE_M`, `DRAFT_KV_COMPACT`, spin-wait 16)
against this repo, both measured with the same scripts on the same two Sparks. Decode = sparkDash, TTFT excluded,
mean of 2 passes, per stream / aggregate.

| | **this repo** | MiaAI kit `0f49cfd` |
|---|---:|---:|
| structured, 1 stream | **86.4** | 63.2 ¹ |
| code, 1 stream | **81.4** | 33.6 ¹ |
| prose, 1 stream | **40.4** | 30.0 |
| structured, 2 streams (aggregate) | **118** | 71 |
| code, 2 streams (aggregate) | 105 | 108 |
| prose, 2 streams (aggregate) | **55** | 45 |
| structured / code, 4 streams (aggregate) | 130 / 161 | 168 / 192 ² |
| prose, 4 streams (aggregate) | **71** | 45 |
| cold prefill 8K / 32K / 100K | 1 449 / 1 513 / **1 499** ³ | 1 479 / 1 510 / 1 298 |
| multi-turn agent 23K, turns 2-4 (TTFT) | **0.6-0.7 s** ⁴ | 1.2-3.8 s |
| multi-turn agent 68K, turns 2-4 (TTFT) | 0.88-0.95 s ⁴ | 0.8-1.0 s |
| same prompt + 64 tokens (20K) | **0.67-0.86 s** ⁴ | 1.85 s |
| KV pool | 1.81× @ 1M (2.14× slow loader) | 1.85× @ 850K |
| quality (HumanEval 164 + 8 fr, 5 samples at T=0.7, 2026-09-22, kit stock BF16 dense) | 97.0 % | 96.3 % (CI95 of the diff [−0.5; +1.7]) |

¹ The kit's single-stream numbers were very noisy that evening (structured 74 → 52 between the two passes); on the
official protocol on 2026-09-12 it gave 73.5 structured / 32.6 prose / 41.7 code-fr.
² sparkDash showed the kit ahead at 4 streams on structured and code. A dedicated 4-stream decomposition (accepted
tokens per step and ms per iteration from `/metrics`, 400 tokens, T=0, same probe on both stacks) found **no gap**:
structured 155-180 vs 143-188, code 62-74 vs 59-66, prose 58-62 vs 51-54 tok/s aggregate, 3.45-3.58 vs 3.37-3.53
tokens/step. The sparkDash difference is run-to-run variance.
³ Before `MLA_BMM`; this morning 1 498 / 1 578 / 1 582.
⁴ With `SWA_TAIL` (adopted the same night, right after this head-to-head); before it our hits stopped at the last
multiple of 4 608 (68K turns: 3.1-3.3 s).

**Summary**: single-stream decode clearly ahead, parity at 4 streams, prose ahead at every concurrency, cold prefill
at parity up to 32K and +15 % at 100K, multi-turn prefix hits now at or better than the kit, same quality.
The older 2026-09-18 comparison (kit `ca85576`, published numbers, KL panel) stays in
[`results/2026-09-18/comparison-vs-miaai-kit.md`](results/2026-09-18/comparison-vs-miaai-kit.md); this one is in
[`results/2026-09-24/h2h-vs-miaai-kit.md`](results/2026-09-24/h2h-vs-miaai-kit.md).

**Where the difference comes from** (vs the MiaAI kit; each item measured in isolation, see [RESULTS.md](RESULTS.md)):

| Difference vs the MiaAI kit | Effect here |
|---|---|
| Engine: **stock vLLM nightly** (0.28.1rc1) + Python overlays, not their vLLM fork image | same decode as the fork at first boot; everything below stacks on top |
| **Dense layers, shared experts, lm_head and the DFlash2 draft in EXL3** (their kit keeps them BF16) | decode **+22–26 %** (79–80 / 33 / 48–50 → from 77.5 / 34.9 / 45.3), KL +0.004 nats |
| Prefix-cache fixes for the hybrid KV layout (mamba block alignment, prefix-match unit 64, eagle block drop off) | 100K repeat TTFT 105 s → 3.6 s; c4 repeated aggregate 15 → 41 tok/s |
| Sparse-indexer prefill workspace right-sized (pools, not tokens) | KV pool 1.7–1.8 M → 2.1–2.27 M @ 1M |
| Adaptive verification length (port of their patch, on by default here) | prose +19 % short, +31 % @131K, code long +7 %, prefill unchanged |
| Cooperative decode MoE C1 (port of their kernel, geometry 1) | decode +9 / +9 / +12 % |
| **Fused Triton LSE merge** of our 2048+128 top-k split (their fork has a native 2176 kernel; ours had 8 % of the chunk in eager fp32) | cold prefill **+6–7 %** |
| Pre-allocated output in the dense EXL3 forward (no `torch.cat`) | cold prefill +2 % |
| Their SM121 thin-decode fast path, ported to the fork tree (`MOE_FAST=1`) | = the cooperative MoE gain, **not additive**; kept for +1–2 % prefill |
| 1M context at GMU 0.87, `MAX_NUM_SEQS=6` (their defaults: 850K / 0.85 / 4) | 1M instead of 850K at similar headroom (their pool grew to 1.85× in `0f49cfd`) |
| **Control plane pinned to the fabric** (`VLLM_HOST_IP`; stock vLLM binds EngineCore ↔ worker ZMQ on the default route = Wi-Fi here) | c4 p99 inter-chunk 526-576 → 192-195 ms, no more 0.3-0.7 s stalls |
| **Probabilistic DFlash2 drafts** + standard rejection sampling (lossless) | +2 to +14 % tok/s at T=1 (the clients' default) |
| **`SWA_TAIL`**: prefix hits down to the prompt end for the drafter's sliding-window group | multi-turn 68K TTFT 3.1 → 0.9 s |
| **`MLA_BMM`**: MLA absorbed bmm in Triton (cuBLAS picks an sm80 wmma kernel on sm121) | cold prefill +2.1-2.4 % |

Protocol: `bench/bench_decode.py` (MiaAI's `bench_decode.py`: streaming, temp 0, thinking off, TTFT excluded, median of 3);
cold prefill = prompt tokens / TTFT, best of 2 salted prompts after a discarded warm-up pass; quality = 8 executable code
functions, multi-turn tool calling, teacher-forced top-20 logprob panel read against a same-boot or inter-boot noise floor
(greedy output is **not** reproducible run-to-run on this stack, see PITFALLS.md). Hardware: 2× ASUS Ascent GX10
(GB10, 128 GB unified), ConnectX-7 RoCE, one rail (~109 Gb/s).

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
| — | `exl3-fat-kernel/` | the E2 fat-expert GEMM (MiaAI PR #77, +10 % prefill) lives outside upstream ExLlamaV3 | grafted into the extension at build time; extended with `exl3_fat_gemm2` (reads gate and up from their own trellises, no 4 MB staging copy per expert — bit-exact, neutral end to end) and an `atomicAdd` scatter (multi-stream safe) |
| 10 | `overlay/apc/patch_scheduler_mamba_align.py` | **upstream bug**: `Scheduler._mamba_block_aligned_split` aligns prefill chunks on `cache_config.block_size` (1152 = the drafter group) while the KDA groups use the `lcm` block 4608 → KDA states are written at non-aligned positions and **never hashed** (zero prefix-cache hits, even on exact repeats) | align on the real Mamba block (`[apc-align]` boot line) |
| 11 | `overlay/apc/patch_coordinator_swa_partial.py` | `--prefix-match-unit 64` is refused because the drafter's `SlidingWindowManager` "requires block-aligned lookups" — it actually resolves the block view itself | do not let that manager veto fine-grained hits |
| 12 | `overlay/rightsize/patch_indexer_workspace.py` | the sparse-indexer prefill workspace is `max_model_len × 40` entries (5 GiB at 1M, charged to the KV pool) while the splitter is fed pool-compressed lengths (`index_kpool` 4) | MiaAI's right-sizing (vllm#55222) with the nightly anchor (`tokens_per_state`): **+18–27 % KV pool**, chunking and speed unchanged |
| 13 | `overlay/e3/` | E2's per-expert host loop is latency-bound: a fat GEMM costs a flat ~200 µs at ≤ 512 rows (16–32 CTAs on 48 SMs), ~12k launches per chunk | **MiaAI's E3 grouped MoE** (3 launches per layer from device-side tables) built as the additive `exl3_fat_moe_ext` module — **+40 % cold prefill on real code here** |
| 14 | `run.sh` (`EXL3_FAT_STREAMS`) | same latency bound, E2 fallback tier | fat experts round-robined over 4 CUDA streams (one scratch set each, atomic scatter): +14–19 % on E2; superseded by E3 |
| 15 | `overlay/adaptive_k/` | the verifier always checks all k=7 drafts, even when the drafter's acceptance is low (prose) | opt-in **adaptive verification length** (MiaAI port): per-step prefix from a CPU-side EMA of accepted drafts, uniform over the batch so every step still hits a full CUDA graph — **+19 % prose, +31 % prose @131K, +7 % long code**, zero prefill cost |
| 16 | `extensions/cooperative_moe/` | stock fused EXL3 `exl3_moe` re-reads expert weights per token, launch-bound at decode | opt-in **cooperative decode MoE** (MiaAI port, geometry 1 A-wide/B-wide): decode-sized fused path 1-32 rows, shared scratch, prefill/E3 unchanged — **+9 % structured, +9 % prose, +12 % code-fr**; GPU gate passed on both ranks |
| 17 | `overlay/loadclone/weight_utils.py` | **weight loading at 0.6 GB/s on the head rank (274 s for 169 GiB)**: a host→device copy whose source is a file-backed mmap tensor runs at **~0.1 GB/s under a CUDA context on GB10** (4 KiB-page kernel, not only the 64 KiB "wedge") | `clone()` each tensor into anonymous memory in `safetensors_weights_iterator` + a **threaded shard prefetcher** (`GLM53_LOAD_PREFETCH=6`): **274/101 s → 50/52 s** (head/worker), full boot 8.5 → ~4.5 min, bytes identical |

Also required: exllamav3 ≥ 1.4 instantiates `NullConfig` inside `LinearEXL3` — the plugin imports
the real `exllamav3.model.config` under its namespace stub instead of a hand-made stub.

## Weight loading: 274 s → 48 s on the head rank (2026-09-21)

The two ranks load the same 169 GiB from identical NVMe drives, yet `Loading weights took` read
**274 s on the head and 101 s on the worker** for weeks. Everything usually blamed was measured and
cleared: swap (`vm.swappiness=1` → 272 s), fragmentation (16–43 extents per shard), big.LITTLE
placement (95–99 % of samples on the 3.9 GHz X925 cores on both nodes), memory pressure (free/cache
identical), vLLM's multithread loader (273 s — safetensors 0.8 `load_file` is zero-copy mmap too).
py-spy (`docker exec --privileged`, wheel pip-installed into the live container) showed the same
profile on both ranks — 75 % in `_narrow_tp` / `dest.copy_` — just 2.7× slower on the head.

A replay of the loader's exact pattern in a bare container found it (4 cold shards, 5.2 GiB):

| source of the H2D copy | CUDA context | throughput |
|---|---|---|
| file-backed mmap tensor (stock `safe_open().get_tensor()`) | no | disk-bound |
| file-backed mmap tensor | **yes** | **0.09–0.12 GB/s (63 s)** |
| same tensor `.clone()`d into anonymous memory first | yes | 2.1–2.3 GB/s cold, 8.5 GB/s warm |
| `.pin_memory()` first | yes | same as clone |

Tensors that are already contiguous after the TP narrow (row splits, whole tensors) were being copied
straight from the mapping; column splits got a `.contiguous()` (a CPU copy) first and were fine. The
per-rank mix of the two decides how much each rank pays — hence the asymmetry. MiaAI's PR #230 stages
tensors off the mmap only on 64 KiB-page kernels ("cuMemcpyHtoDAsync wedges"); on the 4 KiB DGX OS
kernel it does not wedge, it crawls, and the fix is worth just as much.

`overlay/loadclone/weight_utils.py` = the image's file + two blocks: `param.clone()` in
`safetensors_weights_iterator` (`GLM53_LOAD_CLONE`, default 1) and a prefetcher that keeps *n* shards
ahead of the consumer in the page cache on *n* threads (`GLM53_LOAD_PREFETCH`, default 6). Bytes are
identical by construction; only the copy path changes.

| loader | head | worker |
|---|---:|---:|
| stock nightly | 274 s | 101 s |
| clone | 104 s | 104 s |
| clone + prefetch 3 | 73 s | 72 s |
| **clone + prefetch 6 (default)** | **50 s** | **52 s** |
| clone + prefetch 10 | 48 s | 51 s |
| NVMe `read_ahead_kb` 128 → 2048 | no effect | no effect |

Full boot (container start → `/health` + warmup): 8 min 30 → 4 min 30–5 min 30. Post-boot decode on the
official protocol 87.1 / 43.6 / 54.7 tok/s (structured / prose / code-fr), within boot-to-boot variance
of the reference 90.9 / 41.4 / 53.8; thinking-off and tool-calling unchanged.

## Prefix caching on this hybrid (2026-09-06/07)

The stock nightly served **zero prefix-cache hits** for GLM-5.3-Flash + DFlash2 — an exact 100K repeat
re-prefilled 100K tokens (100 s). Three causes, all verified by instrumentation and fixed here:

1. the chunk-alignment bug (row 10 above) — KDA states never cacheable;
2. the drafter group's veto on `--prefix-match-unit` (row 11) — no tail state at the exact prompt end;
3. `--prefix-cache-retention-interval` defaults to 0 (one KDA state per request, at the prompt-tail block,
   reachable by the next turn only when `n mod 4608 ≳ 2304`) → `RETENTION=4608` keeps one state per block;
4. `disable_eagle_block_drop` (`EAGLE_DROP=0`): the eagle-style drop of the last matching block pushed
   the usable KDA state one block back on prompts ending shortly after a 4608 boundary.

| | before | after |
|---|---|---|
| 8K exact repeat (TTFT) | 8.6 s, 0 hits | 4.0 s, hit 4608 |
| 32K exact repeat | 33 s, 0 hits | **0.4 s** (5.3 s before `EAGLE_DROP=0`) |
| 100K exact repeat | 100 s, 0 hits | **3.5 s**, hit 96768 |
| 8.5K / 32K conversation, turns 2–3 | full re-prefill | 4.4 s / 4.9 s |
| c4 × 12K, second pass | 14.9 tok/s e2e, TTFT p50 43 s | 33 tok/s, p50 11.5 s |

Hit granularity is a multiple of 4608 (KDA states at chunk ends); prompts ≤ 7168 tokens get their first
hit at the third occurrence. Greedy logprobs with and without a hit are identical. Points 1–2 apply to
MiaAI's fork image as well (its scheduler has the same `_mamba_block_aligned_split`; the coordinator
anchor differs by one line), 3–4 need a vLLM with those flags (≥ 0.28 nightly).

## Prefill: from 900 to 1,350 tok/s in one day (2026-09-07, real-code corpus, TTFT-based)

| Config | 8K | 32K | 100K | c4 × 12K TTFT p50 |
|---|---|---|---|---|
| E2 host loop (start of day) | 890 | 940 | 960 | 37 s |
| `exl3_fat_gemm2` (no staging copy) | 900 | 925 | 958 | — |
| MNBT 9216 | 899 | 926 | 953 | 44 s (pool −17 %) |
| E2 + `EXL3_FAT_STREAMS=4` | 1,056 | 1,115 | 1,070 | 32 s |
| E2 + 8 streams | 880 | 969 | 925 | 37 s |
| **E3 grouped MoE (MiaAI) — default** | **1,260** | **1,343** | **1,345** | **26 s** |

Decode (79.6 / 33.4 / 44.2 tok/s structured / prose / code-fr), quality, hits and tool calling unchanged
throughout. MiaAI measures 1,490–1,590 tok/s with E3 on their fork (sparkDash prompts, `MAX_NUM_SEQS=4`,
fused cap 32); on the same repeated-text protocol this stack gives 1,335–1,383 (`bench/prefill_probe.py`).
The diagnosis behind rows 4–6: a fat-expert GEMM launches 16–32 CTAs on 48 SMs and costs ~200 µs
whatever the row count, so the prefill was launch-latency bound, not bandwidth bound.

Two operational notes from the same day:
- **Do not combine `MNBT=9216` with E3 on this stack**: with `SEQS=4`, fused cap 32 and MNBT 9216 the
  worker died twice with `CUDA_ERROR_ILLEGAL_ADDRESS` in DeepGEMM (sparse indexer) on the first ~1.5K-token
  prefill of the warmup sweep; MNBT 9216 on the E2 image and MiaAI's own cap-32/SEQS-4/MNBT-7168 recipe are
  both fine. Root cause not isolated — MNBT 7168 / SEQS 6 / cap 64 (the defaults) served every test.
- **GMU vs host processes**: vLLM refuses to start when free device memory is below `GMU × 121.6 GiB`
  (0.87 → 105.8 GiB). Anything else running on the head (here a Postiz + Temporal stack plus the agent
  session, ~8 GiB) pushes free memory to ~104 GiB: use `GMU=0.85` or free the host. `supervise.sh`
  drops caches and waits for `MEM_FREE_MIN` (115 GiB) before launching, with an explicit message.

## Prefill: the profile that found the missing 9 % (2026-09-18)

Every configuration lever had been tried and measured neutral (context regime, dense reconstruction, weight cache, cuBLAS,
MNBT), so we profiled one 4608-token prefill chunk instead (`PROFILE_DIR` in `run.sh`, `bench/measure.py --profile`,
`bench/trace/` for the analysis). Two facts came out: **nothing overlaps** (`kernel_union == kernel_sum`, NCCL included), and
**the SM120 attention wrapper spent 1.6× its attention kernel in eager PyTorch** — the fp32 LSE merge of our 2048+128 top-k
split, seven ops over `[4608, 32, 512]` tensors, ~3.3 GB of traffic per layer × 11 layers = 8 % of the chunk. KDA was 10 %,
mHC 9 %, NCCL 6.6 % (RING_LL on 37.7 MB messages), the dense EXL3 GEMMs 10 %.

| Fix (same boot, hot toggles) | 8K | 32K | 100K |
|---|---:|---:|---:|
| before | 1 336 | 1 414 | 1 423 |
| + pre-allocated output in the dense EXL3 forward (no `torch.cat`) | 1 367 | 1 440 | 1 450 |
| + **fused Triton LSE merge** (`_lse_merge_kernel`, ≤ 1 bf16 ulp, 19.7 → 2.0 ms per merge) | **1 446** | **1 537** | **1 548** |
| + MiaAI's thin-decode fast path (image b4, thin tier of prefill) | **1 472** | **1 556** | **1 564** |

Decode unchanged (91.1 / 39.5 / 53.4 / 60.0 tok/s structured / prose / code-fr / code-en), code_eval 8/8, tool calling pass,
teacher-forced KL below the same-boot noise floor. Knobs: `FUSED_MERGE=0`, `DENSE_NOCAT=0`, `MOE_FAST=1` (needs the image
built with `exl3-fat-kernel/patch_exl3_decode_pipeline_ours.py`); hot toggles = `glm53_fused_merge.off` /
`glm53_dense_nocat.off` in the JIT cache `vllm/` directory on both nodes. `NCCL_PROTO=Simple` measured neutral.

MiaAI's **thin-decode fast path** (`GLM53_EXL3_MOE_FAST`, kit `ca85576`) is ported here onto their 1.4.2 fork tree; alone it
gives the same decode as the cooperative MoE (89.9 vs 90.9 structured), and with coop on it changes nothing on decode — the
two are one gain, not two. The full comparison against their kit (measured on this hardware and as published) is in
[`results/2026-09-18/comparison-vs-miaai-kit.md`](results/2026-09-18/comparison-vs-miaai-kit.md).

## 2026-09-22 → 09-25: agent latency, the Wi-Fi bug, and what we did not adopt

Four days aimed at the regime this stack actually serves (2-3 coding agents, ~117K of context, clients at
**T=1.0 / top_p 0.95** by default) rather than the T=0 short-prompt benches used so far.

**Adopted**

| change | knob (default) | measured |
|---|---|---|
| **Control plane pinned to the fabric.** Without `VLLM_HOST_IP`, vLLM binds the ZMQ queues EngineCore ↔ remote worker (one `SchedulerOutput` per step) on the default-route IP — **Wi-Fi** on our nodes. At c3-c4 the worker received steps late and rank 0 sat in the all-reduce. | `FABRIC_HOST_IP=1` (`WORKER_IP` or the IPv4 of `NCCL_IF`) | c4: p99 inter-chunk 526-576 → **192-195 ms**, pauses > 250 ms 84-132 → **0**, run 68 → 60-65 s; c1 unchanged |
| **Probabilistic DFlash2 drafts** + standard rejection sampling (canonical speculative sampling: lossless, identical at T=0) | `DRAFT_SAMPLE=probabilistic` (on in `supervise.sh`) | at T=1, 12 reps, 2 boots: **6/6 cells win**, tokens/step +3 to +19 %, tok/s +2 to +14 % (agent edit +14 %) |
| **`SWA_TAIL`**: the drafter's sliding-window KV group (block 1152) could only match whole blocks, so every agent turn re-prefilled up to 4 608 tokens even though MLA and KDA stop at the prompt end. Partial-block caching + fine lookup for that group. | `SWA_TAIL=1` (on in `supervise.sh`) | multi-turn agent at 68K, turns 2-4: TTFT **3.1-3.3 → 0.9 s**; acceptance 0.576 vs 0.517 cold, logprob unchanged, code_eval 8/8 |
| **`MLA_BMM`**: MLA's absorbed `W_UK_T`/`W_UV` bmm in Triton (cuBLAS on sm121 falls back to an sm80 wmma kernel, ~20 TFLOPS) + q written already padded to 576 (no cat, no pad) | `MLA_BMM=1` | prefill **+2.1-2.4 %** (1 498 / 1 578 / 1 582), bit-identical to cuBLAS, KL under the noise floor |
| **E3 prefill cap**: thin/fat split at 16 rows for prefill only (decode keeps 64) | `PREFILL_ROWS=16` | expert layer −15 % (microbench), served prefill +0-2 % |

**Measured and not adopted (kept opt-in, so others don't have to redo it)**

- `FLASHKDA=1` — KDA prefill on the FlashKDA CUDA kernel already shipped in the image: prefill **+6-8 %**,
  HumanEval A/B −0.5 pt (CI95 [−1.5; +0.3], noise), but the KL panel lands above the same-boot noise floor
  (top-1 91.9 % vs 93.75 %) and the KV pool shrinks ~7 %. Not worth it for us.
- `NGRAM=1` — hybrid DFlash2 + n-gram prompt-lookup draft on GPU, lossless. Offline simulation on real agent
  sessions promised +9-15 %; **neutral in serving**: DFlash2 already copies from context.
- `THIN_OVERLAP=1` — E3 thin kernel on a side stream next to the grouped path: neutral (−0.5 %).
- `MAMBA_FREE=1` — port of MiaAI's superseded-KDA-state fix. Tested on the real vLLM block manager: our
  geometry (chunks aligned to 4 608) does not leak on stock; kept as a safety net.
- Adaptive-k driven by concurrency (`GLM53_ADAPTIVE_K_CONC=model`) and by per-position acceptance
  (`GLM53_ADAPTIVE_K_EST=pos`): both neutral. The "bimodal c6 prose" that motivated them was run-to-run noise
  (fixed k=2 gives 65 / 77 / 78 on three runs).
- `--long-prefill-token-threshold 3584`: −20 % p99 freeze for the running stream but +15-18 % intruder TTFT and
  −26 % aggregate at c4×12K. MiaAI's "fair v5" mixed-prefill scheduler: their gain is against their `skip`
  default; stock vLLM already mixes prefill into decode here.
- E3 prefill kernel: ncu + variant builds put the ceiling at ~+5 % (tensor pipe 64 %, register-bound occupancy;
  every structural variant lost). Chantier closed.

**Quality check** (2026-09-22): HumanEval 164 + 8 French tasks, 5 samples at T=0.7 (860 requests per side),
this stack (dense layers + lm_head + draft in EXL3) **97.0 %** vs MiaAI's kit with BF16 dense layers **96.3 %**,
CI95 of the difference [−0.5; +1.7]: no detectable degradation from quantizing the dense layers.

**Profile of a production decode step** (T=1, one request): 77 ms = experts 36 ms (47 %, at the DRAM roofline)
+ dense EXL3 20 ms (~60-65 % of peak) + small BF16 GEMMs ~7 + NCCL 5 + copies 4 + mHC 2 + KDA 2 + MLA 1-2;
nearly the same at 104K of context. Known open issue: the fast loader (`LOAD_CLONE=1`) costs ~0.3 M tokens of
KV pool (~8 GiB of "non-torch" memory); page-cache eviction and `malloc_trim` do not recover it.

## Adaptive-k (opt-in since 2026-09-12, default-on in `supervise.sh`)

DFlash2 drafts 8 tokens (1 anchor + k=7) regardless of how well the draft is accepted. On prose the
acceptance is low, so most of those verifications are wasted compute. The adaptive scheduler keeps the
k=7 draft but sizes the *next* step's draft slots from a CPU-side EMA of accepted tokens per request
(`GLM53_ADAPTIVE_K_SET`, default `2,4,7`), uniform over the batch (minimum) — so the decode step always
lands on a full CUDA graph captured for a candidate length + 1. `run.sh` recomputes the capture-size
union (`_ak_sizes`), purely additive to vLLM's own list.

| vs fixed k=7 (same boot, adaptive toggled live) | |
|---|---|
| prose, short | +19 % |
| prose @131K | +31 % |
| code, long | +7 % |
| short FR code | −4 % |
| prefill | 0 % |

`ADAPTIVE_K=0` is byte-for-byte the baked scheduler (no mount, no extra arg). `supervise.sh` sets it
on; a bare `./run.sh` leaves it off. The two overlay files are generated by
`overlay/adaptive_k/patch_adaptive_k_nightly.py` from `overlay/apc/scheduler.py`; regenerate and run
`test_adaptive_k_nightly.py` whenever the scheduler moves. Runtime retune without a reboot via
`/root/.cache/vllm/glm53_adaptive_k.json` (`{"mode":"off"}` restores k=7).

## Cooperative decode MoE (opt-in, 2026-09-17)

MiaAI's two-stage cooperative kernel for **decode-sized** fused `exl3_moe` (1-32 physical rows, K4 MCG,
H=4096, local I=1024, top-k 8), ported to this stack as `extensions/cooperative_moe/` and selected by a
generated overlay. Prefill, the E3 grouped path and unsupported shapes stay stock. The layer contract of
`overlay/exl3.py` already matched the adapter, so no plugin change was needed.

The distributed deploy is a generated overlay (`EXL3_OVERLAY_HOST`) plus `runtime.py` and
`cooperative_moe.so` staged in `/root/.cache/vllm/cooperative_moe` on both ranks. The `.so` is rebuilt
from the pinned kernel source with this image's nvcc (digest repinned, see the extension's `PORT-NOTES.md`)
and passed the packaged GPU gate (`test_cuda_integration.py`, 48 checks, `rel_l2 ≤ 0.0027`).

| probe (tok/s, official protocol, median of 3) | stock control | + coop geo 1 |
|---|---:|---:|
| structured | 84.0 | **92.9** (+11 %) |
| prose | 40.3 | 41.8 (+4 %, noisy) |
| code-fr | 51.1 | **56.3** (+10 %) |
| code-en | 57.1 | **60.9** (+7 %) |
| code sparkDash | 76.1 | **84.3** (+11 %) |
| prefill 8K / 32K / 100K | — | 1 341 / 1 416 / 1 424 |

Activate with `EXL3_OVERLAY_HOST=<generated overlay> GLM53_COOP_GEOMETRY=1`; `supervise.sh` forwards both
so a supervised restart keeps it. Empty `EXL3_OVERLAY_HOST` is the baked plugin.

## Vision: per-conversation image cap

`--limit-mm-per-prompt` counts the images of the **whole conversation** (agent clients resend the
history), so `image:4` blocked a chat at its second turn. `run.sh` now sends `image:16` and caps the
multi-modal processor cache at 1 GiB (`--mm-processor-cache-gb 1`, vLLM defaults to 4 GiB of unified
memory here); `video:1` stays. No video encoder in the checkpoint — a video is served as its frames.

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
`ADAPTIVE_K=1` (adaptive verification length; on by default under `supervise.sh`),
`EXTRA_ALIAS=<name>` (extra served-model alias), `EXL3_FAT_KERNEL=0` (legacy MoE tier), `FUSED_MERGE=0` /
`DENSE_NOCAT=0` (2026-09-18 prefill fixes off), `MOE_FAST=1` (MiaAI thin-decode kernels, opt-in),
`DRAFT_SAMPLE=probabilistic` / `SWA_TAIL=1` (on under `supervise.sh`), `MLA_BMM=0`, `PREFILL_ROWS=`,
`FABRIC_HOST_IP=0` / `WORKER_IP=<ip>`, opt-ins `FLASHKDA=1`, `NGRAM=1`, `THIN_OVERLAP=1`, `MAMBA_FREE=1`.

Never pass `--language-model-only`: it selects `Glm5NextForCausalLM`, whose module prefixes
(`model.layers.*`) no longer match the pack's `language_model.model.layers.*` keys.

## Layout

```
Dockerfile              official nightly (digest-pinned) + ExLlamaV3 build + overlay
overlay/                the plugin, the SM120 backend, the patch scripts (docstrings = rationale)
overlay/apc/            prefix-cache fixes (scheduler chunk alignment, coordinator SWA veto)
overlay/adaptive_k/     opt-in adaptive verification length (scheduler + cudagraph overlays, generator, test)
overlay/rightsize/      sparse-indexer workspace right-sizing (nightly anchor)
overlay/loadclone/      fast weight loading (clone before H2D + threaded shard prefetch)
overlay/swa_tail/       prefix hits at the prompt end for the drafter's sliding-window group (generator, test)
overlay/mamba_free/     opt-in port of MiaAI's superseded-KDA-state fix (generator, test on the real manager)
overlay/mla_bmm/        Triton bmm for MLA W_UK_T/W_UV at prefill + pre-padded q (generator, test)
overlay/flashkda/       opt-in FlashKDA prefill for GLM's KDA (generator)
overlay/ngram_hybrid/   opt-in hybrid DFlash2 + n-gram draft (generator, interpreter test)
overlay/thin_overlap/   opt-in E3 thin/grouped overlap patch
overlay/e3/             MiaAI's E3 grouped fat-expert MoE (.cu/.cuh + their build script, unmodified, AGPL)
exl3-fat-kernel/        E2 fat-expert GEMM (.cu/.cuh) + graft script (MiaAI Lab) + gemm2 / atomic scatter
                        + patch_exl3_decode_pipeline_ours.py (MiaAI's SM121 thin-decode kernels, ported to this fork tree)
extensions/cooperative_moe/  opt-in cooperative decode MoE (native .so + adapter + generator + tests)
run.sh / warmup.sh / supervise.sh / chain.sh   two-node launcher, warmup sweep, hang-tolerant boot, post-boot bench chain
bench/                  decode protocol, prefill probes, measure.py (auditable streaming bench), apc_turns.py
bench/trace/            torch-profiler trace analysis (per kernel, per family, per outermost op, kernel -> aten op attribution)
tests/                  numerical validation of the top-k split/merge, fused LSE merge (<= 1 ulp), strided hgemm output
results/                dated measurement dumps, incl. the 2026-09-18 comparison against the MiaAI kit
RESULTS.md              measurements; PITFALLS.md: what bit us
```

## License

AGPL-3.0-or-later (this repo, since 2026-09-07: it vendors MiaAI Lab's E3 kernels and their current
`exl3.py`, both AGPL-3.0-or-later since their relicense of the same day). Earlier MiaAI/turboderp code is MIT,
vLLM is Apache-2.0 — see [NOTICE](NOTICE). No weights.
