# GLM-5.3-Flash EXL3 on 2× DGX Spark — upstream-nightly stack vs the MiaAI kit (2026-09-18)

Same hardware for every number below: 2× ASUS Ascent GX10 (GB10, sm_121, 128 GB unified, ~273 GB/s each),
ConnectX-7 RoCE v2, TP=2 across the two nodes. All measurements taken by us unless marked *published*.

## TL;DR

| | Our stack (2026-09-18) | MiaAI kit, best config, on our hardware (2026-09-12, `9348755`) | MiaAI kit `ca85576`, *published* (2026-09-17/18) |
|---|---:|---:|---:|
| Decode, structured (count 1-200) | **91.1** tok/s | 73.5 | 77.3 (coop) / 78.4 (thin-fast) |
| Decode, prose (English, hash map) | **39.5** | 32.6 | 34.9 (thin-fast) / 37.1 (sparkDash) |
| Decode, code (French, BST) | **53.4** | 41.7 | — |
| Decode, code (English, BST) | **60.0** | 48.6 | 73.9 (“code-1”, TheGrill, different prompt) |
| Cold prefill 8K / 32K / 100K | **1 472 / 1 556 / 1 564** tok/s | 1 441 / **1 605** / 1 434 | E3: ~1 580–1 640 at 8K/32K (repeated-text probe) |
| KV pool | **2 140 221 tokens @ 1M** | 883 552 @ 850K | ~1.05 M @ 900K (util 0.85) |
| Executable code eval (8 functions) | 8 / 8 | 8 / 8 | — |
| Multi-turn tool calling | pass | pass | — |

Decode is 21–28 % ahead of the kit measured on our own hardware, and 6–16 % ahead of its best published numbers, on every probe we share;
cold prefill is at parity (−3 % at 32K, +2 % at 8K, +9 % at 100K vs the kit on our machines); the KV pool is 2.4× larger.
Their two latest decode paths (cooperative MoE, SM121 thin-decode fast path) are both real; measured here they are
**equivalent to each other and not additive** (details in §6).

## 1. What is being compared

**Our stack** (public: https://github.com/Alexbob0/glm53-flash-vllm-upstream-sm121; today's changes are in `results/2026-09-18/`, `overlay/`, `exl3-fat-kernel/` and `tests/`)

- Engine: `vllm/vllm-openai:nightly` (vLLM `0.28.1rc1.dev388+g8a728663c`, 2026-09-04), FlashInfer 0.6.18, torch 2.13.0+cu130,
  attention backend `FLASHINFER_MLA_SPARSE_SM120`, KV `fp8_ds_mla`, CUDA graphs (breakable), no torch.compile.
- Weights: routed experts = MiaAI `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (rev `25a44fd`, 4 bpw); **dense layers, shared
  experts and lm_head quantized EXL3 too** (turboderp 4.05 bpw pack: attention K6, shared K6, dense MLP K5, lm_head K6);
  DFlash2 draft quantized EXL3 5 bpw, k = 7 with adaptive verification length (`2,4,7`, EMA policy).
- Serving: `max_model_len` 1 000 000, `gpu_memory_utilization` 0.87, `max_num_seqs` 6, `max_num_batched_tokens` 7168
  (effective chunk 4608 = the hybrid KV block), prefix caching on (prefix-match unit 64, retention 4608), both CX-7 rails
  (`NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`), thinking off by default.
- MoE kernels: MiaAI E3 grouped fat-expert prefill (ported as an additive extension), MiaAI cooperative decode MoE C1
  (geometry 1, ported 2026-09-17), MiaAI SM121 thin-decode fast path (ported 2026-09-18, image b4), plus our own
  overlays (see §5).

**MiaAI kit** (`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`)

- Column “on our hardware”: kit `9348755` rebuilt and run here on 2026-09-12 with its best configuration (their opt-ins,
  `GLM53_ADAPTIVE_K=ema`), same bench scripts, same prompts, same session as our stack. The 2026-09-16 rebuild (`6c22896`)
  gave the same level (structured 66.1 / code-fr 37.1 at stock flags).
- Column “published”: numbers from the kit's own docs at `ca85576` (`docs/sm121-perf-paths.md`, CHANGELOG 1.5.0/1.6.0,
  README sparkDash tables). Their benches (TheGrill `glm-routine-decode-v3`, sparkDash) are not ours; the structured probe
  agrees with our measurement of their kit to within ~1 tok/s, the other rows are indicative only.

## 2. Decode (single stream, TTFT excluded, median of 3, temperature 0, thinking off)

| Probe | Ours 2026-09-18 | Ours 2026-09-05 (first nightly boot) | Kit on our HW 2026-09-12 (best) | Kit stock flags 2026-09-12 | Kit *published* `ca85576` |
|---|---:|---:|---:|---:|---:|
| structured (count 1-200, 200 tok) | **91.1** | 77.5 | 73.5 | 66.1 | 77.29 (coop, +7.3 %) · 78.36 (fast, +7.7 %) |
| prose (hash map, en, 200 tok) | **39.5** | 34.9 | 32.6 | 25.9 | 34.91 (fast) · 37.1 (sparkDash ×1) |
| code (fr, BST, 400 tok) | **53.4** | 45.3 | 41.7 | 38.8 | — |
| code (en, BST, 400 tok) | **60.0** | 58.4 | 48.6 | 46.5 | 73.86 (“code-1”, other prompt) |
| code (sparkDash clamp, 400 tok) | **83.4** | — | 66.6 | 62.4 | — |
| code @ 32K context | 47.4 (09-12) | — | — | 35.9 | — |
| code @ 131K context | 47.9 (09-12) | — | — | 31.5 | — |
| prose @ 32K context | 28.5 (09-12) | — | — | 22.1 | — |
| prose @ 131K context | 29.7 (09-12) | — | — | 21.2 | — |

Long-context rows were measured on 2026-09-12 (before cooperative MoE) and not re-run today.
Boot-to-boot variance on this hardware is ±5 %; the day's decode figures were reproduced across 3 boots (structured 89.9–91.1).

## 3. Cold prefill (prompt tokens / TTFT, best of 2 salted prompts after a warm-up pass, `max_tokens = 1`)

| Context | Ours 2026-09-18 | Ours 2026-09-18 morning | Ours 2026-09-05 | Kit on our HW 2026-09-12 | Kit *published* (E3, repeated text) |
|---|---:|---:|---:|---:|---:|
| 8K | **1 472** | 1 341 | ~1 000 | 1 441 | ~1 580–1 640 |
| 32K | 1 556 | 1 407 | 1 080 | **1 605** | ~1 580–1 640 |
| 100K | **1 564** | 1 418 | 1 165 | 1 434 | — |

Real-code prompts on our side; the kit's published probe is repeated text (our repeated-text figure was 1 335–1 383 before
today's +9 %). The remaining 32K gap is the only prefill cell where the kit leads.

## 4. Capacity, concurrency, quality

| | Ours | Kit on our HW |
|---|---:|---:|
| KV pool (log line `GPU KV cache size`) | **2 140 221 tokens @ 1M** (2.14× a 1M request) | 883 552 @ 850K (1.04×); published ~1.05 M @ 900K |
| c4, repeated 12K prompts, aggregate (2026-09-12) | **41.0** tok/s | 21.0 |
| c4, distinct 12K prompts, 256 tokens (2026-09-18) | 22.6 tok/s aggregate, TTFT 10.9–32.4 s (prefills serialize, `MIXED_PREFILL=skip`) | — |
| 100K repeated prompt (prefix cache) | TTFT 3.4–3.6 s | — |
| code_eval (8 small functions, greedy) | 8 / 8 | 8 / 8 |
| Tool calling (`glm47` parser, multi-turn) | pass | pass |
| Teacher-forced KL vs kit stock (1 792 positions) | top-1 91.0 %, KL median 0.0095 nats | — |
| Noise floor: kit stock vs itself, different boot | top-1 93.5 %, KL median 0.0053 | |
| Noise floor: our stack vs itself, same boot | top-1 94.3 %, KL median 0.0028 | |

Our dense-EXL3 pack therefore costs ~0.004 nats of median KL against BF16 dense layers, for +22–26 % decode; code_eval and
tool calling are unchanged. Note that greedy decoding is **not reproducible run-to-run on this stack** (atomicAdd scatter in the
fat-expert path): “identical greedy output” cannot be used as a regression test here, only a calibrated KL panel can.

## 5. What we changed on top of vLLM nightly, and what each change measured

| Change (date) | Effect measured here |
|---|---|
| SM120 sparse-MLA path with NoPE padding, kpool top-k 2176 = 2048+128 via two calls merged by LSE (09-05) | makes the model run on the nightly; decode parity with the fork (77.5 / 34.9 / 45.3) |
| Dense layers + lm_head + DFlash2 draft in EXL3 (09-04/05) | decode +22–26 % (79–80 / 33 / 48–50), KL +0.004 nats median |
| E2 fat-expert prefill + 4 CUDA streams (09-07) | cold prefill 890 → 1 056 tok/s @ 8K |
| MiaAI E3 grouped fat-expert MoE, additive module (09-07) | cold prefill → 1 260 / 1 343 / 1 345 |
| Indexer prefill workspace right-sizing (09-07) | KV pool 1.7–1.8 M → 2.1–2.27 M @ 1M |
| Prefix-cache fixes for the hybrid layout (mamba block alignment, prefix-match unit 64, eagle block drop off) (09-06/07) | 100K repeat TTFT 105 s → 3.6 s, c4 repeated aggregate 14.9 → 30–34 tok/s |
| Adaptive verification length (port of MiaAI, `2,4,7` EMA) (09-12) | prose +19 % short, +31 % @131K, code long +7 %, prefill unchanged |
| MiaAI cooperative decode MoE C1, geometry 1 (09-17) | decode +9 / +9 / +12 % (structured / prose / code-fr) |
| **Fused Triton LSE merge for the 2048+128 split (09-18)** | cold prefill **+6–7 %** (the eager merge was 8 % of a 4608-token chunk, 1.6× the attention kernel); ≤ 1 ulp bf16 |
| **Pre-allocated output instead of `torch.cat` in the dense EXL3 forward (09-18)** | cold prefill **+2 %** |
| MiaAI SM121 thin-decode fast path, image b4 (09-18) | decode unchanged with coop on; prefill +1–2 % (thin tier); see §6 |
| Tested and rejected | FP8 dense (+10 % decode, −13 % prefill, −12 % pool, KL worse); reconstructed-weight cache (±1 %, −33 % pool); cuBLAS instead of exllamav3 hgemm (±1 %); `no_reconstruct` (−35 %); max_model_len 500K (±2 %); `NCCL_PROTO=Simple` (neutral); MNBT 9216 (neutral, pool −17 %); 8 fat streams (no gain) |

## 6. Findings that may be useful to MiaAI

1. **Cooperative MoE and the thin-decode fast path are the same gain, not two.** On our hardware, `GLM53_EXL3_MOE_FAST=1`
   alone (coop off) gives 89.9 / 40.6 / 51.7 / 61.3 tok/s; coop alone gives 90.9 / 41.4 / 53.8 / 60.2; both together give
   91.1 / 39.5 / 53.4 / 60.0. With coop on, the fast kernel only serves the thin tier of prefill (+1–2 %). Your doc lists the
   composition as untested: it is safe (parity battery `test_exl3_thin_fast_gpu.py` + `compare_thin_fast.py` ALL PASS on our
   build, code_eval 8/8, KL within the inter-boot floor) and neutral on decode. The fast path needed a port to your own
   1.4.2 fork tree (3-parameter `<t_bits, N, cb>` kernel template, `[K][cb-1][N_off]` instance table); happy to share it.
2. **Profile of a 4608-token prefill chunk on this stack** (torch profiler, one iteration, both ranks symmetric, `kernel_union == kernel_sum`
   i.e. nothing overlaps): experts 36 % (E3 fat 21 %, thin `exl3_moe` 7.6 %, staging copies ~4 %), MLA attention wrapper 16 %
   (kernel 5 %, the rest was our eager LSE merge — now fused), dense EXL3 14.5 %, KDA (FLA Triton) 10 %, mHC tilelang 9 %,
   NCCL all-reduce 6.6 % (`AllReduce_Sum_bf16_RING_LL`, 37.7 MB messages, LL protocol; forcing `Simple` changed nothing).
   KDA is not the hidden cost we expected; mHC and the fully serialized NCCL are the next structural items.
3. **The indexer `persistent_topk` is decode-only**: forcing the row-wise fallback (needed to boot at 1M on GB10) costs nothing on prefill;
   max_model_len 500K vs 1M is within ±2 % on prefill from 4K to 100K once the server is warm.
4. **First measurement after boot under-reads by 5–15 %** (JIT/autotune); every number above was taken after a discarded warm-up pass.
5. `MAX_NUM_BATCHED_TOKENS` above the hybrid block (7168 → 9216) did not help prefill here and cost 17 % of KV pool; with E3 it also
   produced a `CUDA_ERROR_ILLEGAL_ADDRESS` in the DeepGEMM indexer on the first prefill in one configuration (cap 32, seqs 4).

## 7. Method

- Decode: your `tests/bench_decode.py` protocol re-implemented (streaming, TTFT excluded, median of 3, temp 0, thinking off);
  probes = count 1-200, English hash-map prose, French/English BST code, sparkDash clamp function.
- Prefill: `prefill_quick.py` — one request per size with a fresh salt, `max_tokens = 1`, prompt tokens / TTFT, best of 2,
  after a discarded warm-up pass at all sizes.
- Quality: 8 executable code functions (greedy); multi-turn tool-calling probe; teacher-forced top-20 logprob panel
  (6 texts, 1 792 positions, `/v1/completions` with `prompt_logprobs`), always read against a same-boot or inter-boot noise floor.
- Everything is TP=2 on the same two machines; images built locally from `vllm/vllm-openai:nightly` (2026-09-04) — the only
  moving part vs your kit is the engine and the overlays.

Raw measurements: `measurements/2026-09-18-{profile,levers,thinfast}/` in our working tree (traces, JSON, logs) — available on request.
