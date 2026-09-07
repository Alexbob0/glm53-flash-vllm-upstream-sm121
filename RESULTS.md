# Results

> **2026-09-07 update** — prefill +40 % (E3 grouped MoE from MiaAI, ported as an additive module), KV pool
> +18–27 % (indexer right-sizing), prefix caching repaired (0 hits → 97 % on a 100K repeat). Section at the end.

Hardware: 2× ASUS Ascent GX10 (DGX Spark class: GB10, sm_121, 128 GB unified, ~273 GB/s each),
ConnectX-7 RoCE v2, both rails (`NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`), TP=2 with the `mp`
executor across nodes. Image: `vllm/vllm-openai:nightly` at vLLM `0.28.1rc1.dev388+g8a728663c`
(2026-09-04), FlashInfer 0.6.18, torch 2.13.0+cu130. Weights: EXL3 pack (routed experts 4 bpw,
dense/attention K6, MLP K5, lm_head K6), DFlash2 draft (BF16 or EXL3 5 bpw), k = 7.

## Decode (bench/bench_decode.py — TTFT excluded, median of 3)

| configuration | structured | prose | code |
|---|---|---|---|
| no speculative decoding | 23.5 | 23.5 | 23.5 |
| DFlash2 BF16 draft, 128K / 512K / 1M ctx | 77.5 / 76.9 / 76.1 | 34.9 / 33.4 / 32.8 | 45.3 / 44.2 / 43.7 |
| DFlash2 EXL3 draft, 1M ctx | 80.4 | 34.1 | 47.6 |
| + E2 fat-expert kernel, 1M ctx | **80.0** | **33.3** | **49.6** |
| reference: MiaAI fork stack, same night | 79.2 | 36.1 | 43.7 |

The code probe was run with an internal coding prompt; the prose spread (32–38 across runs on both
stacks) is measurement noise. Boot-to-boot variance on this hardware is ±5 %.
DFlash2 acceptance: ~4.8 tokens accepted per step at k = 7 (both drafts).

## Prefill (bench/prefill_probe.py, max_tokens = 1, prompt_tokens / TTFT, 2nd pass)

| configuration | 8K | 32K | 100K |
|---|---|---|---|
| legacy MoE tier | 938 | 964 | ~1 050 |
| E2 fat-expert tier | **1 013** | **1 075–1 084** | **1 165** |
| reference: MiaAI fork | ~1 000–1 100 | | ~1 150 |

First encounters of a prompt-size bucket cost a one-off 7–10 s stall (JIT / autotune); `warmup.sh`
sweeps 12 sizes at boot — afterwards 12/12 fresh prompts (88 → 6 034 tokens) had TTFT within 10 %
of prompt_tokens / 950 tok/s.
Decode speed is flat from 8K to 100K of context; a repeated 100K prompt hits the prefix cache
(TTFT 0.6 s).

## Concurrency (bench/bench_conc_long.py, 12K-token distinct prompts, 256 output tokens)

| | c = 1 | c = 4 |
|---|---|---|
| k = 7 | 31.3 tok/s | 5.6–15.6 per stream, ~58 tok/s aggregate (steady state), TTFT 15.7 → 44.8 s (prefills serialize at ~1 000 tok/s) |
| k = 5 | 31.7 | ~62.6 aggregate; but −17 % structured / −9 % code at c = 1 |

Draft KV block 1024 (see PITFALLS): KV usage 31 % at 4 × 12K, no preemption; estimator 1.84M tokens at 1M.

## Memory / context

| max_model_len | gpu_memory_utilization | KV available | notes |
|---|---|---|---|
| 262K | 0.80 | 16.1 GiB | |
| 512K | 0.87 | 22.0 GiB | |
| 1M | 0.87 | 20.1 GiB | 0.80 is not enough at 1M; needs the top-k fallback (patch 8) |

vLLM's "GPU KV cache size / max concurrency" log line is conservative for this hybrid layout (it
counts per group); the scheduler allocates on real blocks (4608-token MLA blocks, ~34 MB each).

## Quality gate

Teacher-forced top-20 logprobs (`/v1/completions`, `echo`, `max_tokens = 0`) on six code panels,
455 positions, compared to the same weights served by the reference fork build:

| pair | KL median | KL mean | p95 | top-1 agreement |
|---|---|---|---|---|
| this repo vs fork reference (E2 build) | **0.00030** | 0.0062 | 0.027 | 97.6 % |
| this repo vs fork full-EXL3 build | 0.00039 | 0.030 | 0.037 | 97.4 % |
| fork full-EXL3 vs fork E2 (for scale) | 0.00046 | 0.021 | 0.031 | 97.6 % |

Executable code eval (8 small functions, greedy): 6/8, same failures as the fork build.
Multi-turn tool calling (`--tool-call-parser glm47`): parsed on turns 1 and 3, tool result
consumed on turn 2.

## Numerical validation of the top-k split (tests/test_split_merge.py)

Decode kernel, 64 tokens: main 1024 + extra 1024 merged by LSE vs a single 2048 call —
max abs error 4.8e-4 (bf16 level) with log2 weights, 2e-2 with natural-log weights (wrong base).
Prefill orchestrator vs decode kernel on the same rows: LSE identical to 1e-6. Two-call decode at
B=8: 0.168 ms vs 0.080 ms single call, i.e. ~+1 ms per 93 ms step over 11 MLA layers.


## 2026-09-07 — prefix cache, KV pool, prefill (image `b3`: nightly + overlay + E2 gemm2/atomic + E3)

Protocol: `bench/measure.py` (streamed token ids, monotonic clock, unique salted prompts, 0 prefix hits on
the "unique" rows; "repeat" = same prompt again), real-code corpus; decode = `bench/bench_decode.py`
(MiaAI's protocol, TTFT excluded, median of 3). 1M context, GMU 0.87, `MAX_NUM_SEQS=6`, MNBT 7168, DFlash2
EXL3 draft k=7, CUDA graphs. Raw measurements: `results/2026-09-07/*.txt` (chain output per boot).

### Cold prefill (prompt tokens / TTFT)

| Config | 8K | 32K | 100K |
|---|---|---|---|
| E2 (host loop), 07:00 | 8.94 s → 890 tok/s | 34.4 s → 940 | 103.9 s → 960 |
| + `exl3_fat_gemm2` | 8.87 s | 34.9 s | 104.1 s (2nd pass) |
| + `EXL3_FAT_STREAMS=4` (2 boots) | 7.64 / 7.56 s → 1,045–1,056 | 28.9 / 28.7 s → 1,106–1,115 | 92.2 / 93.2 s → 1,070–1,082 |
| **E3 grouped (`EXL3_FAT_GROUPED=1`)** | **6.34 s → 1,260** | **23.9 s → 1,343** | **74.2 s → 1,345** |

Repeated-text probe (`bench/prefill_probe.py`, MiaAI-like): E3 1,335–1,383 tok/s at 8K/32K.

### Prefix cache (after `overlay/apc`, `PMU=64`, `RETENTION=4608`, `EAGLE_DROP=0`)

| Test | stock nightly | fixed |
|---|---|---|
| 8K repeat | 8.6 s, 0 hits | 4.0 s (hit 4608) |
| 32K repeat | 33.0 s, 0 hits | 0.35–0.5 s (hit ≈ full) |
| 100K repeat | 100.4 s, 0 hits | 3.4–3.6 s (hit 96768) |
| 8.5K conversation, turns 2–3 | — | TTFT 3.4–4.4 s (hit 4608) |
| 32K conversation, turns 2–3 | — | TTFT 4.4–4.9 s (hit 27648) |
| c4 × 12K + 256 tokens, 2nd pass | 14.9 tok/s e2e, TTFT p50 ~43 s | 30–34 tok/s, p50 8.6–12.8 s |

### KV pool at 1M (log line `GPU KV cache size`)

| | tokens |
|---|---|
| stock workspace (8 boots, 2026-09-06) | 1,708,487 – 1,837,638 |
| `GLM53_INDEXER_WORKSPACE=rightsize` | 2,103,321 – 2,269,372 (E3 scratch included) |

### Decode (unchanged) and quality

Structured 79.5–79.9, prose 33–37, code-fr 42.6–48.8 tok/s across the day's 9 boots (the E2 vs E3 paths
never run on decode-sized steps). Greedy code output, multi-turn hits + DFlash acceptance (0.2–0.4 on
French prose, unchanged), and tool calling checked after each change. `exl3_fat_gemm2` vs the staged
E2 GEMM: bit-identical on random trellises (M=5760). E3 parity vs the LinearEXL3 reference: MiaAI's
`tests/bench_e3_microbench.py` — not yet re-run on this image (needs an idle GPU).

### Things that did not help

- `MAX_NUM_BATCHED_TOKENS=9216` (effective chunk 4608 → 9216): prefill unchanged, KV pool −17 %, c4 TTFT worse.
- 8 fat-expert streams: back to the 1-stream level (oversubscribes the 48 SMs).
- Removing the 4 MB gate|up staging copy per fat expert (`exl3_fat_gemm2`): bit-exact, but the copies were
  overlapped — no wall-time change.
