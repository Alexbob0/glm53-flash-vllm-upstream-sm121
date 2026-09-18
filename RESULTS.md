# Results

> **2026-09-18 update** — a torch profile of one 4608-token prefill chunk found the hidden cost: ~260 ms of eager glue in our
> SM120 attention wrapper (the fp32 LSE merge of the 2048+128 top-k split), 8 % of the chunk. Replaced by one Triton kernel
> (+6–7 % cold prefill) plus a pre-allocated output in the dense EXL3 forward (+2 %): **1 446 / 1 537 / 1 548 → with MiaAI's
> thin-decode fast path also ported (image b4) 1 472 / 1 556 / 1 564 tok/s at 8K / 32K / 100K**, decode unchanged
> (91.1 / 39.5 / 53.4 / 60.0), KL within the same-boot noise floor. MiaAI's thin-decode fast path and the cooperative MoE
> turn out to be the same gain, not two. Section at the end; full comparison in `results/2026-09-18/comparison-vs-miaai-kit.md`.
>
> **2026-09-12 update** — adaptive verification length (opt-in, default-on in `supervise.sh`): +19 % prose,
> +31 % prose @131K, +7 % long code, zero prefill cost. Head-to-head vs the MiaAI fork kit: 82.3 / 38.2 / 48.9
> vs 73.5 / 32.6 / 41.7 tok/s, KV pool 2.14M vs 0.88M tokens @1M, c4 aggregate 41.0 vs 21.0. Section at the end.
>
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

## 2026-09-12 — adaptive-k, and a head-to-head against the MiaAI fork kit

Two serving configurations were measured on one common bench (streaming, temp 0, thinking off, TTFT
excluded, 2 passes): `C-nightly-prod` = this stack with E3 and the fixed k=7 verifier; `D-nightly-adaptive`
= the same plus `ADAPTIVE_K=1`. The fork kit was measured on the same bench, its best config.

| probe (tok/s) | this repo, C | this repo, **D (+adaptive-k)** | MiaAI fork, best |
|---|---:|---:|---:|
| structured (count) | 80.1 | **82.3** | 73.5 |
| prose (hashmap, en) | 34.5 | **38.2** | 32.6 |
| code (fr, BST) | 46.0 | **48.9** | 41.7 |
| code (en, BST) | 58.4 | 55.0 | 48.6 |
| code @32K | 44.1 | **47.4** | — |
| code @131K | 42.8 | **47.9** | — |
| prose @32K | 24.2 | **28.5** | — |
| prose @131K | 22.1 | **29.7** | — |

Prefill (cold, TTFT-based): 1 291 / 1 371 / 1 348 tok/s at 8K / 32K / 100K — adaptive-k is free there
(the fork kit leads cold prefill at 1 441 / 1 605 / 1 434, an open gap). KV pool: **2 140 221 tokens**
(fork: 883 552). c4 aggregate: **41.0** vs 21.0 tok/s. `code_eval` 8/8 both, tool calling OK.

Isolating adaptive-k on a single boot (fixed k=7 → adaptive, live): +19 % short prose, +31 % prose @131K,
+7 % long code, −4 % short FR code, prefill unchanged. A/B with an adaptive set of `2,4,7` vs `2,5` and
`ema` vs `off` confirmed the EMA policy; see `overlay/adaptive_k/test_adaptive_k_nightly.py` for the
policy unit tests (the launcher's capture-size union is tested too, so it cannot drift from `run.sh`).

> The template bug fixed the same day (Reasoning Effort emitted even with thinking off) is *not* a perf
> knob: with it, `code_eval` fell to 6/8 and code came out 3–10× too long. The shipped
> `chat_template.jinja` gates the Reasoning Effort line on `thinking_enabled`.


## 2026-09-18 — prefill profile, two overlay fixes (+8–9 %), MiaAI's thin-decode fast path (= coop, not additive)

Stack: image b3 → b4 (+ thin-decode kernels), cooperative MoE geometry 1, adaptive-k, 1M, GMU 0.87, `MAX_NUM_SEQS=6`, MNBT 7168.

### Profile of one 4608-token prefill chunk (torch profiler, one iteration, `PROFILE_DIR` recipe, both ranks within 1 %)

`kernel_union == kernel_sum` (3 253 vs 3 254 ms): nothing overlaps, NCCL included. By outermost op (head rank):

| Op | ms | % | of which |
|---|---:|---:|---|
| `moe_forward_shared` (E3 + thin + shared) | 1 177 | 36 | fat gate-up 385, fat down 297, thin `exl3_moe` 246, staging copies ~125 |
| `unified_mla_attention_with_output` (11 layers) | 531 | 16 | prefill kernel 164, **eager LSE merge + pad glue ~260**, tail 2048+128 47 |
| `dense_exl3_forward` | 471 | 14.5 | fp16 cuBLAS GEMM 318, reconstruct 35, `torch.cat` 41, `x.to(fp16)` 41 |
| KDA (FLA Triton, 34 layers) | 337 | 10 | |
| mHC tilelang | 294 | 9 | |
| NCCL all-reduce | 216 | 6.6 | 102 × `AllReduce_Sum_bf16_RING_LL`, 37.7 MB each |
| casts / contiguous outside ops | 108 | 3 | |

The merge was seven fp32 ops over `[4608, 32, 512]` tensors: ~3.3 GB of traffic per layer, 1.6× the attention kernel.
Raw: `results/2026-09-18/profile-4608-chunk-*.txt`; tools: `bench/trace/`.

### Same-boot A/B of the fixes (hot toggles, `prefill_quick`, best of 2 after a warm-up pass)

| Config | 8K | 32K | 100K |
|---|---:|---:|---:|
| both off (= `NCCL_PROTO=Simple` only) | 1 336 | 1 414 | 1 423 |
| pre-allocated dense output only | 1 367 | 1 440 | 1 450 |
| **fused Triton LSE merge + pre-allocated output** | **1 446** | **1 537** | **1 548** |
| repeat | 1 453 | 1 531 | 1 538 |

`NCCL_PROTO=Simple`: neutral. Decode 90.9 / 41.4 / 53.8 / 60.2 / 82.9 (unchanged). code_eval 8/8, tool calling pass.
Teacher-forced KL, both fixes vs both off, same boot: top-1 95.2 %, median 0.0019 nats — **below** the same-config repeat on that
boot (94.3 %, 0.0028). Kernel unit test: ≤ 1 bf16 ulp (82 rounding flips / 23.4 M values), 19.7 → 2.0 ms per merge at prefill shape.

### MiaAI's SM121 thin-decode fast path (`GLM53_EXL3_MOE_FAST`, image b4)

Ported onto this fork tree (`exl3-fat-kernel/patch_exl3_decode_pipeline_ours.py`); their parity battery
(`test_exl3_thin_fast_gpu.py --smoke` + `compare_thin_fast.py`) passes on our build. Serving A/B, same day:

| Config | structured | prose | code fr | code en | 8K / 32K / 100K prefill | KL vs b3 coop |
|---|---:|---:|---:|---:|---|---|
| b3, cooperative MoE (reference) | 90.9 | 41.4 | 53.8 | 60.2 | 1 446 / 1 537 / 1 548 | — |
| **b4, coop + FAST=1 (adopted)** | 91.1 | 39.5 | 53.4 | 60.0 | **1 472 / 1 556 / 1 564** | 95.9 % / 0.0027 |
| b4, FAST=1, coop off | 89.9 | 40.6 | 51.7 | 61.3 | 1 468 / 1 557 / 1 557 | 96.1 % / 0.0021 |

The fast path alone equals the cooperative MoE alone (−1 %); together, decode is unchanged (coop already serves decode rows
1–32) and the fast kernel only speeds up the thin tier of prefill. Kept on for that +1–2 %.

### Negative results of the day (all measured, none adopted)

max_model_len 500K vs 1M: ±2 % (the persistent top-k kernel is decode-only) · `no_reconstruct` on dense layers: −35 % ·
cache of reconstructed fp16 dense weights: ±1 % for −33 % KV pool · cuBLAS instead of exllamav3 `hgemm`: ±1 % ·
`NCCL_PROTO=Simple`: neutral.
