# Head-to-head vs the MiaAI kit `0f49cfd` — same machines, same evening (2026-09-24)

- **this repo**: nightly image `glm53-upstream:b4`, production config (`serve-coop.sh`), measured 19:03 local time.
- **MiaAI kit** `0f49cfd`, maximum config: image `glm53-kit:instanttensor-0f49cfd` built from the repo, 850K context,
  `GLM53_MIXED_PREFILL_CHUNK=fair`, adaptive-k `ema`, `GLM53_DENSE_FP8=dense,kda`, cooperative MoE geometry 1,
  `GLM53_EXL3_MOE_FAST=1`, `GLM53_KDA_BF16_LARGE_M=1`, `DRAFT_KV_COMPACT`, `GLM53_SPINWAIT_MS=16`; measured 19:29.
  KV: `GPU KV cache size: 1,572,073 tokens, Maximum concurrency for 850,000 tokens per request: 1.85x`.

Same scripts for both: sparkDash (concurrency 1/2/4, 400 tokens, 2 passes), `measure.py` cold prefill (best of 2
salted prompts) + exact repeat, `turns_probe.py` multi-turn (client TTFT, cached tokens from `usage`).

## Decode (sparkDash, TTFT excluded, mean of 2 passes) — per stream / aggregate

| probe | c | this repo | kit |
|---|---:|---:|---:|
| structured | 1 | 86.4 / 86.4 | 63.2 / 63.2 |
| structured | 2 | 59.2 / 118.4 | 36.0 / 71.0 |
| structured | 4 | 36.5 / 130.3 | 42.7 / 167.5 |
| code | 1 | 81.4 / 81.4 | 33.6 / 33.6 |
| code | 2 | 53.4 / 104.9 | 54.4 / 108.4 |
| code | 4 | 40.6 / 160.5 | 48.5 / 191.9 |
| prose | 1 | 40.4 / 40.4 | 30.0 / 30.0 |
| prose | 2 | 27.8 / 54.6 | 23.0 / 44.9 |
| prose | 4 | 18.0 / 71.0 | 12.1 / 45.3 |

The kit's single-stream results were very noisy (structured 74 → 52 between passes).

### 4-stream decomposition (follow-up, same night)
`c4_decomp.py`: 4 streams, 400 tokens, T=0, accepted tokens per step and ms per iteration from `/metrics` deltas.

| | this repo | kit |
|---|---:|---:|
| structured, aggregate tok/s | 155-180 | 143-188 |
| code, aggregate tok/s | 62-74 | 59-66 |
| prose, aggregate tok/s | 58-62 | 51-54 |
| code, tokens per step | 3.45-3.58 | 3.37-3.53 |
| ms per iteration | 168-226 | 183-226 |

No gap at 4 streams: the sparkDash c4 difference above is run-to-run variance.

## Prefill (tok/s, best of 2) and exact repeat (TTFT)

| tokens | this repo | kit |
|---:|---:|---:|
| 8 000 | 1 449 · repeat 2.57 s | 1 479 · repeat 0.77 s |
| 32 000 | 1 513 · repeat 3.64 s | 1 510 · repeat 2.15 s |
| 100 000 | 1 499 · repeat 2.81 s | 1 298 · repeat 2.25 s |

(this repo on 2026-09-25 with `MLA_BMM=1`: 1 498 / 1 578 / 1 582)

## Multi-turn (client TTFT, cached tokens / prompt tokens)

| probe | this repo, 2026-09-24 | this repo + `SWA_TAIL` (same night) | kit |
|---|---|---|---|
| same prompt + 64 tokens (20.4K) | 1.73-1.86 s (18 432) | 0.67-0.86 s (20 352) | 1.85 s (17 920) |
| agent 23K, turns 2 / 3 / 4 | 3.3 / 0.7-0.9 / 0.8 s | 0.71 / 0.67 / 0.6-0.7 s | 1.2-2.1 / 1.3-3.7 / 1.4-3.8 s |
| agent 68K, turns 2-4 | 3.1-3.3 s (64 512) | 0.88-0.95 s (prompt end) | 0.8-1.0 s (68 096) |

The gap before `SWA_TAIL` came from the drafter's sliding-window KV group (block 1152), which could only match whole
blocks, so our hit fell back to the last multiple of 4 608. `SWA_TAIL` caches the drafter's partial tail block at the
last 64-token boundary of the prompt (same rule as MLA). Quality of served-from-cache turns: drafter acceptance
0.576 vs 0.517 recomputed cold, mean logprob −0.140 vs −0.155, code_eval 8/8.
