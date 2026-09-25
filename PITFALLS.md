# Pitfalls

Things that cost us a boot (8–10 minutes each on this hardware), in the order we hit them.

- **`--language-model-only` silently breaks EXL3 matching.** It instantiates
  `Glm5NextForCausalLM` (prefixes `model.layers.N`) instead of `ForConditionalGeneration`
  (`language_model.model.layers.N`); the dense layers fall back to BF16 and the loader dies on
  the first `.mul1` tensor with `KeyError: 'layers.0.self_attn.in_proj_qkvbfg_a.mul1'`.
- **exllamav3 ≥ 1.4 needs `NullConfig`.** `LinearEXL3(config=None)` imports it from
  `exllamav3.model.config`; a hand-written stub module lacks it. Import the real module (it does
  not need flash_attn) under the namespace stub.
- **`pe_dim must be 64 for fp8_ds_mla`** is the NoPE wall of vllm#53963. Do not "fix" it with
  `--hf-overrides '{"qk_rope_head_dim": 64}'`: the indexer would apply RoPE on 64 of its 128 dims
  and the softmax scale would become 320^-0.5 instead of 256^-0.5. Pad at the attention boundary.
- **`SM120 sparse MLA v32/GLM expects sparse block_tables shape (B, 1, 2048), got (B, 1, 2176)`**
  then **`no decode kernel for this shape ... topk=2176`**: the kpool indexer widens the table by
  the in-progress tail. FlashInfer 0.6.17 fell back to the paged orchestrator for any decode
  shape; 0.6.18 asserts `num_tokens > 64` in C++ (`Decode must go through
  sparse_mla_sm120_decode_dsv3_2`). Hence the 2048 + 128 split.
- **The LSE returned by the SM120 kernels is base 2.** Merging with natural-log weights is off by
  2e-2 (see tests).
- **`Model does not support EAGLE3 interface`**: the DFlash speculator needs `SupportsEagle3` on
  the target and `EagleModelMixin` on the inner model (the holder assertion in
  `set_aux_hidden_state_layers`).
- **`No valid attention backend found ... kv_cache_dtype=fp8_ds_mla, use_mla=False`**: the MLA
  canonicalization writes `fp8_ds_mla` into the global `cache_config`, and the draft's dense
  attention inherits it. `--kv-cache-dtype-skip-layers` by index does not help: DFlash2 draft
  layers are indexed 0–4 in the base class and would collide with target layer 3 (MLA).
- **`page size is not divisible by the maximum page size and cannot be padded`** on the indexer
  cache: the hybrid model forces a 4608-token attention block (page ≥ KDA state page); the draft
  on the same block becomes the largest page and MLA pages (656 B × 4608 = 2^13 × 41) cannot
  divide it. The generic unification would also pad every KDA state to the draft page (tens of GB).
  The GLM grouping needs an explicit drafter group.
- **`persistent_topk would oversubscribe ... total_ctas=85 > num_sms*occupancy`** appears only at
  `max_model_len` ≈ 1M (85 CTAs vs 48 SMs; the FilteredTopK fallback wants 128 KB smem, GB10 has
  99 KB). 512K is fine; beyond ~600K use the row-wise kernel.
- **1M needs `--gpu-memory-utilization 0.87`** on 2× GB10 (0.80 leaves 6.8 GiB of KV, 9.4 needed
  for one request); the indexer workspaces grow with max length.
- **Boot / first-request hang = breakable-CUDA-graph capture race (root-caused).** 3 boots out of
  10 froze, always at the first CUDA-graph capture of a new shape (boot `capture_model`, or the lazy
  capture triggered by the first real request). `py-spy dump` on both ranks
  (`results/hang-capture-pyspy-*.txt`): rank 0 stuck in `gather_initial_states`, rank 1 in
  `l2norm_fwd`, both inside the KDA layer's `@eager_break_during_capture` region under
  `capture_model`, both blocked in the Triton launch call (`triton/backends/nvidia/driver.py`),
  GPUs idle, workers spinning at 170 % CPU. The nightly auto-enables
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for this model and there is no alternative: with it off the
  engine refuses to start ("piecewise CUDA graphs unavailable, model is not torch-compiled").
  Mitigation: `supervise.sh` boots both ranks, waits for `/health` (bounded), runs `warmup.sh` and
  restarts BOTH ranks on a hang (up to 3 tries). Independent of the E2 kernel and of the draft.
  The MiaAI fork stack (same mechanism) hung the same way once in our hands.
- **Sporadic 7–10 s TTFT stalls on ~30 % of requests** after boot, at any prompt size (a 140-token
  prompt at 9.8 s, a 1.2K one at 8.4 s), then never again for that shape bucket: first-encounter
  JIT/autotune/specialization cost. `warmup.sh` now sweeps 12 prompt sizes (60 → 15K tokens);
  after it, 0 stalls in 12 fresh prompts and TTFT sits at ~900–1 000 tok/s from 300 tokens up.
  Without the sweep an agent harness sees TTFT 20–26 s on 14K prompts instead of ~14 s.
- **Draft KV group block size vs concurrent prefills.** With 64-token draft blocks (the fork's
  padded slot-share default) an 18K prompt transiently needs 281 shared block ids during its
  prefill (the SWA window is trimmed only afterwards); three concurrent prefills exceed the
  ~590-id pool, the scheduler preempts running decodes and the engine thrashes (KV usage
  oscillating 65 → 99 %, generation < 10 tok/s). `GLM53_DRAFT_BLOCK=1152` (default now) keeps the
  page under the MLA page and cuts the id pressure 18×: KV usage ~30 % with 4 × 12K requests.
- **Draft block must divide the MLA block (4608) or prefix-cache hits disappear.** Hybrid-model
  hits are aligned on the lcm of all group block sizes: with a 1024-token draft block the first
  possible hit is at 9 216 tokens (an identical 9.1K prompt: 0 hit; 36K: 27.6K reused). 1152
  divides 4608, so hits happen every 4 608 tokens — still coarse (the KDA state is checkpointed per
  block): prompts shorter than 4 608 tokens never hit, agent turns get ~11 % block hits.
- **Concurrency scaling is structural**: c1 → c4 at 12K context is ×1.9 on this model (34 KDA
  layers verified per draft block, 288 experts top-8 → nearly all experts touched per step at
  4 × 8 draft rows); k=5 buys +8 % at c4 but costs −17 % structured / −9 % code at c1. Keep k=7.
- **`docker rm -f` deletes the container log.** `docker logs > file` first.
- **Two engines on one node.** A forgotten worker container from another stack holds ~98 GB and
  makes the next NCCL init fail with `NCCL error: unhandled cuda error`. Check `docker ps` on both
  nodes before every boot.
- **`nvidia-smi` on GB10 can report 96 % utilization at 17 W** after a process dies; it is a stale
  reading, not a wedged GPU (a 4096² matmul completes in 0.5 s).
- **Streaming chunks ≠ tokens.** With DFlash2 each SSE chunk carries ~3 tokens; count
  `usage.completion_tokens`, not chunks, or every decode number is 3× too low.
- **Greedy decoding is not reproducible run-to-run on this stack** (atomicAdd scatter in the fat-expert path): two identical
  `temperature=0` requests diverge after a dozen lines. "Identical output" is not a regression test here; use a teacher-forced
  logprob panel and read it against a same-boot or inter-boot noise floor (same config repeated: top-1 ~94–96 %, KL median
  ~0.002–0.003; different boots: ~93.5 %, ~0.005).
- **The first measurement after boot under-reads by 5–15 %** (JIT / autotune), even after `warmup.sh`. Run a discarded pass at
  every prompt size before the pass you keep.
- **An env knob is only real if the launcher forwards it.** `GLM53_PERSISTENT_TOPK_MAX_LEN` is read inside the container by
  `patch_kpool_topk_fallback.py`; until 2026-09-18 neither `run.sh` nor `supervise.sh` passed it through, so a 500K test booted
  three times "with the fallback forced" while running the persistent kernel, and failed each time. Check `docker inspect` env.
- **`kernel_union == kernel_sum` in a profile means nothing overlaps.** On this build every kernel, NCCL included, is serialized:
  each ms of Python glue or communication is wall time. That is how 260 ms of eager fp32 merge per 4608-token chunk hid for
  two weeks behind "structural" explanations (KDA, dense GEMM, context regime) that all measured neutral.
- **A hot toggle beats a reboot for A/B on this hardware**: boot-to-boot variance is ±5 %, most of the effects worth chasing are
  smaller. Both 2026-09-18 fixes read an `.off` file in the JIT cache so both arms run on one boot.
- **Never `copy_()` to the GPU straight from a file-backed mmap tensor on GB10.** Under a CUDA
  context that path runs at ~0.1 GB/s (63 s for 5.2 GiB) on the 4 KiB DGX OS kernel; `clone()` (or
  `pin_memory()`) into anonymous memory first and it is disk-bound again. This alone was the
  274 s vs 101 s head/worker asymmetry of `Loading weights took` — the two ranks just differ in how
  many tensors are still contiguous after the TP narrow. The standalone benchmark that "proved" the
  mmap path was fast had no CUDA context.
- **`$( [ -n "$V" ] && echo … )` inside a bash array kills a `set -e` script when `V` is empty** (the
  assignment takes the substitution's exit status). Write `[ -z "$V" ] || echo …`. Cost one silent
  non-boot: both containers absent, empty logs, supervisor polling `/health` until its timeout.
- **Multi-node vLLM without `VLLM_HOST_IP` puts the control plane on the default route.** NCCL was on
  RoCE, but the ZMQ queues between the EngineCore and the remote worker (one `SchedulerOutput` per
  step) were bound on the node's default-route IP — Wi-Fi here (power save on, 140 ms spikes). Nothing
  logs it; c1 looks fine. At c3-c4 rank 1 got each step late: ~0.5 s freezes every 2-3 s, 15-22 % of
  decode time. Check with `ss -tnp` inside the container; `run.sh` now pins it to the fabric
  (`FABRIC_HOST_IP=1`).
- **Benchmarking at T=0 hides the regime clients use.** Most OpenAI clients send no temperature, so
  the server's `generation_config` (T=1.0 / top_p 0.95) applies; acceptance at 128K drops from 3.79 to
  2.68 tokens/step for agent/code. Greedy drafting is the wrong default there (see `DRAFT_SAMPLE`).
- **Greedy output is not reproducible on this stack** (E3 atomicAdd scatter): never use two greedy
  runs as an equality test; compare with the KL panel against a same-boot noise floor.
