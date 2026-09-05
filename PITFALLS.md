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
- **First-request hang.** 2 boots out of 8 froze on the very first requests after `/health`
  (engine `TimeoutError` on the shm broadcast, worker silent, GPUs idle), never later and never
  reproduced on the same boot afterwards. Independent of the E2 kernel. Run `warmup.sh` right after
  boot; if it fails, stop and restart **both** ranks. Root cause not yet identified (concurrent
  JIT/autotune on both ranks is the leading suspect).
- **The first 8K prefill after boot is ~2× slower** (JIT/autotune); measure on the second pass.
- **`docker rm -f` deletes the container log.** `docker logs > file` first.
- **Two engines on one node.** A forgotten worker container from another stack holds ~98 GB and
  makes the next NCCL init fail with `NCCL error: unhandled cuda error`. Check `docker ps` on both
  nodes before every boot.
- **`nvidia-smi` on GB10 can report 96 % utilization at 17 W** after a process dies; it is a stale
  reading, not a wedged GPU (a 4096² matmul completes in 0.5 s).
- **Streaming chunks ≠ tokens.** With DFlash2 each SSE chunk carries ~3 tokens; count
  `usage.completion_tokens`, not chunks, or every decode number is 3× too low.
