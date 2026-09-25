# SPDX-License-Identifier: Apache-2.0
"""[2026-09-25] GLM53_NGRAM: hybrid DFlash2 + n-gram prompt-lookup draft (GPU, no host sync).

After DFlash2Speculator.propose(), for each request: if its last n tokens (n >= GLM53_NGRAM_MIN, default 8, searched up
to 16) appear earlier in its history (prompt + output, `req_states.all_token_ids`), the k tokens that followed the
longest, then latest, occurrence replace the DFlash2 draft at those positions.
Lossless: the draft distribution cached for probabilistic verification becomes a one-hot on the proposed token
(sparse DFlash2 cache: its top-k candidates are cleared and logit 0 is written for the token), so the standard
rejection sampler applies min(1, p/q) with q = 1. Uncovered positions: DFlash2 draft untouched.
Offline simulation on real agent sessions: +9 % tokens/step, +15 % on GLM tool calls. MEASURED NEUTRAL in serving
(DFlash2 already copies from context), hence opt-in. Hot toggle: /root/.cache/vllm/glm53_ngram.off (re-read every 64 steps).
"""
import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("GLM53_NGRAM", "0") == "1"
_NMIN = int(os.environ.get("GLM53_NGRAM_MIN", "8"))
_NMAX = 16
_LOG_EVERY = int(os.environ.get("GLM53_NGRAM_LOG", "2048"))
_OFF_FILE = "/root/.cache/vllm/glm53_ngram.off"
_CHUNK = 2048

_state = {"calls": 0, "on": _ENABLED, "logged": False, "best": None, "stats": None}


@triton.jit
def _ngram_match_kernel(
    tokens_ptr, tokens_stride,       # all_token_ids [max_num_reqs, max_model_len] int32
    total_len_ptr,                   # [max_num_reqs] int32
    idx_mapping_ptr,                 # [num_reqs] int32 (batch -> request state)
    best_ptr,                        # [num_reqs] int64, init -1 : (longueur << 32) | fin
    NMIN: tl.constexpr, NMAX: tl.constexpr, CHUNK: tl.constexpr,
):
    i = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.load(idx_mapping_ptr + i)
    L = tl.load(total_len_ptr + s)
    lo = c * CHUNK + NMIN            # candidate end e: e - NMIN >= 0 and e <= L - 1 (at least 1 following token)
    if lo > L - 1:
        return
    base = tokens_ptr + s.to(tl.int64) * tokens_stride
    e = lo + tl.arange(0, CHUNK)
    valid = e <= L - 1
    m = tl.zeros([CHUNK], dtype=tl.int32)
    alive = valid
    for j in tl.static_range(1, NMAX + 1):
        ok = alive & (e - j >= 0) & (L - j >= 0)
        a = tl.load(base + e - j, mask=ok, other=-1)
        b = tl.load(base + L - j, mask=L - j >= 0, other=-2)
        eq = ok & (a == b)
        m += eq.to(tl.int32)
        alive = eq
    key = tl.where(m >= NMIN, (m.to(tl.int64) << 32) | e.to(tl.int64), -1)
    tl.atomic_max(best_ptr + i, tl.max(key, axis=0))


@triton.jit
def _ngram_apply_kernel(
    tokens_ptr, tokens_stride, total_len_ptr, idx_mapping_ptr, best_ptr,
    draft_ptr, draft_stride,         # draft_tokens [num_reqs, K] int64 (view of the speculator buffer)
    logits_ptr, logits_s0, logits_s1,   # draft_logits [max_num_reqs, K, V] fp32 (or dummy)
    cand_ptr,                        # _cached_candidate_ids [max_num_reqs, K, TOPK] int64 (or dummy)
    stats_ptr,                       # [3] int64: requests seen, requests substituted, tokens substituted
    K: tl.constexpr, TOPK: tl.constexpr, BLOCK_T: tl.constexpr, HAS_LOGITS: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    best = tl.load(best_ptr + i)
    if j == 0:
        tl.atomic_add(stats_ptr, 1)
    if best < 0:
        return
    s = tl.load(idx_mapping_ptr + i)
    L = tl.load(total_len_ptr + s)
    e = (best & 0xFFFFFFFF).to(tl.int32)
    count = tl.minimum(L - e, K)
    if j >= count:
        return
    tok = tl.load(tokens_ptr + s.to(tl.int64) * tokens_stride + e + j).to(tl.int64)
    tl.store(draft_ptr + i * draft_stride + j, tok)
    if j == 0:
        tl.atomic_add(stats_ptr + 1, 1)
    tl.atomic_add(stats_ptr + 2, 1)
    if HAS_LOGITS:
        row = logits_ptr + s.to(tl.int64) * logits_s0 + j * logits_s1
        offs = tl.arange(0, BLOCK_T)
        msk = offs < TOPK
        cbase = cand_ptr + (s.to(tl.int64) * K + j) * TOPK
        old = tl.load(cbase + offs, mask=msk, other=0)
        tl.store(row + old, float("-inf"), mask=msk)
        tl.debug_barrier()
        tl.store(row + tok, 0.0)
        tl.store(cbase + offs, tok + tl.zeros([BLOCK_T], dtype=tl.int64), mask=msk)


def maybe_override(draft_tokens: torch.Tensor, input_batch, req_states, speculator) -> None:
    """Replaces DFlash2 drafts in place with an n-gram continuation when one exists."""
    st = _state
    if not _ENABLED:
        return
    st["calls"] += 1
    if st["calls"] % 64 == 1:
        st["on"] = not os.path.exists(_OFF_FILE)
    if not st["on"]:
        return
    num_reqs = draft_tokens.shape[0]
    if num_reqs == 0:
        return
    dev = draft_tokens.device
    if st["best"] is None or st["best"].numel() < req_states.max_num_reqs:
        st["best"] = torch.empty(req_states.max_num_reqs, dtype=torch.int64, device=dev)
        st["stats"] = torch.zeros(3, dtype=torch.int64, device=dev)
    best = st["best"][:num_reqs]
    best.fill_(-1)
    tokens = req_states.all_token_ids.gpu
    total_len = req_states.total_len.gpu
    idx = input_batch.idx_mapping
    K = draft_tokens.shape[1]
    nchunk = triton.cdiv(tokens.shape[1], _CHUNK)
    _ngram_match_kernel[(num_reqs, nchunk)](
        tokens, tokens.stride(0), total_len, idx, best,
        NMIN=_NMIN, NMAX=_NMAX, CHUNK=_CHUNK,
    )
    logits = getattr(speculator, "draft_logits", None)
    cand = getattr(speculator, "_cached_candidate_ids", None)
    has_logits = logits is not None
    if has_logits and cand is None:
        if not st["logged"]:
            logger.warning("[glm53-ngram] draft cache is not sparse (not DFlash2): disabled")
            st["logged"] = True
        return
    topk = cand.shape[-1] if cand is not None else 1
    _ngram_apply_kernel[(num_reqs, K)](
        tokens, tokens.stride(0), total_len, idx, best,
        draft_tokens, draft_tokens.stride(0),
        logits if has_logits else best, logits.stride(0) if has_logits else 0, logits.stride(1) if has_logits else 0,
        cand if has_logits else best,
        st["stats"],
        K=K, TOPK=topk, BLOCK_T=triton.next_power_of_2(max(topk, 2)), HAS_LOGITS=has_logits,
    )
    if not st["logged"]:
        st["logged"] = True
        logger.info("[glm53-ngram] actif : n >= %d (max %d), k = %d, top-k cache = %d, probabiliste = %s",
                    _NMIN, _NMAX, K, topk, has_logits)
    if _LOG_EVERY and st["calls"] % _LOG_EVERY == 0:
        seen, hit, ntok = st["stats"].tolist()
        logger.info("[glm53-ngram] request-steps %d, substituted %d (%.1f %%), tokens substituted %d",
                    seen, hit, 100.0 * hit / max(seen, 1), ntok)
