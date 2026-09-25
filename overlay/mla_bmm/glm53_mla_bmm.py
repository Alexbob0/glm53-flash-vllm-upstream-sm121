"""[2026-09-25] MLA_BMM: Triton bmm for MLA's absorbed W_UK_T / W_UV projections at prefill (sm121: cuBLAS picks
cutlass_80_wmma at ~20 TFLOPS; Triton tl.dot ~36 TFLOPS, same bf16 rounding), and q written directly into the buffer padded
to 512+64 that FLASHINFER_MLA_SPARSE_SM120 expects (removes the cat then the pad of q, two full copies).
Only for B >= GLM53_MLA_BMM_MIN_TOKENS (default 256) and outside CUDA-graph capture; decode keeps cuBLAS.
Hot kill switch: /root/.cache/vllm/glm53_mla_bmm.off (re-read every 64 calls); GLM53_MLA_BMM=0 = stock."""
import os

import torch
import triton
import triton.language as tl

_ENV = os.environ.get("GLM53_MLA_BMM", "1") == "1"
_MIN_TOKENS = int(os.environ.get("GLM53_MLA_BMM_MIN_TOKENS", "256"))
_OFF_FILE = "/root/.cache/vllm/glm53_mla_bmm.off"
_state = {"calls": 0, "on": _ENV, "logged": None}


@triton.jit
def _bmm_kernel(a, b, c, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_b = tl.program_id(1)
    pid = tl.program_id(0)
    nn = tl.cdiv(N, BN)
    pm = pid // nn
    pn = pid % nn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    A = a + pid_b.to(tl.int64) * sab + rm[:, None].to(tl.int64) * sam + rk[None, :] * sak
    B = b + pid_b.to(tl.int64) * sbb + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(A, mask=rm[:, None] < M, other=0.0), tl.load(B))
        A += BK * sak
        B += BK * sbk
    C = c + pid_b.to(tl.int64) * scb + rm[:, None].to(tl.int64) * scm + rn[None, :] * scn
    tl.store(C, acc.to(c.dtype.element_ty), mask=rm[:, None] < M)


def active(num_tokens: int, logger=None) -> bool:
    st = _state
    if st["calls"] % 64 == 0:
        st["on"] = _ENV and not os.path.exists(_OFF_FILE)
    st["calls"] += 1
    if logger is not None and st["logged"] != st["on"]:
        st["logged"] = st["on"]
        logger.info("[glm53-mla-bmm] %s (min tokens %d)", "ON" if st["on"] else "OFF", _MIN_TOKENS)
    return st["on"] and num_tokens >= _MIN_TOKENS and not torch.cuda.is_current_stream_capturing()


def bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, bn: int = 128) -> torch.Tensor:
    """out[i] = a[i] @ b[i] ; a (Bt, M, K), b (Bt, K, N), out (Bt, M, N), strides quelconques ; K % 64 == 0, N % bn == 0."""
    bt, m, k = a.shape
    n = b.shape[2]
    assert k % 64 == 0 and n % bn == 0 and b.shape[1] == k and out.shape == (bt, m, n)
    grid = (triton.cdiv(m, 64) * (n // bn), bt)
    _bmm_kernel[grid](a, b, out, m, n, k, *a.stride(), *b.stride(), *out.stride(),
                      BM=64, BN=bn, BK=64, num_warps=4, num_stages=3)
    return out
