"""[2026-09-25] FLASHKDA: GLM-5.3-Flash KDA prefill on the FlashKDA CUDA kernel (vllm/_flashkda_C, already built into the image,
used by Kimi-K3) instead of FLA Triton chunk_kda_with_fused_gate. Same recurrence (bounded gate lower_bound*sigmoid, l2norm q/k, beta
sigmoid inside the kernel, fp32 state). GLM microbench (H=32, D=128, T=4608): ~4.7 ms vs ~8 ms in production (FLA + glue), cos 0.99998.
Prefill outside CUDA-graph capture only. Hot kill switch: /root/.cache/vllm/glm53_flashkda.off (re-read every 64 calls);
GLM53_FLASHKDA=0 = original FLA path."""
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_ENV = os.environ.get("GLM53_FLASHKDA", "1") == "1"
_OFF_FILE = "/root/.cache/vllm/glm53_flashkda.off"
_state = {"calls": 0, "on": _ENV, "logged": None, "ws": {}}


def active() -> bool:
    st = _state
    if st["calls"] % 64 == 0:
        st["on"] = _ENV and not os.path.exists(_OFF_FILE)
    st["calls"] += 1
    if st["logged"] != st["on"]:
        st["logged"] = st["on"]
        logger.info("[glm53-flashkda] %s", "ON" if st["on"] else "OFF")
    return st["on"] and not torch.cuda.is_current_stream_capturing()


def _workspace(T: int, H: int, N: int, device: torch.device) -> torch.Tensor:
    size = int(torch.ops._flashkda_C.get_workspace_size(T, H, N))
    buf = _state["ws"].get(device)
    if buf is None or buf.numel() < size:
        buf = _state["ws"][device] = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
    return buf[:size]


def prefill(q, k, v, g, beta, A_log, dt_bias, lower_bound, initial_state, cu_seqlens, out=None):
    """q/k/v/g : (1, T, H, D) ; beta : (1, T, H) brut (bf16) ; initial_state : (N, H, D, D) fp32. Renvoie (out, final_state)."""
    import vllm._flashkda_C  # noqa: F401

    _, T, H, D = q.shape
    N = initial_state.shape[0]
    if out is None:
        out = torch.empty((1, T, H, D), dtype=q.dtype, device=q.device)
    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    torch.ops._flashkda_C.fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta,
        D ** -0.5, out, _workspace(T, H, N, q.device),
        A_log.reshape(-1).contiguous(), dt_bias.view(-1, D).contiguous(), float(lower_bound),
        initial_state.float().contiguous(), final_state, cu_seqlens.to(torch.int32).contiguous(), None, None,
    )
    return out, final_state
