"""Fixed-shape cooperative MoE adapter for the GLM-5.3-Flash TP2 EXL3 overlay.

Decode-sized fused exl3_moe only (tiny M): K4 MCG, H=4096, local I=1024, top-k 8,
1..32 physical rows. Prefill and unsupported shapes stay stock, including E3
grouped fat experts. Allocate/prepare only after weight load, never during graph
capture. Disabled unless explicitly selected.

Shared-scratch contract: install() caches one CoopLaunch per device and attaches
it to every eligible layer. The workspace is reused across layers and invocations.
Kernel A resets B's counters; kernel B resets A's counters for the next *ordered*
call. Overlapping cooperative launches on different CUDA streams are unsupported.
Ubatching and dual-batch overlap are rejected at configuration time. Per-layer
scratch replication is not the preferred fix on this memory-constrained kit.

The DS4.1 cooperative .so is compile-time locked to DeepSeek-V4.1 and must not
be loaded here. This adapter talks to the GLM specialization's glm53_coop_* ABI.
"""

import ctypes as C
import hashlib
import math
import os
import re
from pathlib import Path
import torch

# Our rebuild of the same pinned kernel source (turboderp 58d4d732 / MiaAI commit f4a505c)
# with this stack's nvcc 13.0.88. The upstream-published digest (aa3fe5e9...) was built
# with a different toolchain, so a clean rebuild differs; repinned here after the
# host-side tests and the GPU integration gate. Keep the check: an unvalidated
# binary is refused.
SHA256 = "bfc4cfae36ae147010566b6cdb5db48b153dc0cbf1c13403b23f07b0bf6cb0de"
PTR_KEYS = (
    "gate_trellis",
    "gate_suh",
    "gate_svh",
    "up_trellis",
    "up_suh",
    "up_svh",
    "down_trellis",
    "down_suh",
    "down_svh",
)
HIDDEN = 4096
INTERMEDIATE_LOCAL = 1024
TOPK = 8
ROWS_MAX = 32
SLOTS_MAX = TOPK * ROWS_MAX
EXPERTS_MAX = 288
BITS = 4
# DS4.1's working serve passes geometry 1 (A wide, B wide). Blackwell auto
# (geometry 2) makes B narrow because I/16=64, which sizes the down grid as
# rows*topk*(H/32) = 8192 blocks at DFlash2 ×1. Wide B uses H/128 → 2048.
# Stock fused EXL3 is already near this kit's copy ceiling; extra B blocks
# and 32-column down tiles are overhead with almost no expert-weight reuse
# (8 tokens × 8 top-k into 288 local experts). Override with GLM53_COOP_GEOMETRY
# before native preparation and graph capture; it is not a live graph switch.
DEFAULT_GEOMETRY = 1
GEOMETRY = DEFAULT_GEOMETRY
GEOMETRY_TILES = {
    0: "A-narrow/B-narrow",
    1: "A-wide/B-wide",
    2: "A-wide/B-narrow",
}
CTR_LEN = SLOTS_MAX * (INTERMEDIATE_LOCAL // 128) + ROWS_MAX * (HIDDEN // 128) + 2 + (
    SLOTS_MAX + 1
) + SLOTS_MAX
PARAMS_SIZE = 344
_OVERLAP_ENV = (
    "VLLM_UBATCH",
    "VLLM_UBATCHING",
    "VLLM_ENABLE_UBATCHING",
    "VLLM_USE_UBATCHING",
    "VLLM_V1_UBATCH",
    "VLLM_V1_ENABLE_UBATCHING",
    "VLLM_DUAL_BATCH",
    "VLLM_DUAL_BATCH_OVERLAP",
    "VLLM_ENABLE_DUAL_BATCH",
    "VLLM_ENABLE_DBO",
    "VLLM_DBO",
)
_OVERLAP_ARG_RE = re.compile(
    r"(--ubatch(?:ing)?(?:\b|=)|--enable-ubatch(?:ing)?\b|--dual-batch(?:-overlap)?\b|--enable-dbo\b)",
    re.IGNORECASE,
)


class CooperativeMoEError(RuntimeError):
    """Required cooperative-MoE initialization or configuration failure."""


class CoopDiagnostics:
    """Host-side selection log. Capture entries identify graph contents.

    Eager and capture-time counts are not graph-replay counts. CUDA graph
    replay does not re-enter this Python wrapper.
    """

    def __init__(self):
        self.geometry = None
        self.geometry_name = None
        self.native_prepared = False
        self.eligible_layers = 0
        self.ineligible_layers = 0
        self.ineligible_reasons = {}
        self.capture_selection = {}
        self.eager_selection = {}
        self.summary_logged = False
        self.seen_selection = set()


def resolve_geometry(raw=None):
    if raw is None:
        raw = os.environ.get("GLM53_COOP_GEOMETRY", "")
    text = "" if raw is None else str(raw).strip()
    if not text:
        return DEFAULT_GEOMETRY
    try:
        geometry = int(text, 10)
    except ValueError as exc:
        raise CooperativeMoEError(
            f"GLM53_COOP_GEOMETRY must be 0, 1, or 2 (got {raw!r})"
        ) from exc
    if geometry not in GEOMETRY_TILES:
        raise CooperativeMoEError(
            f"GLM53_COOP_GEOMETRY must be 0, 1, or 2 (got {geometry})"
        )
    return geometry


def enforce_serialized_execution(extra_args=None):
    """Reject configs that would overlap cooperative launches on shared scratch."""
    for key in _OVERLAP_ENV:
        value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            continue
        lowered = str(value).strip().lower()
        if lowered in ("0", "off", "false", "no"):
            continue
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported {key}={value!r}"
        )
    extra = os.environ.get("EXTRA_ARGS", "") if extra_args is None else extra_args
    if extra and _OVERLAP_ARG_RE.search(str(extra)):
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported EXTRA_ARGS={extra!r}"
        )


def layer_eligible(layer):
    bits = int(getattr(layer, "_exl3_bits", 0) or getattr(layer, "_exl3_k", 0) or 0)
    inners = getattr(layer, "_exl3_inners", None)
    if (
        bits != BITS
        or getattr(layer, "_exl3_hidden_size", None) != HIDDEN
        or getattr(layer, "_exl3_intermediate_local", None) != INTERMEDIATE_LOCAL
        or not inners
        or not 1 <= len(inners) <= EXPERTS_MAX
        or not getattr(layer, "_exl3_ptrs", None)
        or getattr(layer, "_exl3_fused_temps", None) is None
    ):
        return False
    return all(
        int(pack[p].K) == BITS
        and bool(pack[p].mcg)
        and not bool(pack[p].mul1)
        for pack in inners
        for p in ("gate", "up", "down")
    )


def layer_ineligible_reason(layer):
    bits = int(getattr(layer, "_exl3_bits", 0) or getattr(layer, "_exl3_k", 0) or 0)
    inners = getattr(layer, "_exl3_inners", None)
    if bits != BITS:
        return f"bits_{bits}"
    if getattr(layer, "_exl3_hidden_size", None) != HIDDEN:
        return "hidden"
    if getattr(layer, "_exl3_intermediate_local", None) != INTERMEDIATE_LOCAL:
        return "intermediate_local"
    if not inners:
        return "no_inners"
    if not 1 <= len(inners) <= EXPERTS_MAX:
        return f"expert_count_{len(inners)}"
    if not getattr(layer, "_exl3_ptrs", None):
        return "no_ptrs"
    if getattr(layer, "_exl3_fused_temps", None) is None:
        return "no_fused_temps"
    if not layer_eligible(layer):
        return "quant_not_k4_mcg"
    return None


def call_ineligible_reason(x, ids, weights, layer, limit):
    native = getattr(layer, "_glm53_coop_native", None)
    if native is None:
        return "native_unattached"
    if len(getattr(x, "shape", ())) != 2:
        return "input_rank"
    rows = int(x.shape[0])
    if not 1 <= rows <= ROWS_MAX:
        return "rows_out_of_range"
    if x.shape[1] != HIDDEN:
        return "hidden_mismatch"
    if tuple(ids.shape) != (rows, TOPK) or tuple(weights.shape) != (rows, TOPK):
        return "route_shape"
    if not getattr(x, "is_cuda", False):
        return "input_not_cuda"
    device = getattr(native, "device", None)
    if ids.device != device or weights.device != device or x.device != device:
        return "device_mismatch"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input_dtype"
    if ids.dtype != torch.int64:
        return "ids_dtype"
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "weights_dtype"
    try:
        finite_limit = math.isfinite(float(limit)) and float(limit) >= 0
    except (TypeError, ValueError):
        finite_limit = False
    if not finite_limit:
        return "limit_invalid"
    return None


def call_eligible(x, ids, weights, layer, limit):
    return call_ineligible_reason(x, ids, weights, layer, limit) is None


def _require(condition, message):
    if not condition:
        raise CooperativeMoEError(message)


class CoopLaunch:
    def __init__(self, device, library_root, geometry=None):
        self.device = device
        self.geometry = DEFAULT_GEOMETRY if geometry is None else int(geometry)
        _require(
            self.geometry in GEOMETRY_TILES,
            f"unsupported cooperative geometry {self.geometry}",
        )
        capability = torch.cuda.get_device_capability(device)
        _require(
            capability == (12, 1),
            f"cooperative MoE requires SM121, got {capability}",
        )
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "initialize cooperative MoE before graph capture",
        )
        path = Path(library_root) / "cooperative_moe.so"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            digest == SHA256,
            f"unvalidated cooperative_moe.so digest {digest}",
        )
        self.library = C.CDLL(str(path))
        abi = int(self.library.glm53_coop_abi())
        _require(abi == 1, f"unexpected glm53_coop_abi {abi}")
        self.launch = self.library.glm53_coop_launch
        self.launch.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.launch.restype = C.c_int
        with torch.cuda.device(device):
            info = (C.c_int * 18)()
            status = int(self.library.glm53_coop_info(BITS, self.geometry, info))
            _require(status == 0, f"glm53_coop_info failed status={status}")
            _require(
                info[16] == CTR_LEN and info[17] == PARAMS_SIZE,
                f"native layout mismatch ctr={info[16]} params={info[17]}",
            )
            _require(
                info[4] >= 1 and info[9] >= 1 and info[14] >= 1,
                "cooperative kernel occupancy < 1",
            )
            self.occupancy = (int(info[4]), int(info[9]), int(info[14]))
            self.scratch = [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in (
                    ((SLOTS_MAX, HIDDEN), torch.float16),
                    ((SLOTS_MAX, HIDDEN), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, INTERMEDIATE_LOCAL), torch.float16),
                    ((SLOTS_MAX, HIDDEN), torch.float32),
                )
            ]
            self.counters = torch.zeros(CTR_LEN, dtype=torch.int32, device=device)

    def __call__(self, module, x, ids, weights, layer, inners, expert_map, limit):
        _require(
            call_eligible(x, ids, weights, layer, limit),
            "cooperative launch preconditions failed",
        )
        rows = int(x.shape[0])
        experts = len(inners)
        xh = x.contiguous().half()
        local = (
            module.map_topk_to_local(ids, experts, expert_map)
            .reshape(rows, TOPK)
            .contiguous()
        )
        rw = weights.to(dtype=torch.float16).contiguous()
        out = torch.empty((rows, HIDDEN), dtype=torch.float32, device=x.device)
        tables = [layer._exl3_ptrs[k] for k in PTR_KEYS]
        _require(
            all(
                t.device == x.device
                and t.dtype == torch.int64
                and t.is_contiguous()
                and tuple(t.shape) == (experts,)
                for t in tables
            ),
            "cooperative pointer tables are not contiguous int64 expert vectors",
        )
        tensors = [xh, local, rw, *tables, *self.scratch, self.counters, out]
        pointers = (C.c_void_p * 20)(*[t.data_ptr() for t in tensors])
        status = self.launch(
            pointers,
            BITS,
            rows,
            experts,
            float(limit),
            self.geometry,
            0,
            C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            # Never retry stock after a partially launched CUDA operation.
            raise RuntimeError(f"cooperative MoE kernel CUDA launch status {status}")
        layer._exl3_last_fat_fallback = "none"
        layer._exl3_last_fat_reason = "no_fat_experts"
        return out


def _log_selection(module, diag, rows, selected, reason, capturing):
    kind = "capture" if capturing else "eager"
    key = (kind, int(rows), bool(selected), str(reason))
    if key in diag.seen_selection:
        return
    diag.seen_selection.add(key)
    record = {"selected": bool(selected), "reason": reason, "kind": kind}
    store = diag.capture_selection if capturing else diag.eager_selection
    store[int(rows)] = record
    module.logger.info(
        "cooperative MoE %s-time selection rows=%s selected=%s reason=%s "
        "geometry=%s (%s); this is not a CUDA-graph replay count"
        % (
            kind,
            rows,
            bool(selected),
            reason,
            diag.geometry,
            diag.geometry_name,
        )
    )


def install(module, library_root="/root/.cache/vllm/cooperative_moe", *, enabled=False):
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not enabled and os.environ.get("GLM53_COOPERATIVE_MOE", "0") != "1":
        return False
    if getattr(module, "_glm53_coop_installed", False):
        raise RuntimeError("cooperative MoE already installed")
    enforce_serialized_execution()
    geometry = resolve_geometry()
    original_process = module.Exl3MoEMethod.process_weights_after_loading
    original_apply = module.apply_exl3_fused_moe
    kernels = {}
    diag = CoopDiagnostics()
    diag.geometry = geometry
    diag.geometry_name = GEOMETRY_TILES[geometry]

    def process(method, layer):
        result = original_process(method, layer)
        if hasattr(layer, "_glm53_coop_native"):
            delattr(layer, "_glm53_coop_native")
        if layer_eligible(layer):
            device = layer.w13_trellis.device
            key = str(device)
            if key not in kernels:
                kernels[key] = CoopLaunch(device, library_root, geometry=geometry)
                diag.native_prepared = True
                launch = kernels[key]
                module.logger.info(
                    "cooperative MoE native prepared: geometry=%s (%s) "
                    "occupancy_a/b/rot=%s/%s/%s library=%s"
                    % (
                        geometry,
                        GEOMETRY_TILES[geometry],
                        launch.occupancy[0],
                        launch.occupancy[1],
                        launch.occupancy[2],
                        library_root,
                    )
                )
            layer._glm53_coop_native = kernels[key]
            diag.eligible_layers += 1
        else:
            reason = layer_ineligible_reason(layer) or "ineligible"
            diag.ineligible_layers += 1
            diag.ineligible_reasons[reason] = diag.ineligible_reasons.get(reason, 0) + 1
        return result

    def apply(x, ids, weights, layer, inners, expert_map, limit):
        if not diag.summary_logged:
            diag.summary_logged = True
            module.logger.info(
                "cooperative MoE prepared-layer summary: eligible=%s ineligible=%s "
                "ineligible_reasons=%s geometry=%s (%s) native_prepared=%s"
                % (
                    diag.eligible_layers,
                    diag.ineligible_layers,
                    diag.ineligible_reasons or "{}",
                    geometry,
                    GEOMETRY_TILES[geometry],
                    diag.native_prepared,
                )
            )
        capturing = bool(torch.cuda.is_current_stream_capturing())
        reason = call_ineligible_reason(x, ids, weights, layer, limit)
        selected = reason is None
        rows = int(getattr(x, "shape", [0])[0]) if getattr(x, "shape", None) else 0
        _log_selection(
            module,
            diag,
            rows,
            selected,
            "cooperative" if selected else reason,
            capturing,
        )
        if not selected:
            return original_apply(x, ids, weights, layer, inners, expert_map, limit)
        return layer._glm53_coop_native(
            module, x, ids, weights, layer, inners, expert_map, limit
        )

    module.Exl3MoEMethod.process_weights_after_loading = process
    module.apply_exl3_fused_moe = apply
    module._glm53_coop_original_apply = original_apply
    module._glm53_coop_installed = True
    module._glm53_coop_geometry = geometry
    module._glm53_coop_geometry_name = GEOMETRY_TILES[geometry]
    module._glm53_coop_diag = diag
    module.logger.info(
        "Fixed-shape cooperative MoE wrappers installed: K4 MCG, 1..32 rows, "
        "geometry=%s (%s); native prepare runs after weight load; "
        "prefill/E3 and other calls stay stock. Shared scratch must not overlap."
        % (geometry, GEOMETRY_TILES[geometry])
    )
    return True
