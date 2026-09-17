"""Prepared integration gate; run only in a confirmed maintenance window.

Exercises the actual overlay + Torch tensors/CUDA graphs, not CPU dispatch stubs.
Numerical screening includes the historical peak-normalized check plus per-row
and relative-L2 diagnostics. This is not bit-exact with stock and is not by
itself a model-quality proof.
"""

import json
import os
import sys

assert (
    os.environ.get("GLM53_COOP_MAINTENANCE_TEST") == "1"
), "requires an explicit maintenance test"
os.environ.update(
    EXL3_FUSED_MOE="1",
    EXL3_FAT_GROUPED="1",
    EXL3_FAT_KERNEL="0",
    EXL3_TEMP_ROWS_FUSED="32",
    EXL3_FAT_EXPERT_LOG="0",
)
sys.path.insert(0, "/opt/glm53")
import torch
import test_exl3_overlay as tests
from vllm.model_executor.layers.quantization import exl3

assert exl3._glm53_coop_installed
wrapper = exl3.apply_exl3_fused_moe
stock = exl3._glm53_coop_original_apply
checks = 0
strict_raw_failures = 0
strict_bf16_failures = 0
CAPTURE_ROWS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32)
ALL_ROWS = CAPTURE_ROWS + (7, 33, 40)
SANITIZER = os.environ.get("GLM53_COOP_SANITIZER") == "1"
if SANITIZER:
    CAPTURE_ROWS = (1, 8, 32)
    ALL_ROWS = (1, 8, 32, 40)
PEAK_TOL = 0.003
ROW_PEAK_TOL = 0.05
REL_L2_TOL = 0.05


def stats(actual, reference):
    delta = (actual - reference).abs()
    peak = reference.abs().max().clamp_min(1e-30)
    ref_l2 = reference.float().norm().clamp_min(1e-30)
    rel_l2 = float((actual.float() - reference.float()).norm() / ref_l2)
    row_peaks = reference.abs().amax(dim=1).clamp_min(1e-30)
    row_rel = (delta.amax(dim=1) / row_peaks).tolist()
    return {
        "peak_rel": float(delta.max() / peak),
        "rel_l2": rel_l2,
        "row_rel_max": float(max(row_rel)),
        "row_rel_mean": float(sum(row_rel) / len(row_rel)),
    }


violations = []
peak_samples = []


def compare(actual, reference, *, label, require_peak=True):
    global strict_raw_failures, strict_bf16_failures
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), label
    summary = stats(actual, reference)
    summary["label"] = label
    peak_samples.append(summary)
    if summary["rel_l2"] > REL_L2_TOL:
        violations.append(("rel_l2", label, summary))
    if summary["row_rel_max"] > ROW_PEAK_TOL:
        violations.append(("row_peak", label, summary))
    if summary["peak_rel"] > PEAK_TOL:
        violations.append(("peak", label, summary))
        if require_peak:
            raise AssertionError(f"failed peak-normalized screen {summary}")
    delta = (actual - reference).abs()
    strict_raw_failures += int((delta > 1e-3 + 1e-3 * reference.abs()).sum())
    a, r = actual.bfloat16().float(), reference.bfloat16().float()
    strict_bf16_failures += int(((a - r).abs() > 1e-3 + 1e-3 * r.abs()).sum())
    return summary


def make_routes(rows, n_exp, concentrated, invalid=False, zero_weight=False):
    if concentrated:
        ids = torch.randperm(n_exp, device="cuda")[:8].repeat(rows, 1)
    else:
        ids = torch.stack([torch.randperm(n_exp, device="cuda")[:8] for _ in range(rows)])
    if invalid:
        ids.fill_(-1)
    weights = torch.rand(rows, 8, device="cuda").softmax(-1)
    if zero_weight:
        weights.zero_()
    return ids, weights


def run_case(owner, native, rows, n_exp, concentrated, limit=10.0, mutate=True):
    global checks
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    ids, weights = make_routes(rows, n_exp, concentrated)

    def run():
        return exl3.apply_exl3_experts(
            x.float(), ids, weights, owner, fused=True, limit=limit
        )

    grouped_before = exl3.exl3_fat_diag()["grouped_calls"]
    exl3.apply_exl3_fused_moe = stock
    baseline = run()
    calls = []
    original_native = native.launch

    def counted(*args):
        calls.append(True)
        return original_native(*args)

    native.launch = counted
    exl3.apply_exl3_fused_moe = wrapper
    actual = run()
    torch.cuda.synchronize()
    selected = 1 <= rows <= 32
    assert bool(calls) == selected, (rows, bool(calls), selected)
    summary = compare(
        actual,
        baseline,
        label=f"eager-vs-stock rows={rows} conc={concentrated} limit={limit}",
        require_peak=False,
    )
    if rows > 32:
        print(
            json.dumps(
                {
                    "stage": "fallback",
                    "rows": rows,
                    "experts": n_exp,
                    "concentrated": concentrated,
                    "candidate_selected": selected,
                    "fat_fallback": owner._exl3_last_fat_fallback,
                    "fat_reason": getattr(owner, "_exl3_last_fat_reason", None),
                    "grouped_calls": exl3.exl3_fat_diag()["grouped_calls"],
                    "grouped_delta": (
                        exl3.exl3_fat_diag()["grouped_calls"] - grouped_before
                    ),
                }
            ),
            flush=True,
        )
    if rows > 32 and concentrated:
        assert owner._exl3_last_fat_fallback == "grouped", (
            "oversized concentrated routing must execute E3 grouped, got "
            f"{owner._exl3_last_fat_fallback} reason={owner._exl3_last_fat_reason}"
        )
        assert exl3.exl3_fat_diag()["grouped_calls"] > grouped_before
    if not SANITIZER and selected and rows in (1, 8) and limit == 10.0:
        loop_ref = exl3.apply_exl3_experts(
            x.float(), ids, weights, owner, fused=False, limit=limit
        )
        loop_summary = compare(
            actual,
            loop_ref,
            label=f"eager-vs-loop rows={rows} conc={concentrated} limit={limit}",
            require_peak=False,
        )
        print(
            json.dumps(
                {
                    "stage": "loop",
                    "rows": rows,
                    "concentrated": concentrated,
                    "limit": limit,
                    "status": "pass" if loop_summary["peak_rel"] <= PEAK_TOL else "recorded",
                    **loop_summary,
                }
            ),
            flush=True,
        )
    if SANITIZER and rows <= 32:
        saved_ids, saved_w = ids.clone(), weights.clone()
        ids.fill_(-1)
        invalid_out = run()
        torch.cuda.synchronize()
        assert not torch.any(invalid_out), (rows, "sanitizer invalid routes")
        ids.copy_(saved_ids)
        weights.zero_()
        zero_out = run()
        torch.cuda.synchronize()
        assert not torch.any(zero_out), (rows, "sanitizer zero weights")
        ids.copy_(saved_ids)
        weights.copy_(saved_w)
        print(
            json.dumps(
                {
                    "stage": "sanitizer-routes",
                    "rows": rows,
                    "concentrated": concentrated,
                    "invalid_zero": True,
                    "zero_weight_zero": True,
                }
            ),
            flush=True,
        )
    if rows <= 32 and not SANITIZER:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(captured, actual):
            graph_vs_eager = compare(
                captured,
                actual,
                label=f"replay-vs-eager rows={rows} conc={concentrated} limit={limit}",
                require_peak=False,
            )
            violations.append(
                (
                    "graph_vs_eager",
                    f"rows={rows} conc={concentrated} limit={limit}",
                    graph_vs_eager,
                )
            )
        for replay_i in range(4):
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(captured, actual):
                compare(
                    captured,
                    actual,
                    label=f"replay-vs-eager rows={rows} i={replay_i} limit={limit}",
                    require_peak=False,
                )
        if mutate:
            for scale in (0.03, 0.1, 0.6):
                x.copy_(torch.randn_like(x) * scale)
                ids.copy_(torch.randperm(n_exp, device="cuda")[:8].repeat(rows, 1))
                weights.copy_(torch.rand_like(weights).softmax(-1))
                exl3.apply_exl3_fused_moe = stock
                changed = run()
                exl3.apply_exl3_fused_moe = wrapper
                changed_candidate = run()
                compare(
                    changed_candidate,
                    changed,
                    label=f"eager-mut-vs-stock rows={rows} scale={scale}",
                    require_peak=False,
                )
                for replay_i in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(captured, changed_candidate):
                        compare(
                            captured,
                            changed_candidate,
                            label=f"replay-mut-vs-eager rows={rows} scale={scale} i={replay_i}",
                            require_peak=False,
                        )
            ids.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.any(captured)
            ids.copy_(torch.randperm(n_exp, device="cuda")[:8].repeat(rows, 1))
            weights.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.any(captured)
        del graph, captured
    native.launch = original_native
    checks += 1
    print(
        json.dumps(
            {
                "stage": "fixture",
                "bits": 4,
                "codebook": "mcg",
                "rows": rows,
                "experts": n_exp,
                "concentrated": concentrated,
                "limit": limit,
                "candidate_selected": selected,
                "experts_count": n_exp,
                "geometry": getattr(exl3, "_glm53_coop_geometry", None),
                "status": "pass",
                **summary,
            }
        ),
        flush=True,
    )


def inspect_apply_experts():
    import inspect

    try:
        return inspect.signature(exl3.apply_exl3_experts)
    except (TypeError, ValueError):
        return None


torch.manual_seed(20260916)
_, owner = tests._tiny_layer(
    torch.device("cuda"), n_exp=32, hidden=4096, inter=1024
)
native = getattr(owner, "_glm53_coop_native", None)
assert native is not None
assert owner._exl3_bits == 4
assert owner._exl3_hidden_size == 4096
assert owner._exl3_intermediate_local == 1024
assert getattr(exl3, "_glm53_coop_geometry", 1) in (0, 1, 2)
diag = getattr(exl3, "_glm53_coop_diag", None)
print(
    json.dumps(
        {
            "stage": "prepared",
            "geometry": getattr(exl3, "_glm53_coop_geometry", None),
            "geometry_name": getattr(exl3, "_glm53_coop_geometry_name", None),
            "eligible_layers": getattr(diag, "eligible_layers", None),
            "apply_exl3_experts": str(inspect_apply_experts()),
            "status": "ready",
        }
    ),
    flush=True,
)

# Shared-scratch reuse across alternating capture sizes and routing patterns.
order = []
for rows in ALL_ROWS:
    order.append((rows, False))
    order.append((rows, True))
if not SANITIZER:
    order.extend([(3, True), (32, False), (5, True), (1, False), (20, True)])

for rows, concentrated in order:
    torch.manual_seed(14000 + rows * 2 + int(concentrated))
    run_case(owner, native, rows, 32, concentrated, limit=10.0, mutate=not SANITIZER)

if not SANITIZER:
    # Activation-limit clipping and the zero-limit path.
    for limit in (0.0, 10.0, 0.5):
        torch.manual_seed(42 + int(limit * 10))
        run_case(owner, native, 8, 32, True, limit=limit, mutate=False)

    # Production expert-table width.
    _, wide = tests._tiny_layer(
        torch.device("cuda"), n_exp=288, hidden=4096, inter=1024
    )
    wide_native = getattr(wide, "_glm53_coop_native", None)
    assert wide_native is native, "scratch must stay per-device, not per-layer"
    for rows in (1, 8, 32, 40):
        torch.manual_seed(288000 + rows)
        run_case(wide, wide_native, rows, 288, True, limit=10.0, mutate=rows <= 8)

eager_peak_fail = [
    v
    for v in violations
    if v[0] == "peak"
    and "eager-vs-stock" in str(v[1])
    and "limit=10.0" in str(v[1])
    and v[2]["peak_rel"] > PEAK_TOL
]
eager_mut_peak = [
    v
    for v in violations
    if v[0] == "peak" and "eager-mut-vs-stock" in str(v[1])
]
clip_peak = [
    v
    for v in violations
    if v[0] == "peak" and "eager-vs-stock" in str(v[1]) and "limit=10.0" not in str(v[1])
]
graph_fail = [v for v in violations if v[0] == "graph_vs_eager"]
gross = [
    v
    for v in violations
    if v[0] == "peak" and v[2]["peak_rel"] > 0.01 and "eager-vs-loop" not in str(v[1])
]
status = "pass"
if eager_peak_fail or graph_fail or gross:
    status = "fail"
print(
    json.dumps(
        {
            "stage": "complete",
            "checks": checks,
            "status": status,
            "capture_rows": list(CAPTURE_ROWS),
            "strict_raw_failed_elements_retained": strict_raw_failures,
            "strict_post_bf16_failed_elements_retained": strict_bf16_failures,
            "clip_or_mut_peak_violations": len(eager_mut_peak) + len(clip_peak),
            "max_rel_l2": max((s["rel_l2"] for s in peak_samples), default=None),
            "violations": [
                {"kind": k, "label": lab, **summ} for k, lab, summ in violations
            ],
            "numerical_screen": (
                f"peak<={PEAK_TOL} of reference peak on eager stock compares; "
                f"rel_l2<={REL_L2_TOL}; per-row peak<={ROW_PEAK_TOL}; "
                "graph must match eager cooperative; strict differences retained"
            ),
            "distributed_serving_verified": False,
            "quality_eval_verified": False,
        }
    ),
    flush=True,
)
if status != "pass":
    raise SystemExit(1)
