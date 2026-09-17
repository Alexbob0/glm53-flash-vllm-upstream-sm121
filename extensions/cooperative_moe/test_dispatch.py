"""CPU stub tests of actual adapter dispatch; NOT Torch/CUDA integration proof."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch


def main():
    capturing = {"on": False}
    fake_torch = NS(
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        int64="i64",
        cuda=NS(is_current_stream_capturing=lambda: capturing["on"]),
    )
    spec = importlib.util.spec_from_file_location(
        "cooperative_moe_runtime", Path(__file__).with_name("runtime.py")
    )
    adapter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, torch=fake_torch):
        spec.loader.exec_module(adapter)
    calls = []
    native_creations = []

    class Native:
        def __init__(self, device, root, geometry=None):
            self.device = device
            self.geometry = 1 if geometry is None else geometry
            self.occupancy = (1, 1, 1)
            native_creations.append(device)

        def __call__(self, *args):
            calls.append("candidate")
            return "candidate"

    adapter.CoopLaunch = Native

    class Method:
        def process_weights_after_loading(self, layer):
            calls.append("process")
            return "loaded"

    def stock(*args):
        calls.append("stock")
        return "stock"

    module = NS(
        Exl3MoEMethod=Method,
        apply_exl3_fused_moe=stock,
        logger=NS(info=lambda *args: None),
    )
    with patch.dict(os.environ, {"GLM53_COOPERATIVE_MOE": "0"}):
        assert adapter.install(module) is False and module.apply_exl3_fused_moe is stock
    checks = 1

    def layer(bits=4, mcg=True, mul1=False, hidden=4096, inter=1024, n=8):
        packs = [
            {p: NS(K=bits, mul1=mul1, mcg=mcg) for p in ("gate", "up", "down")}
            for _ in range(n)
        ]
        return NS(
            _exl3_bits=bits,
            _exl3_k=bits,
            _exl3_hidden_size=hidden,
            _exl3_intermediate_local=inter,
            _exl3_inners=packs,
            _exl3_ptrs={"gate_trellis": True},
            _exl3_fused_temps=(True,),
            w13_trellis=NS(device="cuda:0"),
        )

    invalid_layers = [
        ("_exl3_bits", 3),
        ("_exl3_hidden_size", 5120),
        ("_exl3_intermediate_local", 2048),
        ("_exl3_inners", []),
        ("_exl3_ptrs", None),
        ("_exl3_fused_temps", None),
    ]
    for attr, value in invalid_layers:
        obj = layer()
        setattr(obj, attr, value)
        assert not adapter.layer_eligible(obj)
        checks += 1
    obj = layer()
    obj._exl3_inners[-1]["down"].mcg = False
    assert not adapter.layer_eligible(obj)
    checks += 1
    obj = layer()
    obj._exl3_inners[-1]["up"].mul1 = True
    assert not adapter.layer_eligible(obj)
    checks += 1
    obj = layer(bits=2, mcg=False, mul1=True)
    assert not adapter.layer_eligible(obj), "DS4.1 K2/mul1 must not qualify"
    checks += 1

    with patch.dict(os.environ, {"GLM53_COOPERATIVE_MOE": "1"}):
        assert adapter.install(module)
        try:
            adapter.install(module)
        except RuntimeError:
            pass
        else:
            raise AssertionError("double install accepted")
        checks += 1

    def tensor(shape, dtype, device="cuda:0"):
        return NS(
            shape=shape, dtype=dtype, device=device, is_cuda=device.startswith("cuda")
        )

    obj = layer()
    assert Method().process_weights_after_loading(obj) == "loaded"
    assert hasattr(obj, "_glm53_coop_native")
    ds41 = layer(bits=2, mcg=False, mul1=True, hidden=5120, inter=1152)
    assert Method().process_weights_after_loading(ds41) == "loaded"
    assert not hasattr(ds41, "_glm53_coop_native")
    checks += 2

    for rows in (0, 1, 2, 4, 7, 8, 16, 24, 32, 33, 40, 64, 3072):
        for topk in (6, 8):
            x = tensor((rows, 4096), "bf16")
            ids = tensor((rows, topk), "i64")
            rw = tensor((rows, topk), "fp32")
            result = module.apply_exl3_fused_moe(
                x, ids, rw, obj, obj._exl3_inners, None, 10.0
            )
            expected = (
                "candidate" if 1 <= rows <= 32 and topk == 8 else "stock"
            )
            assert result == expected, (rows, topk, result)
            checks += 1
    assert native_creations == [
        "cuda:0"
    ], "scratch not shared or created for unsupported layer"
    obj = layer()
    Method().process_weights_after_loading(obj)
    obj._exl3_bits = 3
    Method().process_weights_after_loading(obj)
    assert not hasattr(
        obj, "_glm53_coop_native"
    ), "stale candidate after unsupported weight reload"
    checks += 1
    obj = layer()
    Method().process_weights_after_loading(obj)
    for changed in (
        "input_device",
        "input_dtype",
        "ids_device",
        "ids_dtype",
        "weights_device",
        "weights_shape",
        "limit",
        "hidden",
    ):
        x = tensor((8, 4096), "bf16")
        ids = tensor((8, 8), "i64")
        rw = tensor((8, 8), "fp32")
        limit = 10.0
        if changed == "input_device":
            x.device = "cpu"
            x.is_cuda = False
        elif changed == "input_dtype":
            x.dtype = "i64"
        elif changed == "ids_device":
            ids.device = "cuda:1"
        elif changed == "ids_dtype":
            ids.dtype = "i32"
        elif changed == "weights_device":
            rw.device = "cuda:1"
        elif changed == "weights_shape":
            rw.shape = (8, 6)
        elif changed == "limit":
            limit = float("nan")
        elif changed == "hidden":
            x.shape = (8, 5120)
        assert (
            module.apply_exl3_fused_moe(x, ids, rw, obj, obj._exl3_inners, None, limit)
            == "stock"
        ), changed
        checks += 1
    fresh = NS(
        Exl3MoEMethod=type(
            "FreshMethod", (), {"process_weights_after_loading": lambda *args: None}
        ),
        apply_exl3_fused_moe=stock,
        logger=NS(info=lambda *args: None),
    )
    with patch.dict(os.environ, {"GLM53_COOPERATIVE_MOE": "0"}):
        assert adapter.install(fresh, enabled=True)
    checks += 1
    assert adapter.resolve_geometry("") == 1
    assert adapter.resolve_geometry("2") == 2
    try:
        adapter.resolve_geometry("9")
    except adapter.CooperativeMoEError:
        checks += 1
    else:
        raise AssertionError("invalid geometry accepted")
    try:
        adapter.enforce_serialized_execution("--enable-dbo")
    except adapter.CooperativeMoEError:
        checks += 1
    else:
        raise AssertionError("dual-batch overlap accepted")
    with patch.dict(os.environ, {"VLLM_ENABLE_UBATCHING": "1"}):
        try:
            adapter.enforce_serialized_execution("")
        except adapter.CooperativeMoEError:
            checks += 1
        else:
            raise AssertionError("ubatch env accepted")
    capturing["on"] = True
    x = tensor((3, 4096), "bf16")
    ids = tensor((3, 8), "i64")
    rw = tensor((3, 8), "fp32")
    module.apply_exl3_fused_moe(x, ids, rw, obj, obj._exl3_inners, None, 10.0)
    diag = module._glm53_coop_diag
    assert 3 in diag.capture_selection and diag.capture_selection[3]["selected"]
    assert diag.capture_selection[3]["kind"] == "capture"
    capturing["on"] = False
    x = tensor((40, 4096), "bf16")
    ids = tensor((40, 8), "i64")
    rw = tensor((40, 8), "fp32")
    assert (
        module.apply_exl3_fused_moe(x, ids, rw, obj, obj._exl3_inners, None, 10.0)
        == "stock"
    )
    assert diag.eager_selection[40]["selected"] is False
    assert diag.eager_selection[40]["reason"] == "rows_out_of_range"
    checks += 3
    # This stack's launcher is run.sh (+ supervise.sh); both must forward the geometry and the
    # generated overlay so the opt-in survives a supervised restart.
    root = Path(__file__).resolve().parents[2]
    run_text = (root / "run.sh").read_text()
    sup_text = (root / "supervise.sh").read_text()
    assert '-e GLM53_COOP_GEOMETRY=' in run_text
    assert "EXL3_OVERLAY_HOST" in run_text and "EXL3_OVERLAY_HOST" in sup_text
    assert "GLM53_COOP_GEOMETRY" in sup_text
    checks += 1
    opt = subprocess.run(
        [
            sys.executable,
            "-O",
            str(Path(__file__).with_name("test_optimized_init.py")),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert opt.returncode == 0, opt.stdout + opt.stderr
    checks += 1
    print(
        {
            "status": "pass",
            "checks": checks,
            "cpu_stub_only": True,
            "cuda_adapter_verified": False,
        }
    )


if __name__ == "__main__":
    main()
