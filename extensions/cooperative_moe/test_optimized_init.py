"""Regression: required native init must run under python -O (no assert)."""

import ctypes
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class _Device:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main():
    if not sys.flags.optimize:
        raise SystemExit("run with python -O")
    fake_torch = NS(
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        int64="i64",
        int32="i32",
        cuda=NS(
            get_device_capability=lambda device: (12, 1),
            is_current_stream_capturing=lambda: False,
            device=lambda device: _Device(),
        ),
        empty=lambda *a, **k: "tensor",
        zeros=lambda *a, **k: "zeros",
    )
    spec = importlib.util.spec_from_file_location(
        "cooperative_moe_runtime_opt", Path(__file__).with_name("runtime.py")
    )
    adapter = importlib.util.module_from_spec(spec)
    calls = []

    class FakeCDLL:
        def __init__(self, path):
            calls.append(("cdll", path))
            self.glm53_coop_launch = NS(argtypes=None, restype=None)

        def glm53_coop_abi(self):
            calls.append("abi")
            return 1

        def glm53_coop_info(self, bits, geometry, info):
            calls.append(("info", bits, geometry))
            info[4] = info[9] = info[14] = 2
            info[16] = adapter.CTR_LEN
            info[17] = adapter.PARAMS_SIZE
            return 0

    with tempfile.TemporaryDirectory() as tmp:
        so = Path(tmp) / "cooperative_moe.so"
        so.write_bytes(b"native-fixture-for-optimized-init")
        with patch.dict(sys.modules, torch=fake_torch), patch.object(
            ctypes, "CDLL", FakeCDLL
        ):
            spec.loader.exec_module(adapter)
            adapter.SHA256 = hashlib.sha256(so.read_bytes()).hexdigest()
            launch = adapter.CoopLaunch("cuda:0", tmp, geometry=1)
    names = [c[0] if isinstance(c, tuple) else c for c in calls]
    if "abi" not in names or "info" not in names:
        raise AssertionError(names)
    if launch.geometry != 1 or launch.occupancy != (2, 2, 2):
        raise AssertionError((launch.geometry, launch.occupancy))
    print({"status": "pass", "optimized": True, "native_init_calls": names})


if __name__ == "__main__":
    main()
