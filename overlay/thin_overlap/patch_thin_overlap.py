"""[2026-09-25] THIN_OVERLAP: at E3 prefill, launch the thin kernel (experts <= cap, memory-bound) on a side stream while
the grouped path (gather/gateup/down, compute-bound) runs on the current stream. Both accumulate into `out` with fp32 atomicAdd
(had_hf_r_128_d_inner / fm_down), disjoint work buffers (temps vs h13/h2 scratch) -> correct.
Outside CUDA-graph capture only. Hot kill switch: /root/.cache/vllm/glm53_thin_overlap.off (re-read every 64 calls);
GLM53_THIN_OVERLAP=0 = original path. Measured neutral (-0.5 %): opt-in. Usage: python3 patch_thin_overlap.py <exl3.py> [...] (idempotent)."""
import sys

MARK = "# [25/09] THIN_OVERLAP"
HELPERS = MARK + ''' helpers
_THIN_OVERLAP_ENV = os.environ.get("GLM53_THIN_OVERLAP", "1") == "1"
_THIN_OVERLAP_OFF_FILE = "/root/.cache/vllm/glm53_thin_overlap.off"
_thin_overlap_state = {"calls": 0, "on": _THIN_OVERLAP_ENV, "streams": {}, "logged": None}


def _thin_overlap_stream(device: torch.device):
    """Per-device side stream when the thin/grouped overlap is active, else None."""
    st = _thin_overlap_state
    if st["calls"] % 64 == 0:
        st["on"] = _THIN_OVERLAP_ENV and not os.path.exists(_THIN_OVERLAP_OFF_FILE)
    st["calls"] += 1
    if st["logged"] != st["on"]:
        st["logged"] = st["on"]
        logger.info("[glm53-thin-overlap] %s", "ON" if st["on"] else "OFF")
    if not st["on"] or torch.cuda.is_current_stream_capturing():
        return None
    s = st["streams"].get(device)
    if s is None:
        s = st["streams"][device] = torch.cuda.Stream(device=device)
    return s


def _prefill_cap_enabled() -> bool:'''

OLD_CALL = '''        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap_p, limit
        )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")'''
NEW_CALL = '''        side = _thin_overlap_stream(xh.device)  ''' + MARK + '''
        if side is not None:
            main = torch.cuda.current_stream(xh.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                _exl3_moe_launch(
                    fn, xh, out, expert_count, token_sorted, weight_sorted,
                    temps, ptrs, k, limit, n_active_host,
                )
            apply_exl3_grouped_fat(
                xh, out, counts, token_sorted, weight_sorted, layer, cap_p, limit
            )
            main.wait_stream(side)
        else:
            _exl3_moe_launch(
                fn, xh, out, expert_count, token_sorted, weight_sorted,
                temps, ptrs, k, limit, n_active_host,
            )
            apply_exl3_grouped_fat(
                xh, out, counts, token_sorted, weight_sorted, layer, cap_p, limit
            )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")'''

for path in sys.argv[1:]:
    s = open(path).read()
    if MARK in s:
        print(f"{path}: already patched"); continue
    assert s.count(OLD_CALL) == 1, f"{path}: ancre appel introuvable"
    assert s.count("\ndef _prefill_cap_enabled() -> bool:") == 1, f"{path}: ancre helpers introuvable"
    assert "\nlogger = " in s, f"{path}: no module logger"
    s = s.replace(OLD_CALL, NEW_CALL).replace("\ndef _prefill_cap_enabled() -> bool:", "\n" + HELPERS, 1)
    open(path, "w").write(s); print(f"{path}: patched")
