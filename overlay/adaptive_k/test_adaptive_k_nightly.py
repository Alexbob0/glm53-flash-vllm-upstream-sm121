#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, tests/test_adaptive_k_patch.py,
#         commits 3b31297 + a710ac7 (2026-09-08). Adapted for the nightly port:
#         inputs are our overlay files (not site-packages), the generator takes
#         CLI paths, and the capture-size list is cross-checked against run.sh.
"""CPU-only tests for overlay/adaptive_k/patch_adaptive_k_nightly.py.

No GPU, no container, no network. Run:  python3 overlay/adaptive_k/test_adaptive_k_nightly.py

Coverage:
  1. generator produces ast.parse-clean scheduler.py + cudagraph_utils.py
  2. idempotence (function level and CLI level, byte-identical second run)
  3. fail-closed on a drifted anchor (scheduler and cudagraph), and on a
     scheduler input that is not our APC overlay
  4. EMA -> chosen-length policy, including batch_k() which is the hook our
     pile actually runs (AsyncScheduler: dflash is in EagleModelTypes, so
     vLLM auto-enables async scheduling and update_draft_token_ids is skipped)
  5. the cudagraph query-length helper
  6. the committed overlay/adaptive_k/*.py are up to date with the generator
  7. run.sh's --cudagraph-capture-sizes literal matches the arithmetic below
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PATCH = HERE / "patch_adaptive_k_nightly.py"
SCHED_IN = ROOT / "overlay" / "apc" / "scheduler.py"
CG_IN = HERE / "cudagraph_utils.orig.py"
RUN_SH = ROOT / "run.sh"
MARK = "# [glm53-adaptive-k]"

sys.path.insert(0, str(HERE))
import patch_adaptive_k_nightly as P  # noqa: E402


# --------------------------------------------------------------------------- #
# capture sizes
# --------------------------------------------------------------------------- #
def capture_sizes(max_num_seqs: int, k_set, base) -> list[int]:
    """Sizes that must be captured so every adaptive uniform-decode batch has a graph.

    A uniform decode batch of r requests at draft length k is r * (k + 1) tokens,
    and the dispatcher only accepts a descriptor whose token count is an exact
    multiple of the query length (cudagraph_utils round_up + num_reqs check).
    So we need {r * (k+1) : r in 1..max_num_seqs, k in k_set}, unioned with the
    base list vLLM would have computed on its own (keeping its maximum, so
    max_cudagraph_capture_size and the PIECEWISE/mixed coverage are unchanged).
    """
    need = {r * (k + 1) for k in k_set for r in range(1, max_num_seqs + 1)}
    return sorted(set(base) | need)


# vLLM default for SEQS=6, K=7 (decode_query_len 8), sm121 -> default_max_graph_size 512:
#   max_cudagraph_capture_size = min(6 * 8 * 2, 512) = 96
#   sizes = [1, 2, 4] + range(8, 97, 8)
# Verified against a real boot log (logs/hang-capture-head.log).
VLLM_DEFAULT_SIZES_SEQS6_K7 = [1, 2, 4] + list(range(8, 97, 8))
EXPECTED_SIZES = capture_sizes(6, (2, 4, 7), VLLM_DEFAULT_SIZES_SEQS6_K7)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class _Req:
    def __init__(self, rid, k=7):
        self.request_id = rid
        self.spec_token_ids = [-1] * k


class _SReq:
    def __init__(self, rid, structured=False, prefill=False):
        self.request_id = rid
        self.use_structured_output = structured
        self.is_prefill_chunk = prefill


def policy_tests(helper_src: str) -> None:
    def make(env):
        ns = {"os": os}
        old = dict(os.environ)
        # Never inherit knobs from the caller's environment.
        for key in [k for k in os.environ if k.startswith("GLM53_ADAPTIVE_K")]:
            os.environ.pop(key)
        os.environ.update(env)
        try:
            exec(helper_src, ns)
            inst = ns["_Glm53AdaptiveK"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        return inst

    # default off: never trims
    p = make({"GLM53_ADAPTIVE_K": "off"})
    r = _Req("a")
    for _ in range(10):
        p.observe("a", 7, 0)
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7 and not p.enabled

    # on: prose-like (1 accepted) -> trims to 2 after min_steps; structured stays 7
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    r = _Req("a")
    for _ in range(3):
        p.observe("a", 7, 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 7, "min_steps guard"
    for _ in range(12):
        p.observe("a", len(r.spec_token_ids), 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, r.spec_token_ids
    s = _Req("s")
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False), (s, True)], {"a": r, "s": s})
    assert len(r.spec_token_ids) == 7 and len(s.spec_token_ids) == 7, "structured pins the batch"

    # ratchet escape: saturation feeds the full length back, EMA can climb to 7
    r.spec_token_ids = [-1] * 2
    for _ in range(20):
        p.observe("a", len(r.spec_token_ids), len(r.spec_token_ids))
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 7, r.spec_token_ids

    # saturate=n reproduces the ratchet (documented behaviour)
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_SATURATE": "n", "GLM53_ADAPTIVE_K_HIST": "0"})
    r = _Req("a")
    for _ in range(15):
        p.observe("a", 7, 1)
    r.spec_token_ids = [-1] * 7
    p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2
    for _ in range(30):
        p.observe("a", 2, 2)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
    assert len(r.spec_token_ids) == 2, "saturate=n never climbs"

    # margin is honoured: 2 accepted/step -> ema ~2.0
    #   margin 0.0 -> ceil(2.0) = 2   -> largest set value <= 2 is 2
    #   margin 1.0 -> ceil(3.0) = 3+  -> largest set value <= 3/4 is 4
    for margin, want in (("0.0", 2), ("1.0", 4)):
        p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_MARGIN": margin})
        assert p.margin == float(margin)
        for _ in range(40):
            p.observe("a", 7, 2)
        assert p.choose("a", 7, False) == want, (margin, p.state)

    # alpha is honoured: the EMA is seeded at the full length, so a small alpha
    # still reads "nearly 7" after min_steps while a large one has converged to ~1
    for alpha, want in (("0.9", 2), ("0.05", 7)):
        p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_ALPHA": alpha})
        assert p.alpha == float(alpha)
        for _ in range(4):
            p.observe("a", 7, 1)
        assert p.choose("a", 7, False) == want, (alpha, p.state)

    # min_steps is honoured
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_MIN_STEPS": "9"})
    for i in range(1, 12):
        p.observe("a", 7, 1)
        got = p.choose("a", 7, False)
        assert (got is None) == (i < 9), (i, got)

    # runtime override file: mode off -> no trimming; set clamped to the boot set
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "ak.json"
        p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_FILE": str(f)})
        r = _Req("a")
        for _ in range(15):
            p.observe("a", 7, 1)
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 2
        f.write_text(json.dumps({"mode": "off"}))
        os.utime(f, (1, 1))
        p.steps = 0  # force the mtime check
        for _ in range(15):
            p.observe("a", 7, 1)
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 7 and not p.enabled, "file mode=off must disable trimming"
        f.write_text(json.dumps({"mode": "ema", "set": [3, 5, 9]}))
        os.utime(f, (2, 2))
        p.steps = 0
        r.spec_token_ids = [-1] * 7
        p.apply([(r, False)], {"a": r})
        assert p.enabled and p.k_set == [2, 4, 7], (p.enabled, p.k_set)  # no overlap -> boot set
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "ak.json"
        p = make({
            "GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_SET": "2,3,4,5,7",
            "GLM53_ADAPTIVE_K_HIST": "0", "GLM53_ADAPTIVE_K_FILE": str(f),
        })
        f.write_text(json.dumps({"mode": "ema", "set": "3,5,7", "margin": 1.5}))
        p.steps = 0
        p._reload()
        assert p.k_set == [3, 5, 7] and p.margin == 1.5, (p.k_set, p.margin)
        r = _Req("a")
        for _ in range(15):
            p.observe("a", 7, 1)
        p.apply([(r, False)], {"a": r})
        assert len(r.spec_token_ids) == 3, r.spec_token_ids

    # a boot-time-disabled policy never reloads (no graphs for the extra lengths)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "ak.json"
        f.write_text(json.dumps({"mode": "ema"}))
        p = make({"GLM53_ADAPTIVE_K": "off", "GLM53_ADAPTIVE_K_FILE": str(f)})
        p.steps = 0
        p._reload()
        assert not p.enabled and not p.boot_enabled, "override must not switch a cold boot on"

    # schedule-time hook: THE path our pile runs (AsyncScheduler placeholders)
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a, b, s, pf = _SReq("a"), _SReq("b"), _SReq("s", structured=True), _SReq("p", prefill=True)
    live = {"a": a, "b": b, "s": s, "p": pf}
    assert p.batch_k(7, [a], live) == 7, "unobserved request pins full length"
    for _ in range(15):
        p.observe("a", 7, 1)
        p.observe("b", 7, 7)
    assert p.batch_k(7, [a], live) == 2
    assert p.batch_k(7, [b], live) == 7
    assert p.batch_k(7, [a, b], live) == 2, "batch minimum"
    assert p.batch_k(7, [a, s], live) == 7, "structured pins full length"
    assert p.batch_k(7, [a, pf], live) == 2, "prefill chunks are ignored"
    assert p.batch_k(7, [a, None], live) == 2
    assert p.batch_k(0, [a], live) == 0, "no spec -> untouched"
    assert p.hist.get(2, 0) >= 3

    # every value batch_k can return must have a captured graph
    for n in {p.batch_k(7, [a], live), p.batch_k(7, [b], live), 7}:
        for r_count in range(1, 7):
            assert r_count * (n + 1) in EXPECTED_SIZES, (n, r_count)

    # batch minimum across two requests (sync/apply path)
    p = make({"GLM53_ADAPTIVE_K": "ema", "GLM53_ADAPTIVE_K_HIST": "0"})
    a2, b2 = _Req("a"), _Req("b")
    for _ in range(15):
        p.observe("a", 7, 7)
        p.observe("b", 7, 1)
    p.apply([(a2, False), (b2, False)], {"a": a2, "b": b2})
    assert len(a2.spec_token_ids) == 2 and len(b2.spec_token_ids) == 2


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def generator_tests(tmp: Path) -> tuple[str, str]:
    sched_out, cg_out = tmp / "scheduler.py", tmp / "cudagraph_utils.py"
    argv = [
        "--scheduler-in", str(SCHED_IN), "--scheduler-out", str(sched_out),
        "--cg-in", str(CG_IN), "--cg-out", str(cg_out),
    ]
    assert P.main(argv) == 0
    st, ct = sched_out.read_text(), cg_out.read_text()

    assert st.count(MARK) >= 6, st.count(MARK)
    assert "_GLM53_ADAPTIVE_K.observe(" in st
    assert "_GLM53_ADAPTIVE_K.apply(" in st
    assert "_GLM53_ADAPTIVE_K.batch_k(" in st
    assert "[apc-align]" in st, "the APC mamba-alignment fix must survive"
    assert "_glm53_adaptive_k_query_lens(decode_query_lens" in ct
    ast.parse(st, "scheduler.py")
    ast.parse(ct, "cudagraph_utils.py")

    # idempotence: re-running leaves both files byte-identical
    assert P.main(argv) == 0
    assert sched_out.read_text() == st and cg_out.read_text() == ct, "not idempotent"
    # and at function level
    assert P.patch_scheduler_text(st) == st
    assert P.patch_cudagraph_text(ct) == ct

    # fail closed on drifted anchors
    pristine_sched = SCHED_IN.read_text()
    pristine_cg = CG_IN.read_text()
    for drifted, fn, label in (
        (pristine_sched.replace(P.OBS_OLD, "                pass\n"), P.patch_scheduler_text, "observe"),
        (pristine_sched.replace(P.UPD_OLD, "    def update_draft_token_ids(self, d): pass\n"),
         P.patch_scheduler_text, "update_draft_token_ids"),
        (pristine_sched.replace(P.SCHED_K_OLD, "        num_spec_tokens_to_schedule = 7\n"),
         P.patch_scheduler_text, "num_spec_tokens_to_schedule"),
        (pristine_sched.replace(P.SCHED_HELPER_ANCHOR, ""), P.patch_scheduler_text, "helper anchor"),
        (pristine_sched.replace("[apc-align]", "[nope]"), P.patch_scheduler_text, "not the APC overlay"),
        (pristine_cg.replace(P.CG_ANCHOR, ""), P.patch_cudagraph_text, "cg helper anchor"),
        (pristine_cg.replace(P.CG_OLD, "        else:\n            pass\n"), P.patch_cudagraph_text, "decode_query_lens"),
    ):
        try:
            fn(drifted)
        except SystemExit as exc:
            assert "expected one" in str(exc) or "not unique" in str(exc) or "APC" in str(exc), str(exc)
        else:
            raise AssertionError(f"drifted anchor not rejected: {label}")

    # the committed outputs must be what the generator produces today
    for committed, produced, name in (
        (HERE / "scheduler.py", st, "scheduler.py"),
        (HERE / "cudagraph_utils.py", ct, "cudagraph_utils.py"),
    ):
        assert committed.is_file(), f"missing {committed}"
        ast.parse(committed.read_text(), name)
        assert committed.read_text() == produced, f"{committed} is stale — re-run the generator"
    return st, ct


def cg_helper_tests(ct: str) -> None:
    start = ct.index("def _glm53_adaptive_k_query_lens(")
    end = ct.index(P.CG_ANCHOR)
    ns: dict = {}
    exec(ct[start:end], ns)
    fn = ns["_glm53_adaptive_k_query_lens"]
    old = dict(os.environ)
    try:
        os.environ["GLM53_ADAPTIVE_K"] = "off"
        assert fn([8], 8) == [8], "off must be a no-op"
        os.environ["GLM53_ADAPTIVE_K"] = "ema"
        os.environ["GLM53_ADAPTIVE_K_SET"] = "2,4,7"
        assert fn([8], 8) == [3, 5, 8], fn([8], 8)
        # lengths wider than the boot-time decode query length are dropped
        os.environ["GLM53_ADAPTIVE_K_SET"] = "2,4,7,11"
        assert fn([8], 8) == [3, 5, 8], fn([8], 8)
    finally:
        os.environ.clear()
        os.environ.update(old)


def _bash_ak_sizes(seqs: int, k: int, kset: str) -> list[int]:
    """Run run.sh's own _ak_sizes() so the launcher and this file cannot drift."""
    out = subprocess.run(
        ["bash", "-c",
         f'source <(sed -n "/^_ak_sizes()/,/^}}/p" {RUN_SH}); _ak_sizes {seqs} {k} {kset}'],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(x) for x in out.split()]


def run_sh_tests() -> None:
    text = RUN_SH.read_text()
    assert "ADAPTIVE_K" in text, "run.sh has no ADAPTIVE_K opt-in"
    assert "overlay/adaptive_k/scheduler.py" in text
    assert "overlay/adaptive_k/cudagraph_utils.py" in text
    assert "--cudagraph-capture-sizes" in text
    # run.sh is opt-in (0 = byte-for-byte the baked scheduler); supervise.sh forwards it and
    # defaults it on, matching the production configuration adopted on 2026-09-12.
    assert 'ADAPTIVE_K="${ADAPTIVE_K:-0}"' in text, "run.sh ADAPTIVE_K is not opt-in"
    sup = (ROOT / "supervise.sh").read_text()
    assert "ADAPTIVE_K=${ADAPTIVE_K:-" in sup, "supervise.sh does not forward ADAPTIVE_K"

    got = _bash_ak_sizes(6, 7, "2,4,7")
    assert got == EXPECTED_SIZES, f"run.sh _ak_sizes {got} != computed {EXPECTED_SIZES}"
    # purely additive: with a single k = K, it must reproduce vLLM's own default list
    assert _bash_ak_sizes(6, 7, "7") == VLLM_DEFAULT_SIZES_SEQS6_K7
    # and every adaptive batch shape is covered
    for k in (2, 4, 7):
        for r in range(1, 7):
            assert r * (k + 1) in got, (k, r)

    for cmd, msg in (
        ("bash -n", "run.sh"),
        ("bash -n", "supervise.sh"),
    ):
        target = RUN_SH if msg == "run.sh" else ROOT / "supervise.sh"
        subprocess.run(cmd.split() + [str(target)], check=True)


def main() -> int:
    for src in (PATCH, SCHED_IN, CG_IN, RUN_SH):
        if not src.is_file():
            raise SystemExit(f"missing {src}")
    print(f"capture sizes (SEQS=6, k in 2,4,7): {' '.join(map(str, EXPECTED_SIZES))}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        st, ct = generator_tests(tmp)
        print(f"generator OK ({st.count(MARK)} + {ct.count(MARK)} markers, idempotent, fails closed)")
        start = st.index("class _Glm53AdaptiveK:")
        end = st.index("_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()")
        policy_tests(st[start:end])
        print("policy OK")
        cg_helper_tests(ct)
        print("cudagraph query-length helper OK")
        run_sh_tests()
        print("run.sh capture-size literal OK")
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
    print("adaptive-k nightly port OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
