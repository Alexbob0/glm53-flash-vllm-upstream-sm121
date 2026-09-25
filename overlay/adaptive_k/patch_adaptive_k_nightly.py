#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, overlay/patch_adaptive_k.py,
#         commits 3b31297 + a710ac7 (2026-09-08). Adapted here for the vLLM
#         0.28.1rc1 nightly image (glm53-upstream:b3) and our APC scheduler overlay.
"""Adaptive verification length for DFlash2 (env-gated, default OFF) — nightly port.

Generates two overlay files from inputs (never edits site-packages in place; run.sh
bind-mounts the outputs when ADAPTIVE_K=1):

  --scheduler-in   overlay/apc/scheduler.py   (our mamba-align APC scheduler; the
                   adaptive-k hooks are layered ON TOP of it)
  --scheduler-out  overlay/adaptive_k/scheduler.py
  --cg-in          overlay/adaptive_k/cudagraph_utils.orig.py (pristine, extracted
                   from the image)
  --cg-out         overlay/adaptive_k/cudagraph_utils.py

Policy (unchanged from upstream): the drafter keeps its 8-token block (1 anchor +
7 drafts); the scheduler sizes the draft slots of the next step from a CPU-side EMA
of accepted draft tokens per request, uniform over the batch (minimum), so every
decode step still hits a FULL CUDA graph captured for each candidate length + 1.

Knobs (read at runtime inside the container):
  GLM53_ADAPTIVE_K            off (default) | ema
  GLM53_ADAPTIVE_K_SET        candidate draft lengths, default "2,4,7"
  GLM53_ADAPTIVE_K_ALPHA      EMA alpha, default 0.25
  GLM53_ADAPTIVE_K_MARGIN     n = largest set value <= ceil(ema + margin), default 1.0
  GLM53_ADAPTIVE_K_MIN_STEPS  steps observed at full length before trimming, default 4
  GLM53_ADAPTIVE_K_SATURATE   "max" (default) | "n"
  GLM53_ADAPTIVE_K_HIST       histogram log period in steps, default 200 (0 = off)
  GLM53_ADAPTIVE_K_CONC       0 (default) | model  — batch-size-aware bytes model (see class doc)
  GLM53_ADAPTIVE_K_EST        ema (default) | pos  — per-position acceptance estimator (21/09)
  GLM53_ADAPTIVE_K_POS_ALPHA  per-position EMA alpha, default 0.2 (running mean until 1/seen < alpha)
  GLM53_ADAPTIVE_K_POS_BETA   relaxation of never-shown positions toward p_{i-1}, default 0.02
  GLM53_ADAPTIVE_K_POS_PRIOR  initial p_i, default 0.7
  GLM53_ADAPTIVE_K_PROBE      full-length probe period in batch steps (pos only), default 32 (0 = off)
  GLM53_ADAPTIVE_K_FILE       runtime override JSON, default /root/.cache/vllm/glm53_adaptive_k.json
                              (= $HOME/glm53-upstream-cache/<IMG tag>/vllm/glm53_adaptive_k.json
                              on the host with JIT_CACHE=1 in run.sh)

Runtime override file semantics (21/09): a key ABSENT from the JSON keeps the value of the
PREVIOUS reload (partial override), not the boot default. For an A/B, always write every key
you care about (mode, est, conc, set, probe, ...) — a bench arm inherited conc=model this way.

Idempotent (marker comment), fails closed when an anchor is not found exactly once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "# [glm53-adaptive-k]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

SCHED_HELPER = '''
class _Glm53AdaptiveK:  # [glm53-adaptive-k]
    """CPU-only policy for the verified draft prefix length.

    est=ema  (default, byte-identical to 12/09): EMA of the accepted count per request.
    est=pos  (21/09): per-position conditional acceptance p_i (EMA, censoring-aware);
             E[accepted | k] = sum_{j<k} prod_{i<=j} p_i drives the choice. Positions a
             trimmed draft never shows are relaxed toward p_{i-1} (rate pos_beta) and the
             batch is probed at full length every probe_every steps, so k can climb again
             without the ema saturation hack (the source of the c6 prose bimodality)."""

    def __init__(self) -> None:
        mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
        self.enabled = mode in ("ema", "on", "1")
        self.alpha = float(os.environ.get("GLM53_ADAPTIVE_K_ALPHA", "0.25"))
        self.margin = float(os.environ.get("GLM53_ADAPTIVE_K_MARGIN", "1.0"))
        self.min_steps = int(os.environ.get("GLM53_ADAPTIVE_K_MIN_STEPS", "4"))
        raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
        self.k_set = sorted({int(x) for x in raw.split(",") if x.strip()})
        self.saturate = os.environ.get("GLM53_ADAPTIVE_K_SATURATE", "max").strip().lower()
        self.hist_every = int(os.environ.get("GLM53_ADAPTIVE_K_HIST", "200"))
        # [conc] "0" = today (target = ceil(estimate + margin), batch-size blind); "model" = pick
        # the k in k_set maximizing sum_r (E_r(k) + 1) / bytes_per_step(conc, k) with the bandwidth
        # model of MiaAI's bench_ceiling: non-expert bytes + layers x distinct experts x expert
        # bytes, distinct = E(1 - (1 - topk/E)^rows), rows = conc x (k + 1). Same verifier, exact
        # sampling: the choice of k is quality-neutral by construction.
        # nonexpert_mib: 9216 = MiaAI BF16 denses. Our denses are EXL3 K6: the 21/09 fixed-k sweep
        # (c6 structured k7/k4/k2 = 240/195/146 ms per step, c1 k7 = 78 ms) fits 18 ms + 1.04 ms per
        # distinct expert, i.e. ~4600 MiB of non-expert traffic per step -> default 4600 here.
        self.conc_mode = os.environ.get("GLM53_ADAPTIVE_K_CONC", "0").strip().lower()
        self.cost = {"nonexpert_mib": 4600.0, "expert_mib": 6.2, "layers": 42.0, "experts": 288.0, "topk": 8.0}
        # [pos] estimator knobs
        self.est = os.environ.get("GLM53_ADAPTIVE_K_EST", "ema").strip().lower()
        self.pos_alpha = float(os.environ.get("GLM53_ADAPTIVE_K_POS_ALPHA", "0.2"))
        self.pos_beta = float(os.environ.get("GLM53_ADAPTIVE_K_POS_BETA", "0.02"))
        self.pos_prior = float(os.environ.get("GLM53_ADAPTIVE_K_POS_PRIOR", "0.7"))
        self.probe_every = int(os.environ.get("GLM53_ADAPTIVE_K_PROBE", "32"))
        self.state: dict[str, list] = {}  # req_id -> [ema, observed_steps, p[], seen[]]
        self.hist: dict[int, int] = {}
        self.steps = 0
        self.k_max = max(self.k_set) if self.k_set else 0
        # Runtime override (no reboot): JSON {"mode","alpha","margin","set","saturate","min_steps",
        # "conc","cost","est","pos_alpha","pos_beta","pos_prior","probe"} at GLM53_ADAPTIVE_K_FILE
        # (default: the mounted vLLM cache dir). "set" is clamped to the boot-time set because
        # graphs are captured for the boot-time lengths only.
        self.boot_set = list(self.k_set)
        self.boot_enabled = self.enabled
        self.file = os.environ.get("GLM53_ADAPTIVE_K_FILE", "/root/.cache/vllm/glm53_adaptive_k.json")
        self.file_mtime = None
        self._reload()
        if self.enabled:
            print(
                f"[glm53-adaptive-k] enabled set={self.k_set} alpha={self.alpha} "
                f"margin={self.margin} min_steps={self.min_steps} saturate={self.saturate} "
                f"conc={self.conc_mode} est={self.est} pos_alpha={self.pos_alpha} "
                f"pos_beta={self.pos_beta} probe={self.probe_every}",
                flush=True,
            )

    def _reload(self) -> None:
        if not self.boot_enabled:
            return  # graphs for the extra lengths exist only when enabled at boot
        try:
            mtime = os.stat(self.file).st_mtime
        except OSError:
            mtime = None
        if mtime == self.file_mtime:
            return
        self.file_mtime = mtime
        if mtime is None:
            return
        try:
            import json
            with open(self.file) as fh:
                cfg = json.load(fh)
            mode = str(cfg.get("mode", "ema")).strip().lower()
            self.enabled = mode in ("ema", "on", "1")
            self.alpha = float(cfg.get("alpha", self.alpha))
            self.margin = float(cfg.get("margin", self.margin))
            self.min_steps = int(cfg.get("min_steps", self.min_steps))
            self.saturate = str(cfg.get("saturate", self.saturate)).strip().lower()
            self.conc_mode = str(cfg.get("conc", self.conc_mode)).strip().lower()
            if isinstance(cfg.get("cost"), dict):
                self.cost.update({str(a): float(b) for a, b in cfg["cost"].items()})
            self.est = str(cfg.get("est", self.est)).strip().lower()
            self.pos_alpha = float(cfg.get("pos_alpha", self.pos_alpha))
            self.pos_beta = float(cfg.get("pos_beta", self.pos_beta))
            self.pos_prior = float(cfg.get("pos_prior", self.pos_prior))
            self.probe_every = int(cfg.get("probe", self.probe_every))
            if "set" in cfg:
                want = {int(x) for x in (cfg["set"] if isinstance(cfg["set"], list) else str(cfg["set"]).split(","))}
                self.k_set = sorted(want & set(self.boot_set)) or list(self.boot_set)
            self.state.clear()
            self.hist.clear()
            print(
                f"[glm53-adaptive-k] reloaded {self.file}: enabled={self.enabled} set={self.k_set} "
                f"alpha={self.alpha} margin={self.margin} min_steps={self.min_steps} saturate={self.saturate} "
                f"conc={self.conc_mode} cost={self.cost} est={self.est} pos_alpha={self.pos_alpha} "
                f"pos_beta={self.pos_beta} pos_prior={self.pos_prior} probe={self.probe_every}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[glm53-adaptive-k] override file ignored: {exc!r}", flush=True)

    def observe(self, req_id: str, num_draft: int, num_accepted: int) -> None:
        if not self.enabled or num_draft <= 0:
            return
        if num_accepted >= num_draft and self.saturate != "n":
            obs = float(max(self.k_max, num_draft))
        else:
            obs = float(num_accepted)
        st = self.state.get(req_id)
        if st is None:
            st = [obs * self.alpha + float(self.k_max) * (1.0 - self.alpha), 1.0,
                  [self.pos_prior] * self.k_max, [0] * self.k_max]
            self.state[req_id] = st
        else:
            st[0] = obs * self.alpha + st[0] * (1.0 - self.alpha)
            st[1] += 1.0
        # [pos] prefix acceptance: positions < num_accepted accepted, position num_accepted
        # rejected when the draft was longer, everything after is censored. A position the
        # trimmed draft did not even show is relaxed toward its predecessor (monotone prior)
        # so a request whose early positions improve gets its k back.
        p, seen = st[2], st[3]
        n_obs = min(num_accepted, self.k_max)
        for i in range(n_obs):
            seen[i] += 1
            a = max(self.pos_alpha, 1.0 / seen[i])
            p[i] += a * (1.0 - p[i])
        if num_accepted < num_draft and num_accepted < self.k_max:
            i = num_accepted
            seen[i] += 1
            a = max(self.pos_alpha, 1.0 / seen[i])
            p[i] += a * (0.0 - p[i])
        if self.pos_beta > 0.0:
            for i in range(max(num_draft, 1), self.k_max):
                if p[i] < p[i - 1]:
                    p[i] += self.pos_beta * (p[i - 1] - p[i])

    @staticmethod
    def expected(p, k: int) -> float:
        """E[accepted draft tokens] for a draft of length k under per-position p."""
        e, run = 0.0, 1.0
        for i in range(min(k, len(p))):
            run *= p[i]
            e += run
        return e

    def _bytes(self, conc: int, v: int) -> float:
        c = self.cost
        rows = max(1, conc) * (v + 1)
        distinct = c["experts"] * (1.0 - (1.0 - c["topk"] / c["experts"]) ** rows)
        return c["nonexpert_mib"] + c["layers"] * c["expert_mib"] * distinct

    def _gain(self, st, v: int) -> float:
        """Expected tokens per step for one request at draft length v (+1 = bonus token)."""
        if self.est == "pos":
            return self.expected(st[2], v) + 1.0
        return min(st[0], float(v)) + 1.0

    def choose(self, req_id: str, k: int, structured: bool, conc: int = 1):
        """Draft length for one request, or None when it must stay at full length."""
        if not self.enabled or structured or k <= 0:
            return None
        st = self.state.get(req_id)
        if st is None or st[1] < self.min_steps:
            return None
        import math
        if self.conc_mode == "model":
            best, best_score = None, -1.0
            for v in self.k_set:
                if v > k:
                    continue
                score = self._gain(st, v) / self._bytes(conc, v)
                if score > best_score:
                    best, best_score = v, score
            if best is None:
                return None
            return max(1, min(best, k))
        est = self.expected(st[2], k) if self.est == "pos" else st[0]
        target = int(math.ceil(est + self.margin))
        cands = [v for v in self.k_set if v <= min(target, k)]
        n = max(cands) if cands else min(self.k_set)
        return max(1, min(n, k))

    def _decide(self, items, k: int) -> int:
        """Uniform draft length for a decode batch. items: list of (req_id, structured).
        Any structured-output or not-yet-observed request pins the batch at k."""
        if self.probe_every > 0 and self.est == "pos" and self.steps % self.probe_every == 0:
            return k  # [pos] full-length probe: refresh the censored positions
        conc = len(items)
        if self.est == "pos" and self.conc_mode == "model":
            sts = []
            for rid, s in items:
                if s:
                    return k
                st = self.state.get(rid)
                if st is None or st[1] < self.min_steps:
                    return k
                sts.append(st)
            best, best_score = k, -1.0
            for v in self.k_set:
                if v > k:
                    continue
                score = sum(self._gain(st, v) for st in sts) / self._bytes(conc, v)
                if score > best_score:
                    best, best_score = v, score
            return max(1, min(best, k))
        ns = []
        for rid, s in items:
            n_i = self.choose(rid, k, s, conc)
            if n_i is None:
                return k
            ns.append(n_i)
        return min(ns) if ns else k

    def apply(self, reqs, live_ids) -> None:
        """reqs: list of (request, structured). Trims spec_token_ids to a uniform n.

        A structured-output or not-yet-observed request pins the whole batch at
        the full length (uniform batch, nothing trimmed)."""
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or not reqs:
            self.steps += 1
            return
        k = max(len(r.spec_token_ids) for r, _ in reqs)
        n = self._decide([(r.request_id, s) for r, s in reqs], k)
        for r, _ in reqs:
            if len(r.spec_token_ids) > n:
                r.spec_token_ids = r.spec_token_ids[:n]
        self._count(n, live_ids)

    def batch_k(self, k: int, reqs, live_ids) -> int:
        """Schedule-time hook (async scheduler): the number of draft slots every
        request gets on the next step. Minimum over the scheduled decode
        requests; any structured-output or not-yet-observed request pins the
        batch at k."""
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or k <= 0:
            self.steps += 1
            return k
        decode_reqs = [r for r in reqs if r is not None and not getattr(r, "is_prefill_chunk", False)]
        items = [(r.request_id, bool(getattr(r, "use_structured_output", False))) for r in decode_reqs]
        n = self._decide(items, k) if items else k
        self._count(n, live_ids)
        return n

    def _count(self, n: int, live_ids) -> None:
        self.hist[n] = self.hist.get(n, 0) + 1
        self.steps += 1
        if self.hist_every > 0 and self.steps % self.hist_every == 0:
            total = sum(self.hist.values())
            parts = " ".join(f"{k}:{v}" for k, v in sorted(self.hist.items()))
            emas = " ".join(f"{rid[:8]}={st[0]:.2f}/{int(st[1])}" for rid, st in list(self.state.items())[:4])
            if self.est == "pos":
                emas += " | p " + " ".join(
                    f"{rid[:8]}=[" + ",".join(f"{x:.2f}" for x in st[2]) + f"]E7={self.expected(st[2], self.k_max):.2f}"
                    for rid, st in list(self.state.items())[:2])
            print(f"[glm53-adaptive-k] step {self.steps} chosen-length hist ({total}): {parts} | ema {emas}", flush=True)
            self.state = {rid: st for rid, st in self.state.items() if rid in live_ids}


_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()  # [glm53-adaptive-k]


'''

OBS_OLD = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
"""
OBS_NEW = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                _GLM53_ADAPTIVE_K.observe(req_id, num_draft_tokens, num_accepted)  # [glm53-adaptive-k]
"""

UPD_OLD = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
"""
UPD_NEW = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        _ak_reqs = []  # [glm53-adaptive-k]
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            _ak_structured = self.structured_output_manager.should_advance(request)  # [glm53-adaptive-k]
            if _ak_structured:
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
            if _GLM53_ADAPTIVE_K.boot_enabled and spec_token_ids:  # [glm53-adaptive-k]
                _ak_reqs.append((request, _ak_structured))
        if _ak_reqs:  # [glm53-adaptive-k]
            _GLM53_ADAPTIVE_K.apply(_ak_reqs, self.requests)
"""

SCHED_K_OLD = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
"""
SCHED_K_NEW = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
        if _GLM53_ADAPTIVE_K.boot_enabled and self.dynamic_sd_lookup is None and num_scheduled_tokens:  # [glm53-adaptive-k]
            num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(
                num_spec_tokens_to_schedule,
                [self.requests.get(_rid) for _rid in num_scheduled_tokens],
                self.requests,
            )
"""

CG_HELPER = '''
def _glm53_adaptive_k_query_lens(lens, decode_query_len):  # [glm53-adaptive-k]
    """Extra uniform decode graph lengths for the adaptive verification prefix."""
    import os

    mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
    if mode not in ("ema", "on", "1"):
        return lens
    raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
    ks = {int(x) for x in raw.split(",") if x.strip()}
    extra = {k + 1 for k in ks if 0 < k + 1 <= decode_query_len}
    out = sorted(set(lens) | extra | {decode_query_len})
    print(f"[glm53-adaptive-k] uniform decode graph query lens: {out}", flush=True)
    return out


'''
CG_ANCHOR = "@dataclass(frozen=True)\nclass BatchExecutionDescriptor:\n"
CG_OLD = """        else:
            decode_query_lens = [self.decode_query_len]
"""
CG_NEW = """        else:
            decode_query_lens = [self.decode_query_len]
        decode_query_lens = _glm53_adaptive_k_query_lens(decode_query_lens, self.decode_query_len)  # [glm53-adaptive-k]
"""

SCHED_HELPER_ANCHOR = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
APC_MARK = "[apc-align]"


def replace_once(name: str, text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{name}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def patch_scheduler_text(text: str, name: str = "scheduler.py") -> str:
    if MARK in text:
        return text
    if APC_MARK not in text:
        raise SystemExit(f"{name}: not our APC scheduler overlay ({APC_MARK} missing) — refuse to layer on it")
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(name, text, IMPORT_OLD, IMPORT_NEW, "import os")
    if text.count(SCHED_HELPER_ANCHOR) != 1:
        raise SystemExit(f"{name}: helper insert point not unique")
    text = text.replace(SCHED_HELPER_ANCHOR, SCHED_HELPER + SCHED_HELPER_ANCHOR, 1)
    text = replace_once(name, text, OBS_OLD, OBS_NEW, "observe")
    text = replace_once(name, text, UPD_OLD, UPD_NEW, "update_draft_token_ids")
    text = replace_once(name, text, SCHED_K_OLD, SCHED_K_NEW, "num_spec_tokens_to_schedule")
    return text


def patch_cudagraph_text(text: str, name: str = "cudagraph_utils.py") -> str:
    if MARK in text:
        return text
    if text.count(CG_ANCHOR) != 1:
        raise SystemExit(f"{name}: helper insert point not unique")
    text = text.replace(CG_ANCHOR, CG_HELPER + CG_ANCHOR, 1)
    text = replace_once(name, text, CG_OLD, CG_NEW, "decode_query_lens")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheduler-in", default=str(HERE.parent / "apc" / "scheduler.py"))
    ap.add_argument("--scheduler-out", default=str(HERE / "scheduler.py"))
    ap.add_argument("--cg-in", default=str(HERE / "cudagraph_utils.orig.py"))
    ap.add_argument("--cg-out", default=str(HERE / "cudagraph_utils.py"))
    a = ap.parse_args(argv)
    for src, dst, fn in (
        (Path(a.scheduler_in), Path(a.scheduler_out), patch_scheduler_text),
        (Path(a.cg_in), Path(a.cg_out), patch_cudagraph_text),
    ):
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        out = fn(src.read_text(), src.name)
        compile(out, dst.name, "exec")  # syntax gate before writing
        if dst.is_file() and dst.read_text() == out:
            print(f"{dst}: up to date")
            continue
        dst.write_text(out)
        print(f"wrote {dst} ({out.count(MARK)} markers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
