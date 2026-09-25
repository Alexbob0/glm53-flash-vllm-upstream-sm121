"""Tests the GLM53_NGRAM kernels against a Python reference (TRITON_INTERPRET=1, CPU: no CUDA context).
Usage: docker exec -e TRITON_INTERPRET=1 glm53-up-head python3 /tmp/ngram/test_glm53_ngram.py
"""
import os, random, sys, types
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["GLM53_NGRAM"] = "1"
os.environ["GLM53_NGRAM_LOG"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import glm53_ngram as g

g._CHUNK = 64  # several chunks even on short sequences


def ref_best(seq, nmin, nmax=16):
    L = len(seq); best = None
    for e in range(nmin, L):            # fin e <= L-1
        m = 0
        for j in range(1, nmax + 1):
            if e - j < 0 or L - j < 0 or seq[e - j] != seq[L - j]:
                break
            m += 1
        if m >= nmin and (best is None or (m, e) > best):
            best = (m, e)
    return best


def run_case(seqs, K=7, TOPK=4, V=97, nmin=4):
    g._NMIN = nmin
    R = len(seqs); maxlen = 512
    tokens = torch.zeros(R + 1, maxlen, dtype=torch.int32)
    total = torch.zeros(R + 1, dtype=torch.int32)
    order = list(range(R))[::-1]                        # idx_mapping non trivial
    for i, s in enumerate(seqs):
        st = order[i]; tokens[st, :len(s)] = torch.tensor(s, dtype=torch.int32); total[st] = len(s)
    idx = torch.tensor(order, dtype=torch.int32)
    draft = torch.full((R, K), -7, dtype=torch.int64)
    logits = torch.full((R + 1, K, V), float("-inf"))
    cand = torch.zeros(R + 1, K, TOPK, dtype=torch.int64)
    for st in range(R + 1):                             # cache creux type DFlash2 : TOPK candidats finis
        for j in range(K):
            c = torch.randperm(V)[:TOPK]; cand[st, j] = c; logits[st, j, c] = torch.randn(TOPK)
    req_states = types.SimpleNamespace(max_num_reqs=R + 1, all_token_ids=types.SimpleNamespace(gpu=tokens),
                                       total_len=types.SimpleNamespace(gpu=total))
    spec = types.SimpleNamespace(draft_logits=logits, _cached_candidate_ids=cand)
    g._state.update(calls=0, on=True, best=None, stats=None)
    g.maybe_override(draft, types.SimpleNamespace(idx_mapping=idx), req_states, spec)
    for i, s in enumerate(seqs):
        st = order[i]; b = ref_best(s, nmin)
        if b is None:
            assert (draft[i] == -7).all(), (i, draft[i])
            continue
        m, e = b; cnt = min(len(s) - e, K)
        exp = s[e:e + cnt]
        assert draft[i, :cnt].tolist() == exp, (i, draft[i].tolist(), exp)
        assert (draft[i, cnt:] == -7).all()
        for j in range(cnt):
            row = logits[st, j]
            assert row[exp[j]] == 0 and torch.isinf(row).sum() == V - 1, (i, j)
            assert (cand[st, j] == exp[j]).all()
    return g._state["stats"].tolist()


random.seed(0); torch.manual_seed(0)
# 1) long copy: the end repeats an earlier passage
base = [random.randrange(97) for _ in range(200)]
s1 = base + [random.randrange(97) for _ in range(50)] + base[40:70]
# 2) no repetition (large alphabet -> almost no repeated 4-gram)
s2 = [random.randrange(90) for _ in range(150)]
# 3) sequence shorter than NMAX, repeated pattern
s3 = [5, 6, 7, 8, 1, 2, 5, 6, 7, 8]
# 4) repetition whose continuation is cut by the end of sequence (count < K)
s4 = [9, 9, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5]
# 5) several occurrences: longest, then latest, must win
s5 = [1, 2, 3, 4, 50, 51, 7, 7, 2, 3, 4, 60, 61, 8, 1, 2, 3, 4]
stats = run_case([s1, s2, s3, s4, s5])
print("stats", stats)
for s in (s1, s2, s3, s4, s5):
    print("ref", ref_best(s, 4))
print("OK")
