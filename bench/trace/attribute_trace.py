#!/usr/bin/env python3
"""Attribute GPU kernels to the outermost aten op that launched them (needs no stack/shapes)."""
import gzip, json, sys, re
from collections import defaultdict
from bisect import bisect_right
p = sys.argv[1]; only = sys.argv[2] if len(sys.argv) > 2 else None
t = json.load(gzip.open(p, 'rt'))['traceEvents']
kern = [e for e in t if e.get('ph') == 'X' and e.get('cat') == 'kernel']
launch = {e['args'].get('correlation'): e for e in t if e.get('ph') == 'X' and e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'args' in e and 'correlation' in e['args']}
ops = defaultdict(list)
for e in t:
    if e.get('ph') == 'X' and e.get('cat') == 'cpu_op':
        ops[e['tid']].append((e['ts'], e['ts'] + e['dur'], e['name']))
for tid in ops: ops[tid].sort()
starts = {tid: [o[0] for o in v] for tid, v in ops.items()}
def enclosing(tid, ts):
    v = ops.get(tid, []); i = bisect_right(starts[tid], ts) - 1; chain = []
    while i >= 0:
        a, b, n = v[i]
        if a <= ts <= b: chain.append(n)
        if b < ts - 5e6: break
        i -= 1
    return chain  # innermost first
agg = defaultdict(lambda: [0, 0.0]); agg_in = defaultdict(lambda: [0, 0.0])
for k in kern:
    c = k['args'].get('correlation'); l = launch.get(c)
    if not l: agg[('?', k['name'][:60])][0] += 1; agg[('?', k['name'][:60])][1] += k['dur']; continue
    chain = enclosing(l['tid'], l['ts'])
    if only and not re.search(only, k['name']): continue
    outer = chain[-1] if chain else '?'; inner = chain[0] if chain else '?'
    key = (outer, inner, re.sub(r'\(.*', '', k['name'])[:70])
    agg[key][0] += 1; agg[key][1] += k['dur']
tot = sum(v[1] for v in agg.values()) / 1000
print(f"total {tot:.0f} ms")
for key, (c, us) in sorted(agg.items(), key=lambda x: -x[1][1])[:40]:
    print(f"{us/1000:8.1f} ms {c:6d}x  outer={key[0][:40]:40s} inner={key[1][:28]:28s} {key[2]}")
