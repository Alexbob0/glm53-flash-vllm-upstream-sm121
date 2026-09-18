#!/usr/bin/env python3
"""Kernel time by outermost cpu op (torch profiler trace, no stack needed)."""
import gzip, json, re, sys
from collections import defaultdict
from bisect import bisect_right
for p in sys.argv[1:]:
    t = json.load(gzip.open(p,'rt'))['traceEvents']
    kern = [e for e in t if e.get('ph')=='X' and e.get('cat')=='kernel']
    launch = {e['args'].get('correlation'): e for e in t if e.get('ph')=='X' and e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    ops = defaultdict(list)
    for e in t:
        if e.get('ph')=='X' and e.get('cat')=='cpu_op': ops[e['tid']].append((e['ts'], e['ts']+e['dur'], e['name']))
    for tid in ops: ops[tid].sort()
    starts = {tid:[o[0] for o in v] for tid,v in ops.items()}
    def chain(tid, ts):
        v=ops.get(tid,[]); i=bisect_right(starts[tid],ts)-1; c=[]
        while i>=0:
            a,b,n=v[i]
            if a<=ts<=b: c.append(n)
            if b<ts-5e6: break
            i-=1
        return c
    agg=defaultdict(lambda:[0,0.0,defaultdict(float)])
    for k in kern:
        l=launch.get(k['args'].get('correlation')); c=chain(l['tid'],l['ts']) if l else []
        o=c[-1] if c else 'KDA/other (no op)'; inner=c[0] if c else '-'
        agg[o][0]+=1; agg[o][1]+=k['dur']; agg[o][2][inner+' | '+re.sub(r'\(.*','',k['name'])[:50]]+=k['dur']
    tot=sum(v[1] for v in agg.values())/1000
    print(f'\n== {p.split("/")[-2]}  total {tot:.0f} ms')
    for o,(c,us,sub) in sorted(agg.items(), key=lambda x:-x[1][1])[:12]:
        print(f'{us/1000:8.1f} ms {100*us/1000/tot:5.1f}% {c:6d}x  {o}')
        for n,u in sorted(sub.items(), key=lambda x:-x[1])[:8]: print(f'            {u/1000:7.1f} ms  {n}')
