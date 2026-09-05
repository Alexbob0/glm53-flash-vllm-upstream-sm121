#!/usr/bin/env python3
"""Concurrency at long context: C distinct ~N-token prompts fired together, 256 output tokens each.
Reports per-stream decode (TTFT excluded) and aggregate over the steady-state window."""
import json, os, sys, time, threading, urllib.request, random
URL=os.environ.get("BENCH_URL","http://127.0.0.1:8888/v1/chat/completions"); MODEL=os.environ.get("BENCH_MODEL","GLM-5.3-Flash-EXL3")
C=int(sys.argv[1]); NREP=int(sys.argv[2]) if len(sys.argv)>2 else 600   # 600 reps ~ 16-17K tokens
base="The key-value cache stores the attention keys and values of previous tokens so they are not recomputed. "
def one(i, out):
    random.seed(i); salt=" ".join(random.choice(["alpha","beta","gamma","delta","omega","sigma"]) for _ in range(20))
    msg=f"Document {i} ({salt}):\n"+base*NREP+"\n\nWrite a detailed, well-structured technical summary of what a KV cache is, why it matters, and how paged attention manages it. Be thorough."
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":msg}],"max_tokens":256,"temperature":0,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(URL,body,{"Content-Type":"application/json"})
    t0=time.time(); tf=None; comp=0; ptok=0
    with urllib.request.urlopen(req,timeout=1200) as r:
        for raw in r:
            line=raw.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            ch=json.loads(line[6:])
            if ch.get("usage"): comp=ch["usage"]["completion_tokens"]; ptok=ch["usage"]["prompt_tokens"]
            if ch.get("choices") and ch["choices"][0].get("delta",{}).get("content") and tf is None: tf=time.time()
    out[i]=(ptok,comp,tf-t0,time.time()-tf)
out={}; th=[threading.Thread(target=one,args=(i,out)) for i in range(C)]
T0=time.time(); [t.start() for t in th]; [t.join() for t in th]; T=time.time()-T0
tot=sum(v[1] for v in out.values()); 
last_first=max(v[2] for v in out.values()); first_end=min(v[2]+v[3] for v in out.values())
per=[ (v[1]-1)/v[3] for v in out.values()]
print(f"C={C} prompt~{out[0][0]} tok | TTFT {min(v[2] for v in out.values()):.1f}-{last_first:.1f}s | per-stream decode {min(per):.1f}-{max(per):.1f} tok/s (mean {sum(per)/C:.1f}) | aggregate {tot/T:.1f} tok/s over {T:.0f}s")
