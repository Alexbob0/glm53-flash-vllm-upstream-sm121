import json,time,urllib.request,sys
import os
URL="http://127.0.0.1:8888/v1/chat/completions"
base="Le cache KV stocke les cles et valeurs de l'attention pour eviter de recalculer les tokens precedents. "
n_rep=int(sys.argv[1]) if len(sys.argv)>1 else 3600
msg=(base*n_rep)+"\n\nQuestion : explique en 150 mots comment fonctionne un cache KV et pourquoi il accelere la generation."
body={"model":os.environ.get("BENCH_MODEL","GLM-5.3-Flash-EXL3"),"messages":[{"role":"user","content":msg}],"max_tokens":150,"temperature":0,"stream":True,"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.time();first=None;txt="";n=0
with urllib.request.urlopen(req,timeout=900) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:]); ch=d["choices"]
        if ch and ch[0]["delta"].get("content"):
            if first is None: first=time.time()
            txt+=ch[0]["delta"]["content"]; n+=1
print(f"ctx~{n_rep*28//1000}K: TTFT {first-t0:.1f}s, decode {n/(time.time()-first):.1f} tok/s ({n} chunks)")
