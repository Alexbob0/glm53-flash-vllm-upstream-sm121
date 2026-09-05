#!/usr/bin/env python3
"""Prefill à froid : TTFT client sur prompts uniques (anti prefix-cache),
max_tokens=1. tok/s ≈ prompt_tokens / TTFT. 2 passes par taille."""
import json, time, urllib.request, uuid, sys

import os
URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "GLM-5.3-Flash-EXL3")
BLOCK = ("Le cache LRU conserve les elements recemment utilises et expulse les "
         "plus anciens lorsque la capacite est atteinte. ")  # ~25 tokens

def run(n_target):
    reps = max(1, n_target // 25)
    prompt = f"[{uuid.uuid4()}] " + BLOCK * reps + " Reponds OK."
    body = json.dumps({"model": MODEL, "max_tokens": 1, "temperature": 0,
                       "stream": True,
                       "messages": [{"role": "user", "content": prompt}],
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; ptok = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if t_first is None and d.get("choices"):
                t_first = time.time()
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
    ttft = (t_first or time.time()) - t0
    print(f"  ~{ptok:6d} tok  TTFT {ttft:6.2f} s  -> {ptok/ttft:7.0f} tok/s")

for size in (8000, 32000):
    print(f"prefill ~{size}:")
    for _ in range(2):
        run(size)
