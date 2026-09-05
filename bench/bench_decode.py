#!/usr/bin/env python3
"""Decode benchmark, same protocol as MiaAI-Lab's tests/bench_decode.py:
streaming, temperature 0, thinking off, tok/s = (completion_tokens - 1) / (end - first_token),
i.e. TTFT EXCLUDED. Median of 3 per probe. The first two probes are their exact prompts.

  BENCH_URL=http://127.0.0.1:8888/v1/chat/completions BENCH_MODEL=GLM-5.3-Flash-EXL3 ./bench_decode.py
"""
import json, os, sys, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8888/v1/chat/completions")
MODEL = os.environ.get("BENCH_MODEL", "GLM-5.3-Flash-EXL3")

PROBES = [
    ("structured(count)", "Count from 1 to 200. Output only the numbers, "
     "separated by spaces. No other text.", 200),
    ("prose(hashmap)", "Write a detailed step-by-step explanation of how a "
     "hash map works, including collision handling, resizing, and time "
     "complexity. Be thorough.", 200),
    ("code(bst)", "Write a complete, commented Python implementation of a binary "
     "search tree (insert, search, delete, in-order traversal).", 400),
]


def run(msg, n):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": n, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t_first = None
    completion = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                completion = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = time.time()
    t_end = time.time()
    return (completion - 1) / (t_end - t_first)


label = sys.argv[1] if len(sys.argv) > 1 else "run"
run("Say hello.", 32)  # warmup
print(f"=== {label} (MiaAI protocol, TTFT excluded, median of 3) ===")
for name, msg, n in PROBES:
    vals = sorted(run(msg, n) for _ in range(3))
    print(f"{name:18s} med {vals[1]:5.1f}  (min {vals[0]:5.1f} / max {vals[2]:5.1f}) tok/s")
