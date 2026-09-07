#!/usr/bin/env python3
"""Multi-turn append test + spec acceptance, stdlib only. Usage: BASE=http://127.0.0.1:8888 python3 apc_turns.py <label>
Turn 1 = ~8.5K-token code prompt (fresh salt), turns 2-3 append short questions; prints prefix-cache hit deltas and
DFlash acceptance (accepted/draft) over the 3 turns."""
import json, os, sys, time, urllib.request, uuid
BASE = os.environ.get('BASE', 'http://127.0.0.1:8888'); LABEL = sys.argv[1] if len(sys.argv) > 1 else 'turns'
ROOT = os.path.dirname(os.path.abspath(__file__))
def prom():
    out = {}
    for l in urllib.request.urlopen(BASE + '/metrics', timeout=30).read().decode().splitlines():
        for k in ('prefix_cache_hits_total', 'prefix_cache_queries_total', 'spec_decode_num_accepted_tokens_total', 'spec_decode_num_draft_tokens_total'):
            if l.startswith('vllm:' + k): out[k] = float(l.rsplit(' ', 1)[1])
    return out
def chat(msgs, n, label):
    body = {'model': 'dgx-spark', 'messages': msgs, 'max_tokens': n, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}}
    b = prom(); t0 = time.perf_counter(); ttft = None; text = ''; usage = None
    with urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            ev = json.loads(line[6:])
            if ev.get('usage'): usage = ev['usage']
            for ch in ev.get('choices', []):
                d = ch.get('delta', {}).get('content')
                if d:
                    if ttft is None: ttft = time.perf_counter() - t0
                    text += d
    a = prom(); d = {k: a[k] - b[k] for k in a}
    acc = d['spec_decode_num_accepted_tokens_total'] / d['spec_decode_num_draft_tokens_total'] if d.get('spec_decode_num_draft_tokens_total') else float('nan')
    print(f"{LABEL} {label}: prompt={usage['prompt_tokens']} gen={usage['completion_tokens']} TTFT={ttft:.2f}s total={time.perf_counter()-t0:.1f}s "
          f"hits={d['prefix_cache_hits_total']:.0f}/{d['prefix_cache_queries_total']:.0f} accept={acc:.3f}")
    return text
src = ''.join(open(os.path.join(ROOT, 'overlay', f)).read() for f in sorted(os.listdir(os.path.join(ROOT, 'overlay'))) if f.endswith('.py'))[:int(os.environ.get('TURN_CHARS', '30000'))]
msgs = [{'role': 'user', 'content': f'[{uuid.uuid4()}] Voici du code source. Résume en trois phrases ce qu il fait.\n\n' + src}]
r1 = chat(msgs, 150, 'turn1')
msgs += [{'role': 'assistant', 'content': r1}, {'role': 'user', 'content': 'Continue avec une phrase de plus.'}]
r2 = chat(msgs, 40, 'turn2')
msgs += [{'role': 'assistant', 'content': r2}, {'role': 'user', 'content': 'Et une dernière.'}]
chat(msgs, 40, 'turn3')
