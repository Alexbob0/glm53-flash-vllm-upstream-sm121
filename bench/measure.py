#!/usr/bin/env python3
"""Auditable GLM streaming benchmark; stdlib only, no serving changes.

All timestamps use a process-wide monotonic clock. Tokens are counted from
streamed token_ids and checked against usage; SSE chunks are never called tokens.
Prompt/TTFT is a client proxy including scheduling/decode, not GPU prefill speed.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import threading
import time
import urllib.request
import uuid


def request_json(base, path, data=None, timeout=60):
    body = None if data is None else json.dumps(data).encode()
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    req = urllib.request.Request(base + path, body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw.strip() else None


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    return xs[lo] + (xs[math.ceil(i)] - xs[lo]) * (i - lo)


def summarize_stream(record):
    events = record['events']
    visible = []
    generated = []
    usage = None
    errors = []
    server_metrics = None
    for event in events:
        chunk, t = event['data'], event['t']
        if chunk.get('error'):
            errors.append(chunk['error'])
        if chunk.get('metrics'):
            server_metrics = chunk['metrics']
        if chunk.get('usage'):
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            if any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content', 'tool_calls')):
                visible.append(t)
            ids = choice.get('token_ids')
            if ids:
                generated.append((t, len(ids)))
    count = sum(n for _, n in generated)
    expected = usage.get('completion_tokens') if usage else None
    intervals = [b - a for a, b in zip(visible, visible[1:])]
    tf = visible[0] if visible else None
    first = generated[0][0] if generated else None
    last = generated[-1][0] if generated else None
    duration = last - first if first is not None else 0
    decode_count = count - generated[0][1] if generated else 0
    result = {
        'ttft_visible_s': tf - record['start'] if tf is not None else None,
        'ttft_generated_s': first - record['start'] if first is not None else None,
        'request_s': record['end'] - record['start'],
        'usage': usage,
        'server_metrics': server_metrics,
        'streamed_tokens': count,
        'token_count_matches_usage': expected is not None and count == expected,
        'decode_after_first_event_tps': decode_count / duration if duration > 0 else None,
        'inter_visible_event_p50_s': percentile(intervals, .50),
        'inter_visible_event_p95_s': percentile(intervals, .95),
        'inter_visible_event_max_s': max(intervals) if intervals else None,
        'done': record.get('done', False),
        'errors': errors,
    }
    result['valid'] = bool(result['done'] and usage and generated and not errors and
                           not record.get('error') and result['token_count_matches_usage'])
    # Preserve enough information to recompute any common overlap window.
    result['generated_events'] = generated
    return result


def summarize_group(records):
    summaries = [r['summary'] for r in records]
    valid = all(s['valid'] for s in summaries)
    elapsed = max(r['end'] for r in records) - min(r['start'] for r in records)
    result = {'valid': valid, 'requests': len(records), 'wall_s': elapsed,
              'end_to_end_tps': sum(s['streamed_tokens'] for s in summaries) / elapsed if valid and elapsed else None,
              'ttft_p50_s': percentile([s['ttft_visible_s'] for s in summaries if s['ttft_visible_s'] is not None], .5),
              'ttft_p95_s': percentile([s['ttft_visible_s'] for s in summaries if s['ttft_visible_s'] is not None], .95),
              'steady_window_s': None, 'steady_window_tps': None}
    if valid:
        lo = max(s['generated_events'][0][0] for s in summaries)
        hi = min(s['generated_events'][-1][0] for s in summaries)
        if hi > lo:
            tokens = sum(n for s in summaries for t, n in s['generated_events'] if lo < t <= hi)
            result.update(steady_window_s=hi - lo, steady_window_tps=tokens / (hi - lo),
                          steady_window_tokens=tokens, steady_window_bounds=[lo, hi])
    return result


def stream(base, body, path, barrier=None, stagger=0):
    path.write_text(json.dumps({'request': body}, ensure_ascii=False) + '\n')
    record = {'events': [], 'done': False}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(), headers)
    if barrier:
        barrier.wait(timeout=30)
    if stagger:
        time.sleep(stagger)
    record['start'] = time.perf_counter()
    with path.open('a') as output:
        output.write(json.dumps({'start': record['start'], 'utc': time.time()}) + '\n')
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    t = time.perf_counter()
                    if t - record['start'] > 900:
                        raise TimeoutError('request exceeded 900 seconds')
                    line = raw.decode().strip()
                    if not line.startswith('data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == '[DONE]':
                        record['done'] = True
                        break
                    event = {'t': t, 'data': json.loads(payload)}
                    record['events'].append(event)
                    output.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['end'] = time.perf_counter()
        record['summary'] = summarize_stream(record)
        output.write(json.dumps({k: v for k, v in record.items() if k != 'events'}) + '\n')
    return record


def metrics(base, path):
    try:
        with urllib.request.urlopen(base + '/metrics', timeout=10) as resp:
            path.write_bytes(resp.read())
    except Exception as exc:
        path.write_text(f'# METRICS ERROR {exc}\n')


TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a source file.',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]


def corpus_text():
    root = Path(__file__).resolve().parent
    files = sorted((root / 'overlay').glob('*.py'))
    files += sorted((root / 'exllamav3-src/exllamav3').rglob('*.py'))
    # Fixed order and content hashes recorded in each request artifact.
    chunks = ['\n# FILE ' + str(p.relative_to(root)) + '\n' + p.read_text() for p in files]
    return '\n'.join(chunks)


def messages_for(kind, chars, salt, corpus):
    prefix = f'Benchmark case {salt}. You are reviewing a Python inference project.'
    if kind == 'code':
        text = (corpus * (chars // len(corpus) + 1))[:chars]
        return [{'role': 'system', 'content': prefix}, {'role': 'user', 'content':
                text + '\nReview this code. Explain concrete correctness and performance issues in detail.'}]
    messages = [{'role': 'system', 'content': prefix}]
    text = (corpus * (chars // len(corpus) + 1))[:chars]
    for i, start in enumerate(range(0, len(text), 8000)):
        messages.extend([
            {'role': 'user', 'content': f'Inspect source segment {i} and continue the review.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call_{i}', 'type': 'function',
             'function': {'name': 'read_file', 'arguments': json.dumps({'path': f'segment_{i}.py'})}}]},
            {'role': 'tool', 'tool_call_id': f'call_{i}', 'content': text[start:start + 8000]},
            {'role': 'assistant', 'content': f'Segment {i} read. I will compare its allocation and synchronization behavior with the other segments.'},
        ])
    messages.append({'role': 'user', 'content': 'Synthesize the review so far, with concrete code changes and their validation. Answer directly without calling tools.'})
    return messages


def prepare(base, model, kind, size, corpus):
    salt = uuid.uuid4().hex
    chars = size * 3
    for _ in range(8):
        messages = messages_for(kind, chars, salt, corpus)
        body = {'model': model, 'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}}
        if kind == 'agent':
            body['tools'] = TOOLS
            body['tool_choice'] = 'none'
        tokenized = request_json(base, '/tokenize', body)
        count = tokenized['count']
        if abs(count - size) <= max(64, size * .01):
            break
        chars = max(100, round(chars * size / count))
    if abs(count - size) > max(64, size * .02):
        raise ValueError(f'could not size {kind} prompt: wanted {size}, got {count}')
    body.update(temperature=0, stream=True, stream_options={'include_usage': True}, return_token_ids=True)
    return body, {'kind': kind, 'target_tokens': size, 'tokenized_tokens': count,
                  'messages_sha256': hashlib.sha256(json.dumps(messages).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=os.environ.get('BENCH_BASE', 'http://127.0.0.1:8890'))
    parser.add_argument('--model', default='GLM-5.3-Flash-EXL3')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kinds', nargs='+', choices=['code', 'agent'], default=['code', 'agent'])
    parser.add_argument('--sizes', nargs='+', type=int, default=[8000, 32000, 100000])
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--stagger', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=1)
    parser.add_argument('--profile', action='store_true', help='Warmup and repeat with 150 output tokens, then profile one fresh-salt request.')
    args = parser.parse_args()
    if args.concurrency < 1 or args.passes < 1 or args.stagger < 0:
        parser.error('positive concurrency/passes and nonnegative stagger required')
    if args.profile and args.concurrency != 1:
        parser.error('one-iteration profiling requires concurrency=1')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    (args.out / 'models.json').write_text(json.dumps(request_json(args.base, '/v1/models'), indent=2))
    corpus = corpus_text()
    (args.out / 'corpus.sha256').write_text(hashlib.sha256(corpus.encode()).hexdigest() + '\n')
    all_groups = []
    for kind in args.kinds:
        for size in args.sizes:
            for repeat in range(args.passes):
                prepared = [prepare(args.base, args.model, kind, size, corpus) for _ in range(args.concurrency)]
                # Each pass has new salts; immediate exact repeats probe APC independently.
                phases = ['warmup150', 'repeat150', 'profile'] if args.profile else ['unique', 'repeat']
                for phase in phases:
                    if phase == 'profile':
                        # Warmup may populate APC; fresh salt ensures the profile is a prefill.
                        prepared = [prepare(args.base, args.model, kind, size, corpus)]
                    label = f'{kind}-{size}-p{repeat}-{phase}'
                    metrics(args.base, args.out / f'{label}-before.prom')
                    if phase == 'profile':
                        request_json(args.base, '/start_profile', {})
                    barrier = threading.Barrier(args.concurrency)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                        futures = [pool.submit(stream, args.base, dict(body, max_tokens=150 if phase in ('warmup150', 'repeat150') else args.max_tokens),
                                   args.out / f'{label}-r{i}.jsonl', barrier, args.stagger * i)
                                   for i, (body, meta) in enumerate(prepared)]
                        records = [f.result() for f in futures]
                    if phase == 'profile':
                        request_json(args.base, '/stop_profile', {}, timeout=180)
                    metrics(args.base, args.out / f'{label}-after.prom')
                    group = {'label': label, 'prompts': [m for _, m in prepared],
                             'summary': summarize_group(records),
                             'streams': [{k: v for k, v in r['summary'].items() if k != 'generated_events'} for r in records]}
                    all_groups.append(group)
                    (args.out / 'summary.json').write_text(json.dumps(all_groups, indent=2))
                    print(json.dumps(group), flush=True)
                    if not group['summary']['valid']:
                        raise RuntimeError(f'invalid stream in {label}; inspect JSONL artifacts')


if __name__ == '__main__':
    main()
