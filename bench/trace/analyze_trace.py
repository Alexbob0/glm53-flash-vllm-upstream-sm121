#!/usr/bin/env python3
"""Summarize Chrome/PyTorch GPU traces without treating kernel sums as wall time."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def union_us(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def analyze(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        trace = json.load(f)
    kernels = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    by_name = defaultdict(lambda: {'calls': 0, 'sum_us': 0})
    intervals = []
    for e in kernels:
        stats = by_name[e['name']]
        stats['calls'] += 1
        stats['sum_us'] += e['dur']
        intervals.append((e['ts'], e['ts'] + e['dur']))
    ordered = sorted((dict(name=k, **v) for k, v in by_name.items()), key=lambda x: -x['sum_us'])
    runtime = defaultdict(lambda: {'calls': 0, 'sum_us': 0})
    for e in trace['traceEvents']:
        if e.get('ph') == 'X' and e.get('cat') in ['cuda_runtime', 'cuda_driver']:
            runtime[e['name']]['calls'] += 1
            runtime[e['name']]['sum_us'] += e.get('dur', 0)
    result = {'file': str(path), 'events': len(trace['traceEvents']), 'kernel_calls': len(kernels),
              'kernel_sum_ms': sum(e['dur'] for e in kernels) / 1000,
              'kernel_union_ms': union_us(intervals) / 1000,
              'kernel_span_ms': (max(b for a, b in intervals) - min(a for a, b in intervals)) / 1000 if intervals else None,
              'kernels': ordered,
              'runtime': sorted((dict(name=k, **v) for k, v in runtime.items()), key=lambda x: -x['sum_us'])}
    path.with_name(path.name + '.summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({**result, 'kernels': ordered[:18], 'runtime': result['runtime'][:10]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, nargs='+')
    for path in parser.parse_args().trace:
        analyze(path)
