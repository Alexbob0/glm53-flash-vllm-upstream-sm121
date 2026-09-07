#!/bin/bash
# Post-boot measurement chain: waits for supervise log $1, then decode (official), prefill 8K/32K/100K unique+repeat, c4 12K.
# Usage: ./chain.sh <supervise-log> <out-tag>
LOG="$1"; TAG="$2"; cd "$(dirname "$(readlink -f "$0")")"
until grep -qE "serving on|GAVE UP" "$LOG"; do sleep 15; done
grep -E "serving on|GAVE UP" "$LOG"; grep -q "GAVE UP" "$LOG" && exit 1
OUT=measurements/$TAG; mkdir -p $OUT
docker logs glm53-up-head 2>&1 | grep -E "GPU KV cache size" | tail -1 | cut -c1-160
docker logs glm53-up-head 2>&1 | grep -E "exl3 e2 diag" | tail -1 | grep -oE "effective_tier=[a-z]+|direct_calls=[0-9]+"
BENCH_URL=http://127.0.0.1:8888/v1/chat/completions BENCH_MODEL=dgx-spark python3 bench/bench_decode.py "$TAG" 2>&1 | tee $OUT/decode-official.txt
python3 bench/measure.py --base http://127.0.0.1:8888 --out $OUT/prefill --kinds code --sizes 8000 32000 100000 --passes 1 --max-tokens 8 >/dev/null 2>&1
python3 - "$OUT" <<'PY'
import json, sys
for g in json.load(open(sys.argv[1] + '/prefill/summary.json')):
    st = g['streams'][0]; p = st['usage']['prompt_tokens']; t = st['ttft_generated_s']
    print(f"{g['label']:24s} prompt={p} ttft={t:.2f}s -> {p/t:.0f} tok/s valid={st['valid']}")
PY
python3 bench/measure.py --base http://127.0.0.1:8888 --out $OUT/c4 --kinds code --sizes 12000 --passes 1 --concurrency 4 --max-tokens 256 >/dev/null 2>&1
python3 - "$OUT" <<'PY'
import json, sys
for g in json.load(open(sys.argv[1] + '/c4/summary.json')):
    s = g['summary']; print(f"{g['label']:22s} e2e={s['end_to_end_tps']:.1f} tok/s ttft_p50={s['ttft_p50_s']:.1f}s p95={s['ttft_p95_s']:.1f}s wall={s['wall_s']:.1f}s")
PY
