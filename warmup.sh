#!/bin/bash
# Post-boot warmup: wait for /health, then 3 small requests and one 8K prefill (JIT/autotune).
# Surfaces the first-request hang (see PITFALLS.md) at boot instead of under real traffic.
# Usage: ./warmup.sh [port]   (default 8888)
PORT="${1:-8888}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"
until curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200; do sleep 10; done
MODEL=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
req() {  # $1 = label, $2 = prompt, $3 = max_tokens
  local t0=$(date +%s.%N)
  local code
  code=$(curl -s -m 300 -o /tmp/warmup.out -w '%{http_code}' "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$2")}],\"max_tokens\":$3,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  printf "%-12s http %s  %.1fs\n" "$1" "$code" "$(echo "$(date +%s.%N) - $t0" | bc)"
  [ "$code" = 200 ] || { echo "WARMUP FAILED ($1) — stack is probably hung: stop and restart BOTH ranks"; exit 1; }
}
req small-1 "Hello, answer in one word." 8
req small-2 "Write a Python one-liner that reverses a string." 40
req small-3 "Explique en deux phrases ce qu'est un cache KV." 60
LONG=$(python3 -c "print('Le cache KV stocke les cles et valeurs de l attention pour eviter de recalculer les tokens precedents. '*300 + 'Resume en une phrase.')")
req prefill-8K "$LONG" 20
echo "WARMUP OK"
