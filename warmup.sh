#!/bin/bash
# Warmup post-boot de la pile nightly : attend /health, puis 3 petites requetes et un
# prefill 8K (JIT/autotune a chaud). Detecte l'alea de premiere requete (worker fige)
# au boot plutot qu'en trafic reel. Usage: ./warmup.sh [port]   (defaut 8888)
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
  [ "$code" = 200 ] || { echo "WARMUP FAILED ($1) — pile probablement figee : stop/start des DEUX rangs"; exit 1; }
}
req small-1 "Bonjour, réponds en un mot." 8
req small-2 "Write a Python one-liner that reverses a string." 40
req small-3 "Explique en deux phrases ce qu'est un cache KV." 60
# Length sweep: Triton/FlashInfer autotune caches are keyed on shapes; every new key costs a
# 7-10 s stall on a user request. Sweep the common prompt sizes once at boot instead.
i=0
for reps in ${SWEEP:-4 8 16 30 60 100 150 220 300 450 600 750}; do
  i=$((i+1))
  LONG=$(python3 -c "import sys;print('Text %d: '%$i + ('The key-value cache stores attention keys and values of previous tokens so they are not recomputed. '*$reps) + 'One word: what is this about?')")
  req "sweep-$reps" "$LONG" 8
done
echo "WARMUP OK"
