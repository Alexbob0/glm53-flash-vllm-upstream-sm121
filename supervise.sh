#!/bin/bash
# Boot supervisor for the two-node nightly stack: launches worker (node2) + head, waits for
# /health (bounded), runs warmup.sh; on a boot hang or a warmup failure (the breakable-cudagraph
# capture race: py-spy shows both ranks stuck in KDA kernel launches under capture_model) it
# restarts BOTH ranks, up to MAX_TRIES. Exit 0 when the stack is warm and serving.
# Usage: ./supervise.sh   (env knobs as run.sh; MODEL_SNAP DRAFT_SNAP HEAD_IP NCCL_IF NCCL_HCA required)
set -u
WORKER_HOST="${WORKER_HOST:-node2-ib}"
PORT="${PORT:-8888}"
MAX_TRIES="${MAX_TRIES:-3}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"     # seconds to reach /health (normal boot ~8 min)
DIR="${GLM53_DIR:-$(dirname "$(readlink -f "$0")")}"
ENVS="SPEC=${SPEC:-dflash} K=${K:-7} SEQS=${SEQS:-6} MAX_LEN=${MAX_LEN:-1000000} GMU=${GMU:-0.87} PORT=$PORT BREAKABLE=${BREAKABLE:-1} PMU=${PMU-64} RETENTION=${RETENTION-4608} EAGLE_DROP=${EAGLE_DROP:-0} FAT_STREAMS=${FAT_STREAMS:-4} FAT_GROUPED=${FAT_GROUPED:-1} MNBT=${MNBT:-7168} IMG=${IMG:-glm53-upstream:latest} HF_CACHE=${HF_CACHE:-$HOME/hf} MODEL_SNAP=$MODEL_SNAP DRAFT_SNAP=$DRAFT_SNAP HEAD_IP=$HEAD_IP NCCL_IF=$NCCL_IF NCCL_HCA=$NCCL_HCA"
for try in $(seq 1 "$MAX_TRIES"); do
  echo "[supervise] try $try/$MAX_TRIES ($ENVS) $(date +%T)"
  ssh "$WORKER_HOST" "docker rm -f glm53-up-worker >/dev/null 2>&1; true"; docker rm -f glm53-up-head >/dev/null 2>&1
  until [ "$(free -g | awk 'NR==2{print $7}')" -ge "${MEM_FREE_MIN:-110}" ]; do sleep 5; done
  ssh "$WORKER_HOST" "until [ \$(free -g | awk 'NR==2{print \$7}') -ge "${MEM_FREE_MIN:-110}" ]; do sleep 5; done; setsid nohup env $ENVS $DIR/run.sh worker > $DIR/logs/worker.log 2>&1 < /dev/null &"
  sleep 8
  (env $ENVS setsid nohup "$DIR/run.sh" head > "$DIR/logs/head.log" 2>&1 < /dev/null &)
  t0=$(date +%s); ok=0
  while [ $(( $(date +%s) - t0 )) -lt "$BOOT_TIMEOUT" ]; do
    curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200 && { ok=1; break; }
    docker ps -a --format '{{.Names}} {{.Status}}' | grep -q "glm53-up-head.*Exited" && break
    sleep 15
  done
  if [ "$ok" = 1 ] && "$DIR/warmup.sh" "$PORT"; then echo "[supervise] serving on :$PORT after try $try"; exit 0; fi
  echo "[supervise] boot/warmup failed on try $try — saving logs"; docker logs glm53-up-head > "$DIR/logs/head-hang-try$try.log" 2>&1
  ssh "$WORKER_HOST" "docker logs glm53-up-worker > $DIR/logs/worker-hang-try$try.log 2>&1; true"
done
echo "[supervise] GAVE UP after $MAX_TRIES tries"; exit 1
