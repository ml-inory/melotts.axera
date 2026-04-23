#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    build_decoder_b256_kr.sh
    build_decoder_b512_kr.sh
    build_decoder_b1024_kr.sh
    build_decoder_b1536_kr.sh
)

PIDS=()
for script in "${SCRIPTS[@]}"; do
    log="$LOG_DIR/${script%.sh}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $script -> $log"
    bash "$script" > "$log" 2>&1 &
    PIDS+=($!)
done

echo "All builds launched (PIDs: ${PIDS[*]}). Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    script=${SCRIPTS[$i]}
    if wait "$pid"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done    $script (pid=$pid)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED  $script (pid=$pid)" >&2
        FAILED=1
    fi
done

echo ""
echo "All builds complete. Logs saved to $LOG_DIR/"
exit $FAILED
