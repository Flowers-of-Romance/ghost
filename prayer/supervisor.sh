#!/bin/bash
# prayer daemon supervisor — MLX SIGBUS/SEGV で落ちる前提で auto-restart する。
# 使用:
#   ./prayer/supervisor.sh [--session-prefix NAME] [--audio-device N] [追加オプション]
# 例:
#   ./prayer/supervisor.sh --audio-device 0

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
MODEL="$SCRIPT_DIR/models/qwen3-omni-30b-bf16"
LOG="${PRAYER_LOG:-/tmp/prayer-daemon.log}"
SESSION_PREFIX="prayer"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --session-prefix) SESSION_PREFIX="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "[supervisor] log: $LOG"
echo "[supervisor] PID: $$"

# ctrl-c で supervisor ごと終了
trap 'echo "[supervisor] 停止"; kill $CHILD 2>/dev/null; exit 0' INT TERM

: > "$LOG"
cd "$REPO_DIR"
RESTART_COUNT=0
while true; do
    SESSION_ID="${SESSION_PREFIX}-$(date +%s)"
    echo "[supervisor] start session=$SESSION_ID (restart #$RESTART_COUNT)" >> "$LOG"
    "$PY" prayer/daemon.py \
        --session-id "$SESSION_ID" \
        --model "$MODEL" \
        --audio \
        "${EXTRA_ARGS[@]}" >> "$LOG" 2>&1 &
    CHILD=$!
    wait $CHILD
    EXIT_CODE=$?
    echo "[supervisor] daemon exited with $EXIT_CODE — restarting in 3s" >> "$LOG"
    RESTART_COUNT=$((RESTART_COUNT + 1))
    # 終了が速すぎる場合はループ過剰を避けて長めに待つ
    sleep 3
done
