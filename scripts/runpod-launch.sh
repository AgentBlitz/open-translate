#!/usr/bin/env bash
# Launch open-translate on a RunPod pod.
#
# Assumes the repo lives at /workspace/open-translate and a venv with
# --system-site-packages (inheriting the pod's pre-installed torch) lives
# at /workspace/venv. Run scripts/runpod-install.sh first if neither
# exists.
#
# Usage:
#   bash scripts/runpod-launch.sh            # default: start on :8000
#   PORT=8005 bash scripts/runpod-launch.sh  # override port
#   bash scripts/runpod-launch.sh stop       # stop the running server
#   bash scripts/runpod-launch.sh status     # show pid + health + tail
#   bash scripts/runpod-launch.sh restart    # stop + start
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/open-translate}"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
PID_FILE="$REPO_DIR/server.pid"
LOG_FILE="$REPO_DIR/server.log"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NLLB_MODEL_SIZE="${NLLB_MODEL_SIZE:-1.3B-distilled}"
export DTYPE="${DTYPE:-fp16}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/workspace/.cache/paddlex}"
export OCR_LANG="${OCR_LANG:-en}"

cmd="${1:-start}"

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

stop_server() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "stopping pid $pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "force-killing $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "no running server"
  fi
  rm -f "$PID_FILE"
}

start_server() {
  if is_running; then
    echo "already running (pid $(cat "$PID_FILE")) — use restart to reload"
    return 0
  fi

  if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: $REPO_DIR not found" >&2
    exit 1
  fi
  if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    echo "  create one: python3 -m venv --system-site-packages $VENV_DIR" >&2
    exit 1
  fi

  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  echo "launching uvicorn on $HOST:$PORT (model=$NLLB_MODEL_SIZE, dtype=$DTYPE)"
  nohup python -m uvicorn server:app \
    --host "$HOST" --port "$PORT" --workers 1 --no-access-log \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  disown || true
  echo "pid $(cat "$PID_FILE") — log: $LOG_FILE"

  echo -n "waiting for /health"
  for i in $(seq 1 120); do
    sleep 1
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo " -> OK after ${i}s"
      curl -s "http://127.0.0.1:$PORT/health"
      echo
      return 0
    fi
    echo -n "."
  done
  echo
  echo "WARNING: /health not responding after 120s — tail:" >&2
  tail -30 "$LOG_FILE" >&2
  return 1
}

status_server() {
  if is_running; then
    echo "running: pid $(cat "$PID_FILE")"
    curl -s "http://127.0.0.1:$PORT/health" || echo "health endpoint not responding"
    echo
    echo "--- last 20 log lines ---"
    tail -20 "$LOG_FILE" 2>/dev/null || true
  else
    echo "not running"
    [ -f "$LOG_FILE" ] && { echo "--- last 20 log lines ---"; tail -20 "$LOG_FILE"; }
  fi
}

case "$cmd" in
  start)   start_server ;;
  stop)    stop_server ;;
  restart) stop_server; start_server ;;
  status)  status_server ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
