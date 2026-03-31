#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/workspace/chbx_ja_batch}"
SERVER_PORT="${SERVER_PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$APP_ROOT/runtime_logs"
mkdir -p "$LOG_DIR"

{
  echo "[launch] $(date -u +%FT%TZ)"
  echo "[launch] APP_ROOT=$APP_ROOT"
  echo "[launch] SERVER_PORT=$SERVER_PORT"
  echo "[launch] RCLONE_REMOTE=${RCLONE_REMOTE:-}"
  echo "[launch] VAST_INSTANCE_ID=${VAST_INSTANCE_ID:-}"
  echo "[launch] DESTROY_INSTEAD_OF_STOP=${DESTROY_INSTEAD_OF_STOP:-}"
  echo "[launch] FORK_REPO_URL=${FORK_REPO_URL:-}"
  echo "[launch] FORK_COMMIT_SHA=${FORK_COMMIT_SHA:-}"
} >> "$LOG_DIR/launch_ui.log"

cd "$APP_ROOT"
if ! command -v tmux >/dev/null 2>&1; then
  apt-get update && apt-get install -y tmux
fi

tmux kill-session -t chbx-ui >/dev/null 2>&1 || true
tmux new-session -d -s chbx-ui "cd '$APP_ROOT' && APP_ROOT='$APP_ROOT' SERVER_PORT='$SERVER_PORT' HF_MODEL_REPO_ID='${HF_MODEL_REPO_ID:-}' HF_MODEL_REVISION='${HF_MODEL_REVISION:-}' HF_CACHE_DIR='${HF_CACHE_DIR:-/workspace/.cache/huggingface}' RCLONE_REMOTE='${RCLONE_REMOTE:-}' VAST_INSTANCE_ID='${VAST_INSTANCE_ID:-}' DESTROY_INSTEAD_OF_STOP='${DESTROY_INSTEAD_OF_STOP:-true}' FORK_REPO_URL='${FORK_REPO_URL:-}' FORK_COMMIT_SHA='${FORK_COMMIT_SHA:-}' $PYTHON_BIN app.py >> '$LOG_DIR/app.log' 2>&1"

echo "APP_LOG=$LOG_DIR/app.log"
echo "APP_PORT=$SERVER_PORT"
