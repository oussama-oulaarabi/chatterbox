#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_ROOT="${APP_ROOT:-/workspace/chbx_ja_batch}"
FORK_REPO_URL="${FORK_REPO_URL:?FORK_REPO_URL is required}"
FORK_COMMIT_SHA="${FORK_COMMIT_SHA:?FORK_COMMIT_SHA is required}"
HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-}"
HF_MODEL_REVISION="${HF_MODEL_REVISION:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/workspace/.cache/huggingface}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/usr/local/bin/cloudflared}"

mkdir -p "$APP_ROOT/runtime_logs"
exec > >(tee -a "$APP_ROOT/runtime_logs/setup_remote.log") 2>&1

echo "[setup] starting"
echo "[setup] APP_ROOT=$APP_ROOT"
echo "[setup] FORK_REPO_URL=$FORK_REPO_URL"
echo "[setup] FORK_COMMIT_SHA=$FORK_COMMIT_SHA"
echo "[setup] HF_MODEL_REPO_ID=$HF_MODEL_REPO_ID"
echo "[setup] HF_MODEL_REVISION=$HF_MODEL_REVISION"

apt-get update
apt-get install -y git git-lfs ffmpeg curl ca-certificates tmux build-essential rclone

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "$APP_ROOT/requirements.lock.txt"

if [ ! -d "$APP_ROOT/src/.git" ]; then
  git clone "$FORK_REPO_URL" "$APP_ROOT/src"
fi

cd "$APP_ROOT/src"
git fetch --all --tags
git checkout "$FORK_COMMIT_SHA"
git rev-parse HEAD
python3 -m pip install -e .

python3 -m pip freeze | tail -n 250 > "$APP_ROOT/runtime_logs/pip_freeze_after_setup.log" || true
rclone version > "$APP_ROOT/runtime_logs/rclone_version.log" 2>&1 || true
nvidia-smi > "$APP_ROOT/runtime_logs/nvidia_smi_setup.log" 2>&1 || true

if [ ! -x "$CLOUDFLARED_BIN" ]; then
  ARCH="$(dpkg --print-architecture)"
  case "$ARCH" in
    amd64) CF_PKG="cloudflared-linux-amd64" ;;
    arm64) CF_PKG="cloudflared-linux-arm64" ;;
    *) echo "Unsupported arch for cloudflared: $ARCH"; exit 1 ;;
  esac
  curl -L "https://github.com/cloudflare/cloudflared/releases/latest/download/${CF_PKG}" -o "$CLOUDFLARED_BIN"
  chmod +x "$CLOUDFLARED_BIN"
fi

echo "[setup] complete"
