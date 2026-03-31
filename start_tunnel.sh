#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="${APP_ROOT:-/workspace/chbx_ja_batch}"
SERVER_PORT="${SERVER_PORT:-7860}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-/usr/local/bin/cloudflared}"
LOG_DIR="$APP_ROOT/runtime_logs"
mkdir -p "$LOG_DIR"

echo "[tunnel] $(date -u +%FT%TZ) starting tunnel for port $SERVER_PORT" >> "$LOG_DIR/cloudflared_control.log"
rm -f "$LOG_DIR/cloudflared.log"
nohup "$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${SERVER_PORT}" > "$LOG_DIR/cloudflared.log" 2>&1 &
sleep 5

python3 - <<'PY'
import pathlib, re, time
p = pathlib.Path("/workspace/chbx_ja_batch/runtime_logs/cloudflared.log")
for _ in range(30):
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"https://[a-zA-Z0-9.-]+trycloudflare\.com", txt)
        if m:
            print(m.group(0))
            raise SystemExit(0)
    time.sleep(2)
raise SystemExit("No trycloudflare URL found yet.")
PY
