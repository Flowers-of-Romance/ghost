#!/usr/bin/env python3
"""
memory_sync_server.py - P2P記憶同期サーバー

起動:
  python memory_sync_server.py [--port 7235] [--token SECRET] [--public] [--insecure]

  # 既定はローカルのみ (127.0.0.1)
  # 外部公開は --public を明示
  # 認証必須。無認証を許可するなら --insecure を明示

別の端末から:
  python memory.py sync push host:7235
  python memory.py sync pull host:7235
"""

import json
import sys
import io
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# memory.pyからインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import sync_export, sync_import, _get_node_id

SYNC_PORT = 7235
SYNC_TOKEN = os.environ.get("MEMORY_SYNC_TOKEN", "")
ALLOW_INSECURE = False
MAX_BODY_BYTES = 5 * 1024 * 1024  # 5MB


class SyncHandler(BaseHTTPRequestHandler):
    def _check_auth(self):
        if not SYNC_TOKEN:
            if ALLOW_INSECURE:
                return True
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorized")
            return False
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {SYNC_TOKEN}":
            return True
        self.send_response(401)
        self.end_headers()
        self.wfile.write(b"unauthorized")
        return False

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/sync/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if not self._check_auth():
            return

        if self.path.startswith("/sync/changes"):
            # GET /sync/changes?since=ISO_TIMESTAMP
            since = None
            if "?" in self.path:
                params = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&") if "=" in p)
                since = params.get("since")
            data = sync_export(since=since)
            self._send_json(data)
            return

        if self.path == "/sync/node-id":
            self._send_json({"node_id": _get_node_id()})
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if not self._check_auth():
            return

        if self.path == "/sync/push":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_response(413)
                self.end_headers()
                return
            try:
                raw = self.rfile.read(length)
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("payload must be object")
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            stats = sync_import(body)
            self._send_json({"status": "ok", "stats": stats})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # 同期ログだけ出す
        if "/sync/" in (args[0] if args else ""):
            print(f"  sync: {args[0]}")


if __name__ == "__main__":
    port = SYNC_PORT
    bind_host = "127.0.0.1"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])
    if "--token" in sys.argv:
        idx = sys.argv.index("--token")
        SYNC_TOKEN = sys.argv[idx + 1]
    if "--public" in sys.argv:
        bind_host = "0.0.0.0"
    if "--insecure" in sys.argv:
        ALLOW_INSECURE = True

    if not SYNC_TOKEN and not ALLOW_INSECURE:
        print("error: SYNC_TOKEN is required (set MEMORY_SYNC_TOKEN or use --token).")
        print("       If you really want no auth, pass --insecure explicitly.")
        sys.exit(1)

    node_id = _get_node_id()
    print(f"node_id: {node_id}")
    print(f"sync server: http://{bind_host}:{port}")
    if SYNC_TOKEN:
        print(f"token: {SYNC_TOKEN[:4]}...")
    else:
        print("token: なし（--insecure モード）")

    server = HTTPServer((bind_host, port), SyncHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバー停止")
