#!/usr/bin/env python3
"""
memory_server.py - embeddingモデル常駐サーバー

起動:
  python memory_server.py [--port 7234]

モデルを一度だけロードしてHTTPで提供する。
memory.pyはこのサーバーにembeddingを問い合わせる。
"""

import json
import sys
import io
from http.server import HTTPServer, BaseHTTPRequestHandler

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PORT = 7234

print(f"モデル読み込み中: {EMBEDDING_MODEL} ...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(EMBEDDING_MODEL)
print("✓ モデル準備完了")


class EmbedHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/embed":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            text = body["text"]
            is_query = body.get("is_query", False)
            prefix = "query: " if is_query else "passage: "
            vec = model.encode(prefix + text, normalize_embeddings=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(vec.tolist()).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # リクエストログを抑制
        pass


if __name__ == "__main__":
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    server = HTTPServer(("127.0.0.1", port), EmbedHandler)
    print(f"✓ サーバー起動: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバー停止")
