#!/usr/bin/env python3
"""
local_chat.py — OpenAI互換APIラッパー（Ollama, LM Studio, llama.cpp）

会話をGHOSTのraw_turns + mdに記録する。

Usage:
    python local_chat.py "こんにちは"
    python local_chat.py "hello" --model gemma2
    python local_chat.py "hello" --url http://localhost:1234/v1 --model local-model
"""

import argparse
import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from record_turn import save_turn

DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _detect_source(base_url):
    if "11434" in base_url:
        return "ollama"
    if "1234" in base_url:
        return "lmstudio"
    if "8080" in base_url:
        return "llama.cpp"
    return "local"


def chat(model, messages, base_url=DEFAULT_BASE_URL, session_id=None):
    """OpenAI互換APIでチャットし、raw_turns + mdに記録。"""
    if session_id is None:
        session_id = uuid.uuid4().hex[:16]

    source = _detect_source(base_url)

    # ユーザー発言を記録
    user_msg = messages[-1].get("content", "") if messages else ""
    if user_msg:
        save_turn(session_id, "user", user_msg, cwd=f"{source}:{model}")

    # API呼び出し
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        print(f"[local_chat] API error: {e}", file=sys.stderr)
        raise

    reply = data["choices"][0]["message"]["content"]

    # アシスタント応答を記録
    save_turn(session_id, "assistant", reply, cwd=f"{source}:{model}")

    return reply


def main():
    parser = argparse.ArgumentParser(description="ローカルLLMにチャットしてGHOSTに記録")
    parser.add_argument("prompt", help="質問")
    parser.add_argument("--model", default="gemma2", help="モデル名 (default: gemma2)")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--session", default=None, help="セッションID")
    args = parser.parse_args()

    reply = chat(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        base_url=args.url,
        session_id=args.session,
    )
    print(reply)


if __name__ == "__main__":
    main()
