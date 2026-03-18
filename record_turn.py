#!/usr/bin/env python3
"""
record_turn.py — 会話ターンをraw_turnsに自動保存

Claude Code の UserPromptSubmit hook として起動される。
ユーザーの全発言をリアルタイムで raw_turns テーブルに記録する。
これにより delusion（完全記憶モード）で全会話を検索可能にする。
"""

import sys
import os
import io
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Windows cp932 対策
if sys.platform == "win32":
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# memory.py と同じディレクトリにいる前提
sys.path.insert(0, str(Path(__file__).parent))


def save_turn(session_id, role, content, cwd=""):
    """raw_turns に1ターン保存。"""
    from memory import save_raw_turn
    save_raw_turn(
        session_id=session_id,
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        cwd=cwd,
    )


def handle_user_prompt(hook_input):
    """UserPromptSubmit: ユーザー発言を保存。"""
    prompt = hook_input.get("prompt", "")
    if not prompt.strip():
        return

    save_turn(
        session_id=hook_input.get("session_id", "unknown"),
        role="user",
        content=prompt,
        cwd=hook_input.get("cwd", ""),
    )


def extract_assistant_text(message):
    """transcript JSONL の1行からアシスタントのテキストを抽出。"""
    if message.get("type") == "assistant":
        # message.message.content は配列の場合がある
        msg = message.get("message", {})
        content_parts = msg.get("content", [])
        if isinstance(content_parts, str):
            return content_parts
        texts = []
        for part in content_parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts) if texts else None
    return None


def handle_stop(hook_input):
    """Stop: transcript_path から最後のアシスタント応答を保存。"""
    transcript_path = hook_input.get("transcript_path", "")
    if not transcript_path or not Path(transcript_path).exists():
        return

    session_id = hook_input.get("session_id", "unknown")
    cwd = hook_input.get("cwd", "")

    # JSONL を末尾から読んで最後のアシスタントメッセージを探す
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        return

    # 末尾から探す（最後のアシスタント応答が欲しい）
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = extract_assistant_text(message)
        if text and text.strip():
            save_turn(
                session_id=session_id,
                role="assistant",
                content=text,
                cwd=cwd,
            )
            return


def main():
    # stdin から hook データを読む
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return

    event = hook_input.get("hook_event_name", "")

    try:
        if event == "UserPromptSubmit":
            handle_user_prompt(hook_input)
        elif event == "Stop":
            handle_stop(hook_input)
    except Exception as e:
        print(f"record_turn error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
