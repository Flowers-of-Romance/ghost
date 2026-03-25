#!/usr/bin/env python3
"""
record_turn.py — 会話ターンをraw_turnsに自動保存 + 文脈検索

Claude Code の UserPromptSubmit hook として起動される。
1. ユーザーの全発言をリアルタイムで raw_turns テーブルに記録
2. 発言から関連記憶を自動検索し、stderrに出力（Selecting強化）
"""

import sys
import os
import io
import json
import re
import sqlite3
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


def _should_search(text):
    """検索すべき発言かどうか判定。短すぎる・コマンド・タグは除外。"""
    s = text.strip()
    if len(s) < 15:
        return False
    # コマンド・スラッシュコマンド・yes/no系
    if s.lower() in ("y", "n", "yes", "no", "ok", "はい", "いいえ"):
        return False
    if s.startswith(("/", "!", "```", "git ", "pip ", "npm ", "python ")):
        return False
    # XMLタグが大半を占める
    if s.startswith("<") and s.endswith(">"):
        return False
    stripped = re.sub(r'<[^>]+>', '', s).strip()
    if len(stripped) < 15:
        return False
    return True


def _context_search(prompt):
    """
    ユーザー発言からFTS5で関連記憶を検索し、stderrに出力。
    embeddingモデルは使わない（速度優先）。
    """
    DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))
    if not os.path.exists(DB_PATH):
        return

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        return

    try:
        from tokenizer import tokenize
        tokenized = tokenize(prompt)
        if not tokenized or not tokenized.strip():
            conn.close()
            return

        # 助詞・短すぎるトークンを除去してOR検索
        stop_words = {"の", "に", "は", "を", "が", "で", "と", "も", "へ", "から",
                      "まで", "より", "つい", "て", "た", "だ", "する", "れる",
                      "ない", "いる", "ある", "この", "その", "あの", "どの"}
        tokens = [t for t in tokenized.split() if t not in stop_words and len(t) >= 2]
        if not tokens:
            conn.close()
            return

        fts_query = " OR ".join(tokens)

        # FTS5検索（上位3件、rank + importance で並べる）
        rows = conn.execute(
            """SELECT m.id, m.content, m.keywords, m.emotions, m.importance, m.arousal,
                      rank as fts_rank
               FROM memories_fts f
               JOIN memories m ON f.memory_id = m.id
               WHERE f.memories_fts MATCH ? AND m.forgotten = 0
               ORDER BY (rank * -1.0 + m.importance * 0.5 + m.arousal * 0.3) DESC
               LIMIT 3""",
            (fts_query,)
        ).fetchall()

        if not rows:
            conn.close()
            return

        # stderrに出力（LLMが読む）
        print("[ghost] 関連記憶:", file=sys.stderr)
        for row in rows:
            content = row["content"][:80]
            if len(row["content"]) > 80:
                content += "..."
            print(f"  #{row['id']} 「{content}」", file=sys.stderr)

    except Exception:
        pass
    finally:
        conn.close()


def handle_user_prompt(hook_input):
    """UserPromptSubmit: ユーザー発言を保存 + 文脈検索。"""
    prompt = hook_input.get("prompt", "")
    if not prompt.strip():
        return

    save_turn(
        session_id=hook_input.get("session_id", "unknown"),
        role="user",
        content=prompt,
        cwd=hook_input.get("cwd", ""),
    )

    # 文脈検索（Selecting）
    if _should_search(prompt):
        try:
            _context_search(prompt)
        except Exception:
            pass  # 検索失敗は無視（保存が主務）


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
