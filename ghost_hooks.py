#!/usr/bin/env python3
"""
ghost_hooks.py — 能動的な記憶監視

Claude Code の PostToolUse hook として起動される。
Edit/Write が実行されるたびに、memory.db の prospective を
読み取り専用で照合し、関連する記憶を stderr に表示する。

判断はしない。思い出させるだけ。判断は LLM がやる。
"""

import sys
import os
import io
import json
import sqlite3
import tempfile
import time
from pathlib import Path

# Windows cp932 対策
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))
COOLDOWN_DIR = Path(tempfile.gettempdir()) / "ghost_hooks"

LAST_ACTIVITY_FILE = COOLDOWN_DIR / "last_activity"
NAP_IDLE_SECONDS = 1800  # 30分


def check_and_nap():
    """前回のツール使用から30分以上経っていたらnap実行。"""
    COOLDOWN_DIR.mkdir(exist_ok=True)
    now = time.time()

    if LAST_ACTIVITY_FILE.exists():
        try:
            last = float(LAST_ACTIVITY_FILE.read_text().strip())
            gap = now - last
            if gap >= NAP_IDLE_SECONDS:
                import subprocess
                print("(うとうと...)", file=sys.stderr)
                subprocess.run(
                    ["python", "memory.py", "nap"],
                    cwd=str(Path(__file__).parent),
                    capture_output=True, timeout=30
                )
        except (ValueError, subprocess.TimeoutExpired):
            pass

    # タイムスタンプを更新（nap実行後も含む）
    LAST_ACTIVITY_FILE.write_text(str(now))



def load_prospective(conn):
    """未発火の prospective を取得。"""
    rows = conn.execute(
        "SELECT id, trigger_pattern, action FROM prospective WHERE fired = 0"
    ).fetchall()
    return [{"id": r[0], "trigger": r[1], "action": r[2]} for r in rows]


def extract_edit_text(hook_input):
    """hook の stdin から検索対象テキストを組み立てる。"""
    tool_input = hook_input.get("tool_input", {})
    parts = []
    # ファイルパス
    fp = tool_input.get("file_path", "")
    if fp:
        parts.append(fp)
    # 新しい内容 (Edit の new_string, Write の content)
    for key in ("new_string", "content"):
        v = tool_input.get(key, "")
        if v:
            parts.append(v)
    return "\n".join(parts)


def check_prospective_readonly(prospectives, text):
    """prospective のトリガーマッチ。発火フラグは更新しない。"""
    text_lower = text.lower()
    matched = []
    for p in prospectives:
        if p["trigger"].lower() in text_lower:
            matched.append(p)
    return matched


def format_prospective_warning(p):
    """prospective の警告行。"""
    return f"[ghost] prospective: {p['action']} (trigger: {p['trigger']})"


def main():
    # 30分以上のアイドル検知 → 自動nap
    check_and_nap()

    # stdin から hook データを読む
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            json.dump({"decision": "approve"}, sys.stdout)
            return
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        json.dump({"decision": "approve"}, sys.stdout)
        return

    # Edit/Write 以外は無視
    tool_name = hook_input.get("tool_name", "")
    if tool_name not in ("Edit", "Write"):
        json.dump({"decision": "approve"}, sys.stdout)
        return

    # 編集テキストを抽出
    text = extract_edit_text(hook_input)
    if not text.strip():
        json.dump({"decision": "approve"}, sys.stdout)
        return

    # memory.db を読み取り専用で開く
    if not os.path.exists(DB_PATH):
        json.dump({"decision": "approve"}, sys.stdout)
        return

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        json.dump({"decision": "approve"}, sys.stdout)
        return

    warnings = []

    try:
        # prospective 検証
        prospectives = load_prospective(conn)
        if prospectives:
            p_matched = check_prospective_readonly(prospectives, text)
            for p in p_matched:
                warnings.append(format_prospective_warning(p))

    finally:
        conn.close()

    # stderr に警告出力（LLM が読む）
    if warnings:
        for w in warnings:
            print(w, file=sys.stderr)

    # stdout に approve（常に許可）
    json.dump({"decision": "approve"}, sys.stdout)


if __name__ == "__main__":
    main()
