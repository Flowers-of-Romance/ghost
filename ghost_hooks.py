#!/usr/bin/env python3
"""
ghost_hooks.py — 能動的な記憶監視

Claude Code の PostToolUse hook として起動される。
Edit/Write が実行されるたびに、memory.db の plan 記憶と prospective を
読み取り専用で照合し、関連する記憶を stderr に表示する。

判断はしない。思い出させるだけ。判断は LLM がやる。
"""

import sys
import os
import io
import json
import sqlite3
import struct
import tempfile
import time
from pathlib import Path

# Windows cp932 対策
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))
COOLDOWN_DIR = Path(tempfile.gettempdir()) / "ghost_hooks"
COOLDOWN_SECONDS = 300  # 同一plan を5分間再警告しない
IMPORTANCE_MIN = 3      # importance >= 3 のplanのみ
EMBED_THRESHOLD = 0.45  # embedding類似度の閾値

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


def get_session_id():
    """セッション識別。同じ日・同じディレクトリなら同一セッション扱い。"""
    sid = os.environ.get("CLAUDE_SESSION_ID")
    if sid:
        return sid
    import hashlib
    key = f"{time.strftime('%Y%m%d')}_{os.getcwd()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def is_cooled_down(plan_id):
    """同一セッション内で同じ plan を最近警告したか。"""
    COOLDOWN_DIR.mkdir(exist_ok=True)
    marker = COOLDOWN_DIR / f"{get_session_id()}_{plan_id}"
    if marker.exists():
        age = time.time() - marker.stat().st_mtime
        if age < COOLDOWN_SECONDS:
            return True
    return False


def mark_warned(plan_id):
    """警告済みマーカーを記録。"""
    COOLDOWN_DIR.mkdir(exist_ok=True)
    marker = COOLDOWN_DIR / f"{get_session_id()}_{plan_id}"
    marker.touch()


def load_plans(conn):
    """重要度が閾値以上の plan 記憶を取得。"""
    rows = conn.execute(
        "SELECT id, content, keywords, embedding FROM memories "
        "WHERE category = 'plan' AND forgotten = 0 AND importance >= ?",
        (IMPORTANCE_MIN,)
    ).fetchall()
    plans = []
    for r in rows:
        plans.append({
            "id": r[0],
            "content": r[1],
            "keywords": json.loads(r[2]) if r[2] else [],
            "embedding": r[3],
        })
    return plans


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


def check_keywords(plans, text):
    """plan の keywords とテキストのキーワードマッチ。"""
    text_lower = text.lower()
    matched = []
    for plan in plans:
        for kw in plan["keywords"]:
            if kw.lower() in text_lower:
                matched.append(plan)
                break
    return matched


def _embed_via_server(text, is_query=False):
    """サーバー経由でembedding取得。モデルロード不要で高速。"""
    import urllib.request
    payload = json.dumps({"text": text, "is_query": is_query}).encode()
    req = urllib.request.Request(
        "http://localhost:52836/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        return json.loads(resp.read())


def _bytes_to_vec(b):
    n = len(b) // 4
    return struct.unpack(f'{n}f', b)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def check_embedding(plans, text):
    """embedding 類似度で関連 plan を検索。サーバー経由のみ（高速）。"""
    try:
        query_vec = _embed_via_server(text, is_query=True)
    except Exception:
        return []

    matched = []
    for plan in plans:
        if not plan["embedding"]:
            continue
        plan_vec = _bytes_to_vec(plan["embedding"])
        sim = _dot(query_vec, plan_vec)
        if sim >= EMBED_THRESHOLD:
            matched.append((plan, sim))

    matched.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matched[:5]]


def check_prospective_readonly(prospectives, text):
    """prospective のトリガーマッチ。発火フラグは更新しない。"""
    text_lower = text.lower()
    matched = []
    for p in prospectives:
        if p["trigger"].lower() in text_lower:
            matched.append(p)
    return matched


def format_warning(plan):
    """plan 記憶の警告行。"""
    content = plan["content"]
    # 最初の行 or 最初の80文字
    first_line = content.split("\n")[0][:80]
    return f"[ghost] plan: {first_line} (#{plan['id']})"


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
        # plan 検証
        plans = load_plans(conn)
        if plans:
            # v1: キーワードマッチ
            kw_matched = check_keywords(plans, text)

            # v2: embedding（失敗したら空）
            embed_matched = check_embedding(plans, text)

            # 統合（重複排除）
            seen_ids = set()
            matched = []
            for p in kw_matched + embed_matched:
                if p["id"] not in seen_ids:
                    seen_ids.add(p["id"])
                    matched.append(p)

            # クールダウン適用 & 警告生成
            for plan in matched:
                if not is_cooled_down(plan["id"]):
                    warnings.append(format_warning(plan))
                    mark_warned(plan["id"])

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
