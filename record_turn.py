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
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Windows cp932 対策
if sys.platform == "win32":
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# memory.py と同じディレクトリにいる前提
sys.path.insert(0, str(Path(__file__).parent))


def save_turn(session_id, role, content, cwd=""):
    """raw_turns に1ターン保存 + マークダウンに追記。"""
    from memory import save_raw_turn
    save_raw_turn(
        session_id=session_id,
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        cwd=cwd,
    )
    try:
        _append_to_markdown(session_id, role, content, cwd)
    except Exception as e:
        print(f"[turn_export] {e}", file=sys.stderr)


def _load_export_config():
    """turn_export.json を読み込む。無ければNone。"""
    config_path = Path(__file__).parent / "turn_export.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not config.get("enabled", False):
            return None
        return config
    except (json.JSONDecodeError, OSError):
        return None


def _locked_append(filepath, text):
    """ファイルロック付きappend。複数ウィンドウで同時書き込みしても安全。"""
    lock_path = Path(str(filepath) + ".lock")
    lock_fd = None
    try:
        # ロックファイルを排他的に取得（リトライ付き）
        for _ in range(10):
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                # ロック待ち（50ms × 10 = 最大500ms）
                time.sleep(0.05)
        else:
            # ロック取得失敗でも書き込みは試みる
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(text)
            return

        # ロック取得成功 → 書き込み
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                os.remove(str(lock_path))
            except OSError:
                pass


def _pick_user_face(content):
    """ユーザー発言の雰囲気から顔を選ぶ。"""
    c = (content or "").lower()
    # 怒り・不満
    if any(w in c for w in ["ふざけ", "おかしい", "バグ", "だめ", "ひどい", "最悪", "むかつ", "壊れ", "なんで"]):
        return "😤"
    # 疑問・困惑
    if any(w in c for w in ["？", "?", "わからん", "わからない", "なぜ", "どうして", "どういう"]):
        return "🤔"
    # 感謝・喜び
    if any(w in c for w in ["ありがと", "さんきゅ", "助かる", "最高", "いいね", "すごい", "やった", "完璧"]):
        return "😆"
    # 挨拶
    if any(w in c for w in ["おはよ", "こんにち", "こんばん", "おつかれ", "ただいま", "よろしく"]):
        return "😊"
    # 依頼・お願い
    if any(w in c for w in ["して", "やって", "頼む", "お願い", "変えて", "直して", "作って", "見せて"]):
        return "😙"
    # テンション高い
    if any(w in c for w in ["！", "!", "www", "笑", "ｗ", "草"]):
        return "😜"
    # 短い（雑な投げ）
    if len(c) < 10:
        return "🙂"
    # デフォルト
    return "😀"


def _is_system_noise(content):
    """システム通知など、ユーザー発言でないものを判定。"""
    if not content:
        return False
    return any(marker in content for marker in [
        "Background command",
        "toolu_",
        "completed (exit code",
        "Read the output file to retrieve the result",
    ])


def _append_to_markdown(session_id, role, content, cwd=""):
    """turn_export.json の設定に従い、日付マークダウンに追記。"""
    config = _load_export_config()
    if config is None:
        return

    output_dir = Path(os.path.expandvars(config["output_dir"])).expanduser()
    offset = config.get("timezone_offset_hours", 9)

    local = datetime.now(timezone.utc) + timedelta(hours=offset)
    date_str = local.strftime("%Y-%m-%d")
    time_str = local.strftime("%H:%M")
    md_path = output_dir / f"{date_str}.md"

    output_dir.mkdir(parents=True, exist_ok=True)

    if role == "user" and _is_system_noise(content):
        return
    # XMLタグだけ・空白だけの発言はスキップ
    stripped_content = re.sub(r'<[^>]+>', '', content or '').strip()
    if not stripped_content:
        return

    _SESSION_ANIMALS = [
        "🐱", "🐶", "🦊", "🐸", "🐙", "🦉", "🐻", "🐺",
        "🦈", "🐧", "🦎", "🐝", "🦋", "🐬", "🦅", "🐢",
    ]

    icon = f"{_pick_user_face(content)} User" if role == "user" else "🤖 Assistant"
    short_sid = session_id[:8]
    project_name = Path(cwd).name if cwd else ""
    # セッションIDから動物マーカーを決定（同じセッションは常に同じ動物）
    # hash()はプロセスごとにランダム化されるのでUUIDの先頭を16進数として使う
    animal = _SESSION_ANIMALS[int(short_sid, 16) % len(_SESSION_ANIMALS)]

    if not md_path.exists():
        # 新規作成（frontmatter + 最初のセッション見出し）
        text = f"---\ntitle: \"{date_str}\"\ntags: [claude-turns]\n---\n\n"
        label = f"## {animal} {time_str} [{project_name}] session:{short_sid}\n" if project_name else f"## {animal} {time_str} session:{short_sid}\n"
        text += label
        if cwd:
            text += f"> cwd: {cwd}\n"
        text += f"\n### {icon} {animal} {time_str}\n{content}\n"
        md_path.write_text(text, encoding="utf-8")
    else:
        # 追記
        existing = md_path.read_text(encoding="utf-8")
        entry = ""
        if f"session:{short_sid}" not in existing:
            label = f"\n---\n\n## {animal} {time_str} [{project_name}] session:{short_sid}\n" if project_name else f"\n---\n\n## {animal} {time_str} session:{short_sid}\n"
            entry += label
            if cwd:
                entry += f"> cwd: {cwd}\n"
        entry += f"\n### {icon} {animal} {time_str}\n{content}\n"
        _locked_append(md_path, entry)


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


def _extract_tool_calls(message):
    """assistant message から tool_use ブロックを抽出。[(name, input_dict), ...]"""
    if message.get("type") != "assistant":
        return []
    content_parts = message.get("message", {}).get("content", [])
    if not isinstance(content_parts, list):
        return []
    calls = []
    for part in content_parts:
        if isinstance(part, dict) and part.get("type") == "tool_use":
            calls.append((part.get("name", ""), part.get("input", {}), part.get("tool_use_id", "")))
    return calls


def _extract_tool_results(message):
    """user message から tool_result ブロックを抽出。{tool_use_id: content}"""
    if message.get("type") != "user":
        return {}
    content_parts = message.get("message", {}).get("content", [])
    if not isinstance(content_parts, list):
        return {}
    results = {}
    for part in content_parts:
        if isinstance(part, dict) and part.get("type") == "tool_result":
            tid = part.get("tool_use_id", "")
            content = part.get("content", "")
            if isinstance(content, list):
                # content が [{type: "text", text: "..."}] の場合
                content = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            results[tid] = str(content)
    return results


def _format_tool_for_md(name, input_dict, result_text):
    """tool call を MD 用に整形。"""
    MAX_RESULT = 500
    if name == "Bash":
        cmd = input_dict.get("command", "")
        # 改行を含むコマンドは1行に潰して80文字で切る
        cmd_oneline = cmd.replace("\n", " ").strip()
        if len(cmd_oneline) > 80:
            cmd_oneline = cmd_oneline[:77] + "..."
        truncated = result_text.strip()[:MAX_RESULT]
        if len(result_text) > MAX_RESULT:
            truncated += "\n… (truncated)"
        return f"- 🔧 `{cmd_oneline}`\n```\n{truncated}\n```\n"
    elif name == "Read":
        path = input_dict.get("file_path", "")
        return f"- 📖 Read `{path}`\n"
    elif name == "Edit":
        path = input_dict.get("file_path", "")
        return f"- ✏️ Edit `{path}`\n"
    elif name == "Write":
        path = input_dict.get("file_path", "")
        return f"- 📝 Write `{path}`\n"
    elif name in ("Glob", "Grep"):
        pattern = input_dict.get("pattern", "")
        return f"- 🔍 {name} `{pattern}`\n"
    else:
        return f"- 🔧 {name}\n"


def handle_stop(hook_input):
    """Stop: transcript_path から全アシスタント応答を保存。
    - SQLite: 最後のテキスト応答のみ
    - MD: テキスト + tool call を時系列で追記
    """
    transcript_path = hook_input.get("transcript_path", "")
    if not transcript_path or not Path(transcript_path).exists():
        return

    session_id = hook_input.get("session_id", "unknown")
    cwd = hook_input.get("cwd", "")

    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        return

    messages = []
    for line in lines:
        if not line.strip():
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # --- SQLite: 最後のアシスタントテキストのみ（従来通り） ---
    for msg in reversed(messages):
        text = extract_assistant_text(msg)
        if text and text.strip():
            save_turn(
                session_id=session_id,
                role="assistant",
                content=text,
                cwd=cwd,
            )
            break

    # --- MD: tool call を追記 ---
    config = _load_export_config()
    if config is None:
        return

    # tool_use_id → result のマップを構築
    result_map = {}
    for msg in messages:
        result_map.update(_extract_tool_results(msg))

    # assistant メッセージから tool call を収集して MD に追記
    md_parts = []
    for msg in messages:
        calls = _extract_tool_calls(msg)
        for name, input_dict, tid in calls:
            result_text = result_map.get(tid, "")
            md_parts.append(_format_tool_for_md(name, input_dict, result_text))

    if not md_parts:
        return

    output_dir = Path(os.path.expandvars(config["output_dir"])).expanduser()
    offset = config.get("timezone_offset_hours", 9)
    local = datetime.now(timezone.utc) + timedelta(hours=offset)
    date_str = local.strftime("%Y-%m-%d")
    md_path = output_dir / f"{date_str}.md"

    if md_path.exists():
        callout_lines = ["> [!info]- 🔧 Tool calls"]
        for part in md_parts:
            for line in part.rstrip("\n").split("\n"):
                callout_lines.append(f"> {line}")
        _locked_append(md_path, "\n" + "\n".join(callout_lines) + "\n")


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
