#!/usr/bin/env python3
"""
wander.py - 目的なき連想（DMN agent）

記憶の断片をLLMに渡して、自由連想させる。
何が出てくるか見る。出てこなくてもいい。

LLMバックエンド（優先順）:
  1. Gemini API (GEMINI_API_KEY環境変数)
  2. ローカル llama-cpp + GGUF

使い方:
  python wander.py           # 3回ぼんやりする
  python wander.py N         # N回
  python wander.py --dry-run # 保存せず表示
"""

import sys
import os
import io
import json
import random
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from memory import (
    get_connection, embed_text, vec_to_bytes, bytes_to_vec,
    cosine_similarity, DB_PATH,
)

# --- 設定 ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("WANDER_GEMINI_MODEL", "gemini-3-flash-preview")

WANDER_MODEL = os.environ.get(
    "WANDER_MODEL",
    str(Path(__file__).parent / "models" / "Qwen3-0.6B-Q8_0.gguf"),
)
LLAMA_CLI = os.environ.get(
    "LLAMA_CLI",
    str(Path(__file__).parent / "models" / "llama-cpp" / "llama-completion.exe"),
)

RESONANCE_THRESHOLD = 0.70

SYSTEM_PROMPT = (
    "以下の断片から自由に連想して、日本語で1〜3文だけ書いてください。\n"
    "ルール: 日本語のみ。思考過程・英語・中国語は禁止。連想だけ出力。"
)


# --- LLMチェック ---

def _check_gemini():
    return bool(GEMINI_API_KEY)

def _check_local():
    return Path(WANDER_MODEL).exists() and Path(LLAMA_CLI).exists()

def _check_llm():
    return _check_gemini() or _check_local()


# --- 記憶の選択 ---

def _pick_memories(conn, n=3):
    """3方式からランダムに1つ選び、記憶を2-3件取得。"""
    method = random.choice(["random", "weak_link", "time_cluster"])

    if method == "random":
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE forgotten = 0 "
            "ORDER BY RANDOM() LIMIT ?", (n,)
        ).fetchall()
        return rows

    if method == "weak_link":
        start = conn.execute(
            "SELECT id FROM memories WHERE forgotten = 0 ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if not start:
            return []
        collected_ids = [start["id"]]
        current_id = start["id"]
        for _ in range(n - 1):
            neighbor = conn.execute(
                "SELECT target_id FROM links WHERE source_id = ? "
                "ORDER BY strength ASC LIMIT 1",
                (current_id,)
            ).fetchone()
            if neighbor and neighbor["target_id"] not in collected_ids:
                collected_ids.append(neighbor["target_id"])
                current_id = neighbor["target_id"]
            else:
                fallback = conn.execute(
                    "SELECT id FROM memories WHERE forgotten = 0 AND id NOT IN ({}) "
                    "ORDER BY RANDOM() LIMIT 1".format(
                        ",".join("?" * len(collected_ids))
                    ), collected_ids
                ).fetchone()
                if fallback:
                    collected_ids.append(fallback["id"])
                    current_id = fallback["id"]
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE id IN ({})".format(
                ",".join("?" * len(collected_ids))
            ), collected_ids
        ).fetchall()
        return rows

    if method == "time_cluster":
        pivot = conn.execute(
            "SELECT created_at FROM memories WHERE forgotten = 0 "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if not pivot:
            return []
        pivot_date = pivot["created_at"][:10]
        rows = conn.execute(
            "SELECT id, content FROM memories "
            "WHERE forgotten = 0 AND created_at LIKE ? "
            "ORDER BY RANDOM() LIMIT ?",
            (pivot_date + "%", n)
        ).fetchall()
        return rows

    return []


# --- LLM呼び出し ---

def _has_japanese(text):
    """ひらがな・カタカナを含むかチェック（漢字だけの中国語を除外）。"""
    return any(
        '\u3040' <= c <= '\u309f' or  # ひらがな
        '\u30a0' <= c <= '\u30ff'     # カタカナ
        for c in text
    )


def _ask_gemini(fragments):
    """Gemini REST APIで自由連想。"""
    # 日本語を含む記憶のみ使う（ノイズ記憶を除外）
    clean = [f for f in fragments if _has_japanese(f["content"])]
    if len(clean) < 2:
        clean = fragments  # フォールバック
    user_text = "\n---\n".join(f["content"][:200] for f in clean)
    prompt = f"{SYSTEM_PROMPT}\n\n{user_text}"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 400,
        },
    }).encode("utf-8")

    req = Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # ノイズ除去
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # ローマ字行やカッコ付き英訳を除去
            if stripped.startswith("(") and stripped.endswith(")"):
                continue
            # 思考過程リーク除去
            if any(stripped.startswith(p) for p in (
                "Let me", "I should", "I need", "The user",
                "First,", "OK,", "Okay,", "Maybe", "Also,",
                "Then,", "嗯，", "好的，", "首先，",
            )):
                continue
            # ひらがな・カタカナを含む行のみ残す（中国語除外）
            if _has_japanese(stripped):
                cleaned.append(stripped)

        return "\n".join(cleaned) if cleaned else ""
    except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError):
        return ""


def _ask_local(fragments):
    """llama-completionでローカルLLMに自由連想させる。"""
    user_text = "\n---\n".join(f["content"][:200] for f in fragments)

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        result = subprocess.run(
            [
                LLAMA_CLI,
                "-m", WANDER_MODEL,
                "-p", prompt,
                "-n", "150",
                "--temp", "1.0",
                "--no-display-prompt",
                "-c", "512",
                "--logit-bias", "151667-inf",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        text = result.stdout.strip()

        for marker in ("<|im_end|>", "EOF by user", "<|im_start|>", "</think>"):
            if marker in text:
                text = text[:text.index(marker)]
        if "<think>" in text:
            text = text[:text.index("<think>")]
        text = text.rstrip("> \n").strip()

        if text:
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            if len(sentences) > 2:
                text = ". ".join(sentences[-2:]) + "."

        return text
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def _ask_llm(fragments):
    """Gemini API優先、フォールバックでローカルllama-cpp。"""
    if _check_gemini():
        # Geminiが日本語を返せなかった場合、ローカルに落とさずスキップ
        return _ask_gemini(fragments)
    if _check_local():
        return _ask_local(fragments)
    return ""


def _validate_output(text):
    """LLM出力のノイズを検出して除去する。"""
    if len(text.strip()) < 8:
        return ""
    from collections import Counter
    counts = Counter(text)
    if counts.most_common(1)[0][1] / len(text) > 0.5:
        return ""
    if sum(1 for c in text if c.isalpha()) < len(text) * 0.3:
        return ""
    return text


# --- 共鳴・保存 ---

def _find_resonance(conn, vec, exclude_ids):
    """既存記憶との共鳴を探す。0.70以上で最も近い1件を返す。"""
    rows = conn.execute(
        "SELECT id, content, embedding FROM memories "
        "WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    best_id = None
    best_sim = 0.0
    best_content = ""

    for row in rows:
        if row["id"] in exclude_ids:
            continue
        mem_vec = bytes_to_vec(row["embedding"])
        sim = cosine_similarity(vec, mem_vec)
        if sim >= RESONANCE_THRESHOLD and sim > best_sim:
            best_id = row["id"]
            best_sim = sim
            best_content = row["content"]

    if best_id is not None:
        return {"id": best_id, "sim": best_sim, "content": best_content}
    return None


def _store(conn, text, vec, source_ids, resonant):
    """連想を記憶として保存し、共鳴先とリンクを作る。"""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    source_tag = "wander:" + ",".join(str(i) for i in source_ids)

    cursor = conn.execute(
        "INSERT INTO memories (content, category, importance, arousal, "
        "source_conversation, embedding, created_at, updated_at, "
        "provenance, confidence) "
        "VALUES (?, 'episode', 2, 0.15, ?, ?, ?, ?, 'wander', 0.3)",
        (text, source_tag, vec_to_bytes(vec), now, now)
    )
    new_id = cursor.lastrowid

    for src, tgt in [(new_id, resonant["id"]), (resonant["id"], new_id)]:
        conn.execute(
            "INSERT OR IGNORE INTO links (source_id, target_id, strength, created_at) "
            "VALUES (?, ?, ?, ?)",
            (src, tgt, resonant["sim"], now)
        )

    conn.commit()
    return new_id


# --- メインループ ---

def wander_once(conn, dry_run=False):
    """1回のぼんやり。共鳴があれば保存して返す。"""
    fragments = _pick_memories(conn, n=random.choice([2, 3]))
    if len(fragments) < 2:
        return None

    source_ids = [f["id"] for f in fragments]
    response = _ask_llm(fragments)
    response = _validate_output(response)

    if not response:
        return None

    vec = embed_text(response)
    if vec is None:
        return None

    resonant = _find_resonance(conn, vec, set(source_ids))

    if resonant is None:
        if dry_run:
            print(f"  \"{response}\" → (共鳴なし)")
        return None

    if dry_run:
        print(f"  \"{response}\" → #{resonant['id']} (sim:{resonant['sim']:.3f})")
        return {"text": response, "resonant": resonant}

    new_id = _store(conn, response, vec, source_ids, resonant)
    print(f"  \"{response}\" → #{resonant['id']}")
    return {"id": new_id, "text": response, "resonant": resonant}


def wander(n=3, dry_run=False):
    """メインループ。"""
    if not _check_llm():
        return 0

    conn = get_connection()
    saved = 0

    for _ in range(n):
        result = wander_once(conn, dry_run=dry_run)
        if result and not dry_run:
            saved += 1

    conn.close()

    if saved > 0:
        print(f"(...{saved}件の残響)")
    return saved


def cleanup_noise(conn):
    """既存のwanderノイズ記憶をforgotten=1にする。"""
    # パターンベースのノイズ除去
    cursor = conn.execute("""
        UPDATE memories SET forgotten = 1
        WHERE source_conversation LIKE 'wander:%'
        AND forgotten = 0
        AND (LENGTH(content) < 8
             OR content GLOB '*))))*'
             OR content GLOB '*<think>*')
    """)
    pattern_count = cursor.rowcount

    # 日本語を含まないwander記憶も除去（中国語思考ダンプ等）
    rows = conn.execute(
        "SELECT id, content FROM memories "
        "WHERE source_conversation LIKE 'wander:%' AND forgotten = 0"
    ).fetchall()
    lang_count = 0
    for row in rows:
        if not _has_japanese(row["content"]):
            conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?", (row["id"],))
            lang_count += 1

    total = pattern_count + lang_count
    print(f"✓ {total}件のノイズ記憶を忘却しました (パターン:{pattern_count}, 非日本語:{lang_count})")
    conn.commit()


if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        conn = get_connection()
        cleanup_noise(conn)
        conn.close()
    else:
        args = [a for a in sys.argv[1:] if a != "--dry-run"]
        dry_run = "--dry-run" in sys.argv
        n = int(args[0]) if args else 3
        wander(n=n, dry_run=dry_run)
