#!/usr/bin/env python3
"""
ingest_chat.py - claude.ai コピペ会話専用パーサー & 取り込み

Extract.py が Claude Code の JSONL を扱うのに対し、
こちらは claude.ai の Web UI からコピペされた会話テキストを扱う。

使い方:
  # テキストファイルから記憶を抽出
  python ingest_chat.py conversation.txt [--dry-run]

  # 複数ファイル
  python ingest_chat.py chat1.txt chat2.txt --dry-run

  # JSONL 内の claude.ai 会話を自動検出
  python ingest_chat.py --detect session.jsonl

  # 標準入力からパイプ
  cat conversation.txt | python ingest_chat.py --stdin
"""

import json
import sys
import os
import io
import re
from pathlib import Path
from datetime import datetime, timezone

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from memory import (
    init_db, add_memory, get_connection,
    detect_emotions, embed_text, bytes_to_vec,
    cosine_similarity, get_model, DB_PATH
)


# --- 設定 ---
DUPLICATE_THRESHOLD = 0.85
MIN_AROUSAL = 0.10  # chat は緩め


# --- パーサー ---

# タイムスタンプの正規表現バリエーション
_TS_PATTERNS = [
    re.compile(r'^\d{1,2}:\d{2}$'),                    # 8:40
    re.compile(r'^\d{1,2}:\d{2}:\d{2}$'),              # 8:40:12
    re.compile(r'^\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}$'), # 3/25 8:40
    re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$'), # 2026-03-25 08:40
]

# システムマーカー（claude.ai UI 由来）
_SYSTEM_MARKERS = [
    "ウェブを検索しました",
    "ファイルを作成しました",
    "コマンドを実行しました",
    "ファイルを読み取りました",
    "ファイルを表示しました",
    "コード · ",
    "ドキュメント · ",
    "もっと表示",
    "Searched the web",
    "Created a file",
    "Ran command",
    "Read file",
    "Viewed file",
    "Code · ",
    "Document · ",
    "Show more",
]

# UI ラベル / ファイル名っぽい行を判定
_LABEL_RE = re.compile(
    r'^(Memory|Readme|Claude|PY|Extract|コード\s*·|ドキュメント\s*·|'
    r'[A-Z]:\\.*|Q:|A:|もっと表示|Show more)$',
    re.IGNORECASE,
)


def _is_timestamp(line):
    """行がタイムスタンプかどうか判定する。"""
    s = line.strip()
    return any(p.match(s) for p in _TS_PATTERNS)


def _is_system_marker(line):
    """行がシステムマーカーかどうか判定する。"""
    s = line.strip()
    return any(m in s for m in _SYSTEM_MARKERS)


def _is_role_header(line):
    """
    "あなた" / "Claude" のようなロールヘッダーを検出。
    タイムスタンプがないコピペで話者交代を識別する。
    """
    s = line.strip()
    # 完全一致系
    if s in ("あなた", "You", "Claude", "Human", "Assistant"):
        return s
    # "Claude said:" 等
    m = re.match(r'^(You|あなた|Human|Claude|Assistant)\s*[:：]\s*$', s)
    if m:
        return m.group(1)
    return None


def _role_from_header(header):
    """ヘッダー文字列を user/assistant に正規化。"""
    h = header.strip().rstrip(':：').strip()
    if h in ("あなた", "You", "Human"):
        return "user"
    if h in ("Claude", "Assistant"):
        return "assistant"
    return None


def parse_chat_text(text):
    """
    claude.ai からコピペされた会話テキストをパースする。

    対応フォーマット:
    1. タイムスタンプあり — H:MM が単独行に出現
       ユーザー発言 → タイムスタンプ → Claude応答 の繰り返し
    2. タイムスタンプなし — ロールヘッダー ("あなた" / "Claude") で分割
    3. どちらもなし — 空行ベースのヒューリスティック（短い=user, 長い=assistant）
    """
    lines = text.split('\n')

    # --- 戦略1: タイムスタンプベース ---
    ts_indices = [i for i, line in enumerate(lines) if _is_timestamp(line)]
    if ts_indices:
        return _parse_with_timestamps(lines, ts_indices)

    # --- 戦略2: ロールヘッダーベース ---
    header_indices = []
    for i, line in enumerate(lines):
        h = _is_role_header(line)
        if h:
            header_indices.append((i, h))
    if header_indices:
        return _parse_with_headers(lines, header_indices)

    # --- 戦略3: 空行ベースのヒューリスティック ---
    return _parse_heuristic(lines)


def _parse_with_timestamps(lines, ts_indices):
    """タイムスタンプを使った会話パース。"""
    turns = []

    for idx, ts_i in enumerate(ts_indices):
        timestamp = lines[ts_i].strip()

        # 前の区間の終端
        if idx == 0:
            prev_end = 0
        else:
            prev_end = ts_indices[idx - 1] + 1

        # ユーザー発言: タイムスタンプ直前から逆スキャン
        user_lines = []
        hit_text = False
        for j in range(ts_i - 1, max(prev_end - 1, -1), -1):
            line = lines[j].strip()
            if _is_system_marker(line):
                continue
            if not line:
                if hit_text:
                    break
                continue
            hit_text = True
            user_lines.insert(0, line)

        if user_lines:
            turns.append({
                "role": "user",
                "text": '\n'.join(user_lines),
                "timestamp": timestamp,
            })

        # Claude応答: タイムスタンプ直後〜次の区間
        next_boundary = ts_indices[idx + 1] if idx + 1 < len(ts_indices) else len(lines)
        # 次のタイムスタンプの手前にあるユーザー発言を除くため、
        # next_boundary から逆スキャンしてユーザー発言部分を特定
        assistant_end = next_boundary
        if idx + 1 < len(ts_indices):
            # 次のタイムスタンプの手前のユーザー発言を除外
            scan = next_boundary - 1
            hit = False
            while scan > ts_i:
                line = lines[scan].strip()
                if _is_system_marker(line) or _is_timestamp(line):
                    scan -= 1
                    continue
                if not line:
                    if hit:
                        assistant_end = scan + 1
                        break
                    scan -= 1
                    continue
                hit = True
                scan -= 1

        assistant_text = '\n'.join(lines[ts_i + 1:assistant_end]).strip()
        # システムマーカー行を除去
        cleaned = '\n'.join(
            l for l in assistant_text.split('\n')
            if not _is_system_marker(l)
        ).strip()
        if cleaned:
            turns.append({
                "role": "assistant",
                "text": cleaned,
                "timestamp": timestamp,
            })

    return turns


def _parse_with_headers(lines, header_indices):
    """ロールヘッダー ("あなた" / "Claude") で分割。"""
    turns = []

    for idx, (hi, header) in enumerate(header_indices):
        role = _role_from_header(header)
        if not role:
            continue

        # 次のヘッダーまたは末尾
        if idx + 1 < len(header_indices):
            end = header_indices[idx + 1][0]
        else:
            end = len(lines)

        block = '\n'.join(lines[hi + 1:end]).strip()
        # システムマーカーを除去
        block = '\n'.join(
            l for l in block.split('\n')
            if not _is_system_marker(l)
        ).strip()

        if block:
            turns.append({
                "role": role,
                "text": block,
                "timestamp": "",
            })

    return turns


def _parse_heuristic(lines):
    """
    ヘッダーもタイムスタンプもない場合のフォールバック。
    空行で区切られたブロックを交互に user/assistant と推定。
    短いブロック = user, 長いブロック = assistant。
    """
    blocks = []
    current = []
    for line in lines:
        if line.strip() == '':
            if current:
                blocks.append('\n'.join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append('\n'.join(current).strip())

    if not blocks:
        return []

    # ブロックが1つだけなら全体を user として返す
    if len(blocks) == 1:
        return [{"role": "user", "text": blocks[0], "timestamp": ""}]

    # 長さの中央値で user/assistant を推定
    lengths = sorted(len(b) for b in blocks)
    median_len = lengths[len(lengths) // 2]

    turns = []
    for block in blocks:
        if not block:
            continue
        # 中央値より短い or 最初のブロック → user と推定
        role = "user" if len(block) <= median_len else "assistant"
        turns.append({"role": role, "text": block, "timestamp": ""})

    # user が連続していたら結合
    merged = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["text"] += "\n\n" + t["text"]
        else:
            merged.append(t)

    return merged


# --- JSONL 内の claude.ai 会話を自動検出 ---

def extract_chat_from_jsonl(filepath):
    """
    JSONL ファイルから claude.ai の会話テキスト（大きなユーザーメッセージ）を検出する。
    タイムスタンプパターン（H:MM が単独行にある）を含む長いテキストを claude.ai 会話とみなす。
    """
    timestamp_pattern = re.compile(r'^\d{1,2}:\d{2}$', re.MULTILINE)
    # ロールヘッダーもチェック
    header_pattern = re.compile(r'^(あなた|You|Claude|Human|Assistant)\s*[:：]?\s*$', re.MULTILINE)
    chat_texts = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = data.get("message", {})
            role = msg.get("role", "")
            if role != "user":
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(texts)
            elif isinstance(content, str):
                text = content
            else:
                continue

            # 長い + (タイムスタンプ or ロールヘッダー) → claude.ai 会話
            if len(text) > 3000 and (timestamp_pattern.search(text) or header_pattern.search(text)):
                chat_texts.append(text)

    return chat_texts


# --- 記憶候補の抽出 ---

def _guess_category(text):
    """テキストからカテゴリを推定する。"""
    episode_markers = ["した", "やった", "できた", "始めた", "完了", "作った", "決めた",
                       "議論", "発見", "today", "yesterday"]
    context_markers = ["執筆中", "開発中", "進行中", "取り組", "プロジェクト", "計画",
                       "目標", "方針", "これから"]
    preference_markers = ["好き", "嫌い", "使う", "使わない", "避け", "好み",
                          "方が良い", "の方が", "prefer"]

    text_lower = text.lower()
    if any(m in text_lower or m in text for m in context_markers):
        return "context"
    if any(m in text_lower or m in text for m in preference_markers):
        return "preference"
    if any(m in text_lower or m in text for m in episode_markers):
        return "episode"
    return "fact"


def _is_garbage(text):
    """明確なゴミを判定。"""
    if len(text) < 8:
        return True

    # システムメッセージ系
    if any(tag in text for tag in [
        "<task-notification>", "<command-name>", "<local-command",
        "<system-reminder>", "<available-deferred-tools>",
        "This session is being continued",
        "<user-prompt-submit-hook>",
        "Stop hook feedback:",
        "[Request interrupted by user",
    ]):
        return True

    # XML タグが大半
    stripped = re.sub(r'<[^>]+>', '', text).strip()
    if len(stripped) < 8:
        return True

    return False


def extract_memory_candidates(turns, source=""):
    """
    会話ターンから記憶候補を生成する（chat_mode 専用）。
    """
    from Extract import segment_conversation

    segments = segment_conversation(turns)
    candidates = []
    source_name = Path(source).stem[:20] if source else "claude.ai"

    for seg in segments:
        user_texts = [t["text"] for t in seg if t["role"] == "user"]
        timestamp = seg[0].get("timestamp", "")

        for text in user_texts:
            if _is_garbage(text):
                continue

            # UI ラベル / ファイル名のみの行を除外
            meaningful_lines = [
                l for l in text.strip().split('\n')
                if l.strip() and not _LABEL_RE.match(l.strip())
            ]
            if not meaningful_lines:
                continue
            text = '\n'.join(meaningful_lines)

            # 情動検出
            emotions, arousal, importance = detect_emotions(text)

            if arousal < MIN_AROUSAL and not any(kw in text for kw in [
                "決めた", "始める", "やる", "したい", "方針", "これから",
                "好き", "嫌い", "使わない", "使う", "移行",
                "思う", "気づ", "発見", "わかった", "なるほど",
                "面白", "重要", "本質", "意味", "理由", "なぜ",
            ]):
                continue

            category = _guess_category(text)
            content = text[:200] if len(text) > 200 else text

            candidates.append({
                "content": content,
                "category": category,
                "emotions": emotions,
                "arousal": arousal,
                "importance": importance,
                "timestamp": timestamp,
                "source": source_name,
            })

    return candidates


# --- 重複チェック ---

def _is_duplicate(content, existing_memories):
    """既存の記憶と重複していないかチェック。"""
    model = get_model()
    if model is None:
        for mem in existing_memories:
            if content in mem["content"] or mem["content"] in content:
                return True
        return False

    new_vec = embed_text(content, is_query=True)
    if new_vec is None:
        return False

    for mem in existing_memories:
        if mem.get("embedding"):
            mem_vec = bytes_to_vec(mem["embedding"])
            sim = cosine_similarity(new_vec, mem_vec)
            if sim > DUPLICATE_THRESHOLD:
                return True

    return False


# --- メイン処理 ---

def process_chat_text(text, dry_run=False, source="claude.ai"):
    """
    claude.ai からコピペされた会話テキストを処理して記憶を抽出する。
    """
    print(f"\n💬 claude.ai会話を解析中...")

    turns = parse_chat_text(text)
    user_turns = [t for t in turns if t["role"] == "user"]
    print(f"  {len(turns)}ターン検出 (ユーザー: {len(user_turns)}件)")

    if not turns:
        return 0

    candidates = extract_memory_candidates(turns, source)
    print(f"  {len(candidates)}件の記憶候補を検出")

    if not candidates:
        return 0

    # 既存の記憶を取得（重複チェック用）
    conn = get_connection()
    existing = conn.execute(
        "SELECT content, embedding FROM memories WHERE forgotten = 0"
    ).fetchall()
    conn.close()
    existing_dicts = [dict(row) for row in existing]

    saved = 0
    seen = []
    for cand in candidates:
        if _is_duplicate(cand["content"], existing_dicts):
            print(f"  ⏭ 重複(既存): {cand['content'][:50]}...")
            continue

        cand_prefix = cand["content"][:50]
        if any(cand_prefix == prev[:50] for prev in seen):
            print(f"  ⏭ 重複(バッチ): {cand['content'][:50]}...")
            continue
        seen.append(cand["content"])

        emo_str = ", ".join(cand["emotions"]) if cand["emotions"] else "中立"

        if dry_run:
            print(f"  🧠 [{cand['category']}] ({emo_str}) {cand['content'][:80]}...")
        else:
            add_memory(cand["content"], cand["category"], cand["source"])
            saved += 1

    return saved


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    detect_mode = "--detect" in args
    stdin_mode = "--stdin" in args

    if dry_run:
        print("🔍 ドライランモード（保存しません）\n")

    if not os.path.exists(DB_PATH):
        init_db()

    total = 0

    if stdin_mode:
        text = sys.stdin.read()
        total += process_chat_text(text, dry_run)
    elif detect_mode:
        # JSONL 内の claude.ai 会話を自動検出
        jsonl_files = [a for a in args if not a.startswith("--") and a.endswith(".jsonl") and os.path.exists(a)]
        for f in jsonl_files:
            chat_texts = extract_chat_from_jsonl(f)
            if chat_texts:
                print(f"📖 {Path(f).name}: {len(chat_texts)}件のclaude.ai会話を検出")
                for ct in chat_texts:
                    total += process_chat_text(ct, dry_run, source=Path(f).stem[:20])
    else:
        # テキストファイルを直接パース
        txt_files = [a for a in args if not a.startswith("--") and os.path.exists(a)]
        if not txt_files:
            print("使い方: python ingest_chat.py <conversation.txt> [--dry-run]")
            print("        python ingest_chat.py --detect <session.jsonl>")
            print("        cat conversation.txt | python ingest_chat.py --stdin")
            return

        for f in txt_files:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
            total += process_chat_text(text, dry_run, source=Path(f).stem)

    if dry_run:
        print(f"\n📊 ドライラン結果: {total}件の記憶候補")
    else:
        print(f"\n✓ 完了: {total}件の新しい記憶を保存しました")
        if total > 0:
            print("\n🔄 リプレイ実行中...")
            from memory import replay_memories
            replay_memories()


if __name__ == "__main__":
    main()
