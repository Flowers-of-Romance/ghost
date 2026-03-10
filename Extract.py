#!/usr/bin/env python3
"""
extract.py - Claude Code会話ログから記憶を自動抽出する

海馬のシミュレーション:
  会話ログを読み、「何を覚えるべきか」を自動判断して memory.db に保存する。
  人間の睡眠中のリプレイに相当する処理。

使い方:
  # 最新のセッションから記憶を抽出
  python extract.py

  # 特定のJSONLファイルから抽出
  python extract.py path/to/session.jsonl

  # 全セッションから抽出（初回の大量取り込み用）
  python extract.py --all

  # ドライラン（保存せず表示のみ）
  python extract.py --dry-run

  # Claude Codeのプロジェクトパスを指定
  python extract.py --project /path/to/project

仕組み:
  1. JSONLを読んでユーザーとアシスタントの発言を抽出
  2. 会話を「トピック」に分割
  3. 各トピックから記憶候補を生成（ルールベース + 情動検出）
  4. 既存の記憶と重複チェック（ベクトル類似度）
  5. 新しい記憶だけを memory.db に保存


  # claude.ai会話テキストから記憶を抽出（コピペ対応）
  python extract.py --chat conversation.txt [--dry-run]
"""

import json
import sys
import os
import io
import re
import glob
from pathlib import Path
from datetime import datetime, timezone

# Windows cp932 で emoji が出力できない問題を回避
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# memory.py をインポート
sys.path.insert(0, str(Path(__file__).parent))
import memory as mem_module
from memory import (
    init_db, add_memory, search_memories, get_all, get_connection,
    detect_emotions, extract_keywords, embed_text, vec_to_bytes, bytes_to_vec,
    cosine_similarity, get_model, DB_PATH
)

# --- 設定 ---
# Claude Codeの会話ログの場所
CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
HISTORY_FILE = CLAUDE_DIR / "history.jsonl"

# 重複と判定する類似度の閾値
DUPLICATE_THRESHOLD = 0.85

# 記憶候補の最小情動覚醒度（これ以下は「覚えなくていい」と判断）
MIN_AROUSAL = 0.15


def find_project_dir(project_path=None):
    """Claude Codeのプロジェクトディレクトリを見つける。"""
    if not PROJECTS_DIR.exists():
        print(f"⚠ Claude Codeのプロジェクトディレクトリが見つかりません: {PROJECTS_DIR}")
        return None

    if project_path:
        # パスをClaude Code形式に変換（/をーに）
        normalized = project_path.replace("/", "-").replace("\\", "-")
        if normalized.startswith("-"):
            pass  # そのまま
        candidates = list(PROJECTS_DIR.glob(f"*{normalized}*"))
        if candidates:
            return candidates[0]

    # 全プロジェクトを返す
    return PROJECTS_DIR


def find_session_files(project_dir=None, latest_only=True):
    """セッションのJSONLファイルを見つける。"""
    if project_dir is None:
        project_dir = PROJECTS_DIR

    if not project_dir.exists():
        return []

    pattern = str(project_dir / "**" / "*.jsonl")
    files = glob.glob(pattern, recursive=True)

    # sessions-index.json、history.jsonl、subagentのセッションは除外
    files = [f for f in files
             if "sessions-index" not in f
             and "history.jsonl" not in f
             and "subagents" not in f]

    if not files:
        return []

    # 更新日時でソート
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    if latest_only:
        return files[:1]

    return files


def parse_jsonl(filepath):
    """
    JSONLファイルを読んで会話のターンを抽出する。
    Claude Codeの形式:
      各行がJSON。message.role が "user" or "assistant"
      message.content は配列で、各要素に type="text" のテキストがある
    """
    turns = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = data.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")
            timestamp = data.get("timestamp", "")

            # contentがリストの場合（Claude Codeの標準形式）
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                text = "\n".join(texts)
            elif isinstance(content, str):
                text = content
            else:
                continue

            if not text.strip():
                continue

            if role in ("user", "assistant"):
                turns.append({
                    "role": role,
                    "text": text.strip(),
                    "timestamp": timestamp,
                })

    return turns


def parse_chat_text(text):
    """
    claude.aiからコピペされた会話テキストをパースする。

    フォーマットの特徴:
    - タイムスタンプ行: "8:40", "9:03" のような H:MM 形式が単独行にある
    - タイムスタンプの直前がユーザー発言（短い、数行）
    - タイムスタンプの直後がClaude応答（長い段落）
    - "ウェブを検索しました" は区切りマーカー
    - "ファイルを作成しました" 等もシステムマーカー
    """
    lines = text.split('\n')
    turns = []

    # タイムスタンプ行のインデックスを見つける
    timestamp_pattern = re.compile(r'^\d{1,2}:\d{2}$')
    ts_indices = []
    for i, line in enumerate(lines):
        if timestamp_pattern.match(line.strip()):
            ts_indices.append(i)

    if not ts_indices:
        # タイムスタンプがない場合は全体を1つのテキストとして扱う
        return [{"role": "user", "text": text.strip(), "timestamp": ""}]

    # 各タイムスタンプ区間をパース
    # タイムスタンプの前 = ユーザー発言、後 = Claude応答
    for idx, ts_i in enumerate(ts_indices):
        timestamp = lines[ts_i].strip()

        # ユーザー発言: 前のタイムスタンプ（またはテキスト先頭）からこのタイムスタンプまで
        if idx == 0:
            # 最初のタイムスタンプより前はClaude応答（会話の冒頭）
            prev_end = 0
        else:
            prev_end = ts_indices[idx - 1] + 1

        # タイムスタンプの直前の行群からユーザー発言を抽出
        # パターン: Claude応答 → 空行 → ユーザー発言(数行) → 空行 → タイムスタンプ
        # タイムスタンプから逆スキャンして、空行→テキスト→空行を見つける
        user_lines = []
        hit_text = False
        for j in range(ts_i - 1, max(prev_end - 1, -1), -1):
            line = lines[j].strip()
            # システムマーカーはスキップ
            if any(m in line for m in [
                "ウェブを検索しました", "ファイルを作成しました",
                "コマンドを実行しました", "ファイルを読み取りました",
                "ファイルを表示しました", "コード · ", "ドキュメント · ",
                "もっと表示",
            ]):
                continue
            if not line:
                if hit_text:
                    break  # テキストの後の空行 = ユーザー発言の上端
                continue  # タイムスタンプ直前の空行はスキップ
            hit_text = True
            user_lines.insert(0, line)

        if user_lines:
            user_text = '\n'.join(user_lines)
            turns.append({
                "role": "user",
                "text": user_text,
                "timestamp": timestamp,
            })

        # Claude応答: タイムスタンプの次の行から次のユーザー発言まで
        next_ts = ts_indices[idx + 1] if idx + 1 < len(ts_indices) else len(lines)
        assistant_lines = []
        for j in range(ts_i + 1, next_ts):
            assistant_lines.append(lines[j])

        assistant_text = '\n'.join(assistant_lines).strip()
        if assistant_text:
            turns.append({
                "role": "assistant",
                "text": assistant_text,
                "timestamp": timestamp,
            })

    return turns


def segment_conversation(turns):
    """
    会話をトピック単位に分割する。
    ユーザーの発言を起点に、その応答までを一つのセグメントにする。
    """
    segments = []
    current = []

    for turn in turns:
        if turn["role"] == "user" and current:
            segments.append(current)
            current = []
        current.append(turn)

    if current:
        segments.append(current)

    return segments


def extract_memory_candidates(segments, source_file="", chat_mode=False):
    """
    各セグメントから記憶候補を生成する。

    chat_mode=True: claude.ai会話用（短い対話的発言が多い、フィルタ緩め）
    chat_mode=False: Claude Code用（コマンド的指示が多い、フィルタ厳しめ）

    何を覚えるかの判断基準（海馬のフィルタ）:
    1. ユーザーが明示的に決定・宣言したこと
    2. 新しい発見や気づき
    3. 情動的に強い発言
    4. プロジェクトの文脈や方針
    5. ユーザーの好みや傾向が表れた発言

    何を覚えないか:
    - コードの具体的な内容（一般知識）
    - エラーメッセージとその修正（一時的）
    - 単純な質問と回答（再利用性低い）
    """
    candidates = []
    source_name = Path(source_file).stem[:20] if source_file else ""

    for seg in segments:
        user_texts = [t["text"] for t in seg if t["role"] == "user"]
        assistant_texts = [t["text"] for t in seg if t["role"] == "assistant"]
        timestamp = seg[0].get("timestamp", "")

        for text in user_texts:
            # --- Phase 1: 明確なゴミを除外 ---

            min_len = 8 if chat_mode else 30

            # 短すぎる発言は無視
            if len(text) < min_len:
                continue

            # システムメッセージ・タグ・hookフィードバックを除外
            if any(tag in text for tag in [
                "<task-notification>", "<command-name>", "<local-command",
                "<system-reminder>", "<available-deferred-tools>",
                "This session is being continued from a previous conversation",
                "<user-prompt-submit-hook>",
                "Stop hook feedback:",
                "[Request interrupted by user",
                "ウェブを検索しました",  # claude.aiからのペースト
            ]):
                continue

            # XMLタグが大半を占めるテキストを除外
            import re as _re
            stripped = _re.sub(r'<[^>]+>', '', text).strip()
            if len(stripped) < (8 if chat_mode else 20):
                continue

            # コマンド・コード・URL単体は無視
            if text.startswith(("```", "ls ", "cd ", "cat ", "git ", "pip ", "npm ",
                                "/exit", "/clear", "/init", "/logout",
                                "claude ", "$env:", "http://", "https://",
                                "\\\\wsl")):
                continue

            if not chat_mode:
                # Claude Code用: 単純な指示を除外
                if len(stripped) < 50 and not any(kw in text for kw in [
                    "思う", "気づ", "感じ", "考え", "だろう", "かもしれ",
                    "面白", "重要", "本質", "意味", "理由", "なぜ",
                    "発見", "わかった", "なるほど",
                    "決めた", "始める", "やめる", "方針", "これから",
                    "好き", "嫌い", "不安", "困", "問題",
                ]):
                    continue

                # Claudeの応答がユーザー発言に混入している場合を除外
                if any(phrase in text for phrase in [
                    "確認してみます", "調べてみます", "ですね。",
                    "手順をまとめます", "アクセスできなかったのですが",
                    "Haikuの場合、", "VirtualBoxの",
                ]):
                    continue

            if chat_mode:
                # claude.ai用: UIラベル・ファイル名・パス単体を除外
                stripped_lines = [l for l in text.strip().split('\n') if l.strip()]
                # 全行がUIラベル/ファイル名っぽい場合のみ除外
                label_patterns = re.compile(
                    r'^(Memory|Readme|Claude|PY|Extract|コード\s*·|ドキュメント\s*·|'
                    r'[A-Z]:\\.*|Q:|A:|もっと表示)$', re.IGNORECASE
                )
                meaningful_lines = [l for l in stripped_lines
                                    if not label_patterns.match(l.strip())]
                if not meaningful_lines:
                    continue
                # 意味のある行だけでテキストを再構成
                text = '\n'.join(meaningful_lines)

            # --- Phase 2: 記憶価値の判定 ---

            # 情動を検出
            emotions, arousal, importance = detect_emotions(text)

            if not chat_mode:
                # Claude Code用: 情動が弱い + キーワードなし → スキップ
                if arousal < MIN_AROUSAL and not any(kw in text for kw in [
                    "決めた", "始める", "やる", "したい", "方針", "これから",
                    "好き", "嫌い", "使わない", "使う", "移行",
                    "思う", "気づ", "発見", "わかった", "なるほど",
                ]):
                    continue

            # カテゴリを推定
            category = guess_category(text)

            # 記憶候補として追加
            # 長すぎるテキストは要約（先頭200文字）
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


def guess_category(text):
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


def is_duplicate(content, existing_memories, threshold=DUPLICATE_THRESHOLD):
    """
    既存の記憶と重複していないかチェック。
    ベクトル類似度が閾値を超えたら重複とみなす。
    """
    model = get_model()
    if model is None:
        # embeddingが使えない場合は文字列一致で簡易チェック
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
            if sim > threshold:
                return True

    return False


def process_session(filepath, dry_run=False, seen_contents=None):
    """一つのセッションファイルを処理する。"""
    if seen_contents is None:
        seen_contents = []

    print(f"\n📖 読み込み中: {filepath}")

    turns = parse_jsonl(filepath)
    if not turns:
        print("  （会話ターンが見つかりません）")
        return 0

    print(f"  {len(turns)}ターンの会話")

    segments = segment_conversation(turns)
    print(f"  {len(segments)}セグメントに分割")

    candidates = extract_memory_candidates(segments, filepath)
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
    for cand in candidates:
        # 既存の記憶との重複チェック
        if is_duplicate(cand["content"], existing_dicts):
            print(f"  ⏭ 重複(既存): {cand['content'][:50]}...")
            continue

        # セッション横断の重複チェック（先頭50文字 + ベクトル類似度の簡易版）
        cand_prefix = cand["content"][:50]
        if any(cand_prefix == prev[:50] for prev in seen_contents):
            print(f"  ⏭ 重複(バッチ): {cand['content'][:50]}...")
            continue
        seen_contents.append(cand["content"])

        emo_str = ", ".join(cand["emotions"]) if cand["emotions"] else "中立"

        if dry_run:
            print(f"  🧠 [{cand['category']}] ({emo_str}) {cand['content'][:60]}...")
        else:
            add_memory(cand["content"], cand["category"], cand["source"])
            saved += 1

    return saved


def process_chat_text(text, dry_run=False, source="claude.ai"):
    """
    claude.aiからコピペされた会話テキストを処理して記憶を抽出する。
    """
    print(f"\n💬 claude.ai会話を解析中...")

    turns = parse_chat_text(text)
    user_turns = [t for t in turns if t["role"] == "user"]
    print(f"  {len(turns)}ターン検出 (ユーザー: {len(user_turns)}件)")

    if not turns:
        return 0

    segments = segment_conversation(turns)
    print(f"  {len(segments)}セグメントに分割")

    candidates = extract_memory_candidates(segments, source, chat_mode=True)
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
        if is_duplicate(cand["content"], existing_dicts):
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


def extract_chat_from_jsonl(filepath):
    """
    JONSLファイルからclaude.aiの会話テキスト（大きなユーザーメッセージ）を検出する。
    タイムスタンプパターン（H:MM が単独行にある）を含む長いテキストをclaude.ai会話とみなす。
    """
    timestamp_pattern = re.compile(r'^\d{1,2}:\d{2}$', re.MULTILINE)
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

            # 長い + タイムスタンプパターンを含む → claude.ai会話
            if len(text) > 5000 and timestamp_pattern.search(text):
                chat_texts.append(text)

    return chat_texts


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    process_all = "--all" in args
    chat_mode = "--chat" in args
    project_path = None

    # --project オプション
    for i, arg in enumerate(args):
        if arg == "--project" and i + 1 < len(args):
            project_path = args[i + 1]

    # 明示的なファイルパスが指定された場合
    explicit_files = [a for a in args
                      if not a.startswith("--")
                      and os.path.exists(a)
                      and (a.endswith(".jsonl") or a.endswith(".txt"))]

    if dry_run:
        print("🔍 ドライランモード（保存しません）\n")

    # DBが存在しない場合は初期化
    if not os.path.exists(DB_PATH):
        init_db()

    # --chat モード: JONSLからclaude.ai会話を自動検出、またはテキストファイルを直接パース
    if chat_mode:
        total = 0
        txt_files = [a for a in explicit_files if a.endswith(".txt")]

        if txt_files:
            # テキストファイルを直接パース
            for f in txt_files:
                with open(f, "r", encoding="utf-8") as fh:
                    text = fh.read()
                total += process_chat_text(text, dry_run, source=Path(f).stem)
        else:
            # 全JONSLからclaude.ai会話を自動検出
            project_dir = find_project_dir(project_path)
            if project_dir:
                files = find_session_files(project_dir, latest_only=not process_all)
                print(f"{'全' if process_all else '最新の'}{len(files)}セッションからclaude.ai会話を検出します")
                for f in files:
                    chat_texts = extract_chat_from_jsonl(f)
                    if chat_texts:
                        print(f"\n📖 {Path(f).name}: {len(chat_texts)}件のclaude.ai会話を検出")
                        for ct in chat_texts:
                            total += process_chat_text(ct, dry_run, source=Path(f).stem[:20])

        if dry_run:
            print(f"\n📊 ドライラン結果: {total}件の記憶候補")
        else:
            print(f"\n✓ 完了: {total}件の新しい記憶を保存しました")
            if total > 0:
                print("\n🔄 リプレイ実行中...")
                from memory import replay_memories
                replay_memories()
        return

    seen_contents = []  # セッション横断の重複チェック用

    if explicit_files:
        total = 0
        for f in explicit_files:
            total += process_session(f, dry_run, seen_contents)
    else:
        project_dir = find_project_dir(project_path)
        if project_dir is None:
            print("Claude Codeのプロジェクトが見つかりません。")
            print(f"  確認: {PROJECTS_DIR} が存在するか")
            print(f"  または: python extract.py path/to/session.jsonl で直接指定")
            return

        files = find_session_files(project_dir, latest_only=not process_all)
        if not files:
            print("セッションファイルが見つかりません。")
            return

        print(f"{'全' if process_all else '最新の'}{len(files)}セッションを処理します")
        total = 0
        for f in files:
            total += process_session(f, dry_run, seen_contents)

    if dry_run:
        print(f"\n📊 ドライラン結果: {total}件の記憶候補")
    else:
        print(f"\n✓ 完了: {total}件の新しい記憶を保存しました")

    # リプレイ（リンク更新）
    if not dry_run and total > 0:
        print("\n🔄 リプレイ実行中...")
        from memory import replay_memories
        replay_memories()


if __name__ == "__main__":
    main()