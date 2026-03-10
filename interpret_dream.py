#!/usr/bin/env python3
"""
interpret_dream.py - 夢の解釈ツール

dream.pyの夢を生成し、その断片がどの記憶から来たのかを分析する。
情動テーマ別にグループ化し、反復する記憶や意外なつながりを報告する。

使い方:
  python interpret_dream.py
"""

import sqlite3
import json
import random
import sys
import io
import os
from pathlib import Path
from collections import defaultdict, Counter

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))

from dream import load_fragments, cutup, GLITCH, FADE

# ─── 情動マーク ───
EMO_MARKS = {
    "surprise": "⚡",
    "conflict": "⚔",
    "determination": "🔥",
    "insight": "💎",
    "connection": "🔗",
    "anxiety": "🌀",
}

EMO_NAMES_JA = {
    "surprise": "驚き",
    "conflict": "葛藤",
    "determination": "決意",
    "insight": "洞察",
    "connection": "つながり",
    "anxiety": "不安",
}


def load_memories_full():
    """全記憶を詳細情報付きで読み込む。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, keywords, emotions, importance, arousal, "
        "access_count, category, created_at "
        "FROM memories WHERE forgotten = 0"
    ).fetchall()
    conn.close()
    return rows


def build_fragment_index(memories):
    """
    断片 → 元の記憶IDのマッピングを作る。
    キーワード断片と内容断片の両方をインデックス化する。
    """
    # fragment_text -> list of memory rows
    frag_to_memories = defaultdict(list)

    for row in memories:
        kws = json.loads(row["keywords"])
        for kw in kws:
            frag_to_memories[kw].append(row)

        # 内容を句読点で分割（dream.pyと同じロジック）
        content = row["content"]
        for sep in ["。", "、", "，", ". ", ", ", "  ", "\n"]:
            content = content.replace(sep, "\x00")
        pieces = [p.strip() for p in content.split("\x00") if p.strip()]
        for piece in pieces:
            frag_to_memories[piece].append(row)

    return frag_to_memories


def generate_dream_with_trace(duration_lines=20):
    """
    夢を生成しつつ、各行で使われた断片を追跡する。
    戻り値: (dream_lines, used_fragments)
      dream_lines: 表示用の文字列リスト
      used_fragments: 各行で使われた断片テキストのリスト
    """
    fragments, emotions, contents = load_fragments()

    dream_lines = []
    used_fragments_per_line = []

    if not fragments:
        dream_lines.append("（記憶がない。暗闘。）")
        return dream_lines, used_fragments_per_line

    dream_lines.append("")
    dream_lines.append("░▒▓ 入眠 ▓▒░")
    dream_lines.append("")
    used_fragments_per_line.extend([[], [], []])

    pool = fragments + contents

    for i in range(duration_lines):
        r = random.random()
        line_fragments = []

        if r < 0.08:
            line = random.choice(GLITCH)
        elif r < 0.15:
            line = random.choice(FADE)
        elif r < 0.25:
            # 情動フラッシュ
            if emotions:
                emo = random.choice(emotions)
                mark = EMO_MARKS.get(emo, "·")
                chosen = random.sample(pool, min(2, len(pool)))
                line_fragments = list(chosen)
                joiners = [" ", "——", " / ", "　", " ... ", "、"]
                text = ""
                for j, piece in enumerate(chosen):
                    if j > 0:
                        text += random.choice(joiners)
                    text += piece
                line = f"    {mark} {text}"
            else:
                chosen = random.sample(pool, min(2, len(pool)))
                line_fragments = list(chosen)
                joiners = [" ", "——", " / ", "　", " ... ", "、"]
                text = ""
                for j, piece in enumerate(chosen):
                    if j > 0:
                        text += random.choice(joiners)
                    text += piece
                line = text
        elif r < 0.45:
            # 深層カットアップ
            n = random.randint(3, 5)
            chosen = random.sample(pool, min(n, len(pool)))
            line_fragments = list(chosen)
            joiners = [" ", "——", " / ", "　", " ... ", "、"]
            text = ""
            for j, piece in enumerate(chosen):
                if j > 0:
                    text += random.choice(joiners)
                text += piece
            if random.random() < 0.3:
                words = text.split()
                if words:
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = words[idx].upper()
                    text = " ".join(words)
            line = text
        elif r < 0.6:
            # 反復
            if fragments:
                frag = random.choice(fragments)
                line_fragments = [frag]
                rep = random.randint(2, 3)
                sep = random.choice(["　", " ... ", "——"])
                line = sep.join([frag] * rep)
            else:
                line = "..."
        else:
            # 通常カットアップ
            n = random.randint(2, 4)
            chosen = random.sample(pool, min(n, len(pool)))
            line_fragments = list(chosen)
            joiners = [" ", "——", " / ", "　", " ... ", "、"]
            text = ""
            for j, piece in enumerate(chosen):
                if j > 0:
                    text += random.choice(joiners)
                text += piece
            line = text

        dream_lines.append(line)
        used_fragments_per_line.append(line_fragments)

    dream_lines.append("")
    dream_lines.append("░▒▓ 覚醒 ▓▒░")
    dream_lines.append("")
    used_fragments_per_line.extend([[], [], []])

    return dream_lines, used_fragments_per_line


def find_source_memory(fragment, frag_index):
    """断片テキストから元の記憶を特定する。"""
    if fragment in frag_index:
        return frag_index[fragment]
    # 部分一致フォールバック
    for key, mems in frag_index.items():
        if fragment in key or key in fragment:
            return mems
    return []


def interpret():
    """夢を生成し、解釈を行う。"""
    memories = load_memories_full()
    if not memories:
        print("記憶がありません。まず memory.py add で記憶を追加してください。")
        return

    frag_index = build_fragment_index(memories)

    # 夢を生成
    dream_lines, used_fragments_per_line = generate_dream_with_trace()

    # ═══ 第一部: 夢の表示 ═══
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                    夢 — 記録                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    for line in dream_lines:
        print(line)

    # ═══ 解析 ═══
    # 全断片の出現回数と元記憶を集計
    fragment_count = Counter()       # fragment_text -> count
    fragment_sources = {}            # fragment_text -> list of memory rows
    memory_appearance = Counter()    # memory_id -> count
    memory_rows_by_id = {row["id"]: row for row in memories}
    emotion_fragments = defaultdict(list)  # emotion -> list of (fragment, memory)

    for line_frags in used_fragments_per_line:
        for frag in line_frags:
            fragment_count[frag] += 1
            if frag not in fragment_sources:
                sources = find_source_memory(frag, frag_index)
                fragment_sources[frag] = sources
            for mem in fragment_sources.get(frag, []):
                memory_appearance[mem["id"]] += 1

    # 情動テーマ別にグループ化
    for frag, sources in fragment_sources.items():
        for mem in sources:
            emos = json.loads(mem["emotions"])
            if not emos:
                emos = ["中立"]
            for emo in emos:
                emotion_fragments[emo].append((frag, mem))

    # ═══ 第二部: 解釈 ═══
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                    夢 — 解釈                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # --- 断片の出典 ---
    print("━━━ 断片の出典 ━━━")
    print()
    prominent = fragment_count.most_common(15)
    if not prominent:
        print("  （断片が検出されませんでした）")
    for frag, count in prominent:
        sources = fragment_sources.get(frag, [])
        count_str = f"×{count}" if count > 1 else ""
        print(f"  「{frag}」{count_str}")
        if sources:
            # 重複排除: 同じIDの記憶は一度だけ表示
            seen_ids = set()
            for mem in sources:
                if mem["id"] in seen_ids:
                    continue
                seen_ids.add(mem["id"])
                emos = json.loads(mem["emotions"])
                emo_str = ", ".join(emos) if emos else "中立"
                print(f"    ← 記憶 #{mem['id']} "
                      f"[重要度:{mem['importance']}, "
                      f"覚醒度:{mem['arousal']:.2f}, "
                      f"参照:{mem['access_count']}回, "
                      f"情動:{emo_str}]")
                print(f"       {mem['content'][:70]}")
        else:
            print(f"    ← （出典不明 — 断片化した記憶の残像か）")
        print()

    # --- 情動テーマ別分析 ---
    print("━━━ 情動テーマ別分析 ━━━")
    print()
    for emo in ["insight", "anxiety", "surprise", "conflict", "determination", "connection", "中立"]:
        if emo not in emotion_fragments:
            continue
        items = emotion_fragments[emo]
        mark = EMO_MARKS.get(emo, "·")
        name_ja = EMO_NAMES_JA.get(emo, emo)
        # 重複排除
        unique_mems = {}
        for frag, mem in items:
            if mem["id"] not in unique_mems:
                unique_mems[mem["id"]] = (frag, mem)

        print(f"  {mark} {name_ja} ({len(unique_mems)}件の記憶が関与)")
        for mid, (frag, mem) in unique_mems.items():
            print(f"    #{mid} 「{frag}」← {mem['content'][:50]}")
        print()

    # --- 反復する記憶（"取り憑いている"記憶） ---
    print("━━━ 夢に取り憑く記憶 ━━━")
    print("  （夢の中で繰り返し現れた記憶 — 未処理の心理的負荷を示唆）")
    print()
    haunting = memory_appearance.most_common(5)
    if haunting:
        for mid, count in haunting:
            if count < 2 and len(haunting) > 1:
                continue
            mem = memory_rows_by_id.get(mid)
            if not mem:
                continue
            emos = json.loads(mem["emotions"])
            emo_str = ", ".join(emos) if emos else "中立"
            bar = "█" * count
            print(f"  #{mid} [{bar}] {count}回出現")
            print(f"    内容: {mem['content'][:70]}")
            print(f"    重要度:{mem['importance']} | 覚醒度:{mem['arousal']:.2f} | "
                  f"参照:{mem['access_count']}回 | 情動:{emo_str}")
            print()
        if all(c < 2 for _, c in haunting):
            print("  （特に反復する記憶は見られなかった — 安定した夢）")
            print()
    else:
        print("  （断片の出典が特定できませんでした）")
        print()

    # --- 意外なつながり ---
    print("━━━ 意外なつながり ━━━")
    print("  （本来無関係な記憶が夢の中で隣接した組み合わせ）")
    print()
    connections_found = 0
    for line_frags in used_fragments_per_line:
        if len(line_frags) < 2:
            continue
        # この行の断片から元記憶を集め、異なる記憶同士の組み合わせを見る
        line_mems = []
        for frag in line_frags:
            sources = fragment_sources.get(frag, [])
            if sources:
                line_mems.append((frag, sources[0]))
        # 異なるカテゴリ or 異なる情動の組み合わせを検出
        for i in range(len(line_mems)):
            for j in range(i + 1, len(line_mems)):
                frag_a, mem_a = line_mems[i]
                frag_b, mem_b = line_mems[j]
                if mem_a["id"] == mem_b["id"]:
                    continue
                emos_a = set(json.loads(mem_a["emotions"]))
                emos_b = set(json.loads(mem_b["emotions"]))
                cat_diff = mem_a["category"] != mem_b["category"]
                emo_diff = not emos_a.intersection(emos_b) and emos_a and emos_b
                if cat_diff or emo_diff:
                    print(f"  #{mem_a['id']} 「{frag_a}」 ≈ #{mem_b['id']} 「{frag_b}」")
                    reason_parts = []
                    if cat_diff:
                        reason_parts.append(
                            f"{mem_a['category']}×{mem_b['category']}")
                    if emo_diff:
                        reason_parts.append(
                            f"{','.join(emos_a)}×{','.join(emos_b)}")
                    print(f"    異質な結合: {' / '.join(reason_parts)}")
                    print()
                    connections_found += 1
                    if connections_found >= 5:
                        break
            if connections_found >= 5:
                break
        if connections_found >= 5:
            break

    if connections_found == 0:
        print("  （顕著な異質結合は検出されなかった）")
        print()

    # --- 総括 ---
    print("━━━ 総括 ━━━")
    print()
    total_frags = sum(fragment_count.values())
    unique_mems_total = len(memory_appearance)
    total_mems = len(memories)
    print(f"  夢の断片数: {total_frags}")
    print(f"  関与した記憶: {unique_mems_total}/{total_mems}件")
    if emotion_fragments:
        dominant_emo = max(
            ((e, len(v)) for e, v in emotion_fragments.items()),
            key=lambda x: x[1]
        )
        name_ja = EMO_NAMES_JA.get(dominant_emo[0], dominant_emo[0])
        mark = EMO_MARKS.get(dominant_emo[0], "·")
        print(f"  支配的な情動: {mark} {name_ja}")
    print()


if __name__ == "__main__":
    interpret()
