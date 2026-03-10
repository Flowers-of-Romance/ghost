#!/usr/bin/env python3
"""
dream.py - バロウズ式カットアップ夢表示

記憶の断片をシャッフルし、意識の流れとして表示する。
睡眠（replay/consolidate）中の脳内イメージ。

arousalが高い記憶ほど夢に出やすい（重み付きサンプリング）。
外傷的記憶（arousal >= 0.85）は夢に頻出し、反復する。
"""

import sqlite3
import json
import random
import time
import sys
import io
import os
from pathlib import Path

if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))

TRAUMA_AROUSAL_THRESHOLD = 0.85

# 夢の素材
GLITCH = ["...", "———", "   ", "///", "~~~", "▓▒░", "░▒▓", ":::", "≈≈≈", "∴∴∴"]
FADE = ["　", "　　", "　　　", "　　　　　"]


def load_fragments():
    """記憶の断片をarousal重み付きで取り出す。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, keywords, emotions, importance, arousal, access_count "
        "FROM memories WHERE forgotten = 0"
    ).fetchall()

    # リンク情報を取得（連想クラスタ用）
    links = {}
    for link in conn.execute("SELECT source_id, target_id, strength FROM links").fetchall():
        links.setdefault(link["source_id"], []).append(
            (link["target_id"], link["strength"])
        )

    conn.close()

    # 各記憶の断片を重み付きで収集
    # weight = arousal^2 + 0.1（最低重みを保証）
    weighted_fragments = []  # (fragment_text, weight)
    weighted_contents = []   # (content_piece, weight)
    emotions = []
    memory_clusters = []     # (memory_id, [fragment_texts], weight) — クラスタ出力用

    for row in rows:
        arousal = row["arousal"]
        weight = arousal * arousal + 0.1

        kws = json.loads(row["keywords"])
        emos = json.loads(row["emotions"])

        for kw in kws:
            weighted_fragments.append((kw, weight))
        emotions.extend(emos)

        # 内容を句読点やスペースで切る
        content = row["content"]
        for sep in ["。", "、", "，", ". ", ", ", "  ", "\n"]:
            content = content.replace(sep, "\x00")
        pieces = [p.strip() for p in content.split("\x00") if p.strip()]
        for piece in pieces:
            weighted_contents.append((piece, weight))

        memory_clusters.append((row["id"], kws + pieces, weight))

    return weighted_fragments, weighted_contents, emotions, memory_clusters, links


def weighted_sample(pool, n):
    """重み付きサンプリング。poolは(text, weight)のリスト。"""
    if not pool:
        return []
    texts, weights = zip(*pool)
    # random.choicesは重複を許すのでsetで重複除去してから
    chosen = []
    remaining = list(range(len(pool)))
    for _ in range(min(n, len(pool))):
        if not remaining:
            break
        w = [weights[i] for i in remaining]
        idx = random.choices(remaining, weights=w, k=1)[0]
        chosen.append(texts[idx])
        remaining.remove(idx)
    return chosen


def cutup(weighted_fragments, weighted_contents, n=3):
    """バロウズ式カットアップ: arousal重み付きで断片を組み合わせる。"""
    pool = weighted_fragments + weighted_contents
    if len(pool) < 2:
        return "..."
    chosen = weighted_sample(pool, n)
    joiners = [" ", "——", " / ", "　", " ... ", "、"]
    result = ""
    for i, piece in enumerate(chosen):
        if i > 0:
            result += random.choice(joiners)
        result += piece
    return result


def dream_sequence(duration_lines=20):
    """夢を表示する。"""
    weighted_fragments, weighted_contents, emotions, clusters, links = load_fragments()

    if not weighted_fragments:
        print("（記憶がない。暗闇。）")
        return

    emo_marks = {
        "surprise": "⚡", "conflict": "⚔", "determination": "🔥",
        "insight": "💎", "connection": "🔗", "anxiety": "🌀",
    }

    # 外傷的記憶を特定（反復候補）
    trauma_fragments = [(text, w) for text, w in weighted_fragments
                        if w >= TRAUMA_AROUSAL_THRESHOLD ** 2 + 0.1]

    print()
    print("░▒▓ 入眠 ▓▒░")
    print()
    time.sleep(0.3)

    for i in range(duration_lines):
        r = random.random()

        if r < 0.08:
            # グリッチ
            print(random.choice(GLITCH))
        elif r < 0.15:
            # フェード（空白行）
            print(random.choice(FADE))
        elif r < 0.25:
            # 情動フラッシュ
            if emotions:
                emo = random.choice(emotions)
                mark = emo_marks.get(emo, "·")
                print(f"    {mark} {cutup(weighted_fragments, weighted_contents, 2)}")
            else:
                print(cutup(weighted_fragments, weighted_contents, 2))
        elif r < 0.45:
            # 深層カットアップ（長め）
            line = cutup(weighted_fragments, weighted_contents, random.randint(3, 5))
            if random.random() < 0.3:
                words = line.split()
                if words:
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = words[idx].upper()
                    line = " ".join(words)
            print(line)
        elif r < 0.55:
            # 反復（外傷的記憶があればそちらを優先）
            if trauma_fragments and random.random() < 0.7:
                frag = weighted_sample(trauma_fragments, 1)[0]
            elif weighted_fragments:
                frag = weighted_sample(weighted_fragments, 1)[0]
            else:
                continue
            rep = random.randint(2, 3)
            sep = random.choice(["　", " ... ", "——"])
            print(sep.join([frag] * rep))
        elif r < 0.7:
            # 連想クラスタ（リンクが強い記憶同士を一緒に出す）
            if clusters:
                # 重み付きでクラスタを選ぶ
                cluster_weights = [c[2] for c in clusters]
                base = random.choices(clusters, weights=cluster_weights, k=1)[0]
                base_id, base_frags, _ = base
                parts = [random.choice(base_frags)] if base_frags else []
                # リンク先の断片を混ぜる
                if base_id in links:
                    linked = sorted(links[base_id], key=lambda x: -x[1])[:3]
                    for lid, strength in linked:
                        for c in clusters:
                            if c[0] == lid and c[1]:
                                parts.append(random.choice(c[1]))
                                break
                        if len(parts) >= 3:
                            break
                joiners = [" ", "——", " / ", "　", " ... ", "、"]
                line = ""
                for j, p in enumerate(parts):
                    if j > 0:
                        line += random.choice(joiners)
                    line += p
                print(line)
            else:
                print(cutup(weighted_fragments, weighted_contents, random.randint(2, 4)))
        else:
            # 通常のカットアップ
            print(cutup(weighted_fragments, weighted_contents, random.randint(2, 4)))

        # 表示速度にゆらぎ
        time.sleep(random.uniform(0.08, 0.25))

    print()
    time.sleep(0.3)
    print("░▒▓ 覚醒 ▓▒░")
    print()


if __name__ == "__main__":
    lines = 20
    if len(sys.argv) > 1:
        try:
            lines = int(sys.argv[1])
        except ValueError:
            pass
    dream_sequence(lines)
