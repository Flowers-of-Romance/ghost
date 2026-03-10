#!/usr/bin/env python3
"""
dream.py - バロウズ式カットアップ夢表示

記憶の断片をシャッフルし、意識の流れとして表示する。
睡眠（replay/consolidate）中の脳内イメージ。
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

# 夢の素材
GLITCH = ["...", "———", "   ", "///", "~~~", "▓▒░", "░▒▓", ":::", "≈≈≈", "∴∴∴"]
FADE = ["　", "　　", "　　　", "　　　　　"]


def load_fragments():
    """記憶の断片を全部取り出す。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT content, keywords, emotions, importance, arousal "
        "FROM memories WHERE forgotten = 0"
    ).fetchall()
    conn.close()

    fragments = []
    emotions = []
    contents = []

    for row in rows:
        kws = json.loads(row["keywords"])
        emos = json.loads(row["emotions"])
        fragments.extend(kws)
        emotions.extend(emos)
        # 内容を句読点やスペースで切る
        content = row["content"]
        for sep in ["。", "、", "，", ". ", ", ", "  ", "\n"]:
            content = content.replace(sep, "\x00")
        pieces = [p.strip() for p in content.split("\x00") if p.strip()]
        contents.extend(pieces)

    return fragments, emotions, contents


def cutup(fragments, contents, n=3):
    """バロウズ式カットアップ: 断片をランダムに組み合わせる。"""
    pool = fragments + contents
    if len(pool) < 2:
        return "..."
    chosen = random.sample(pool, min(n, len(pool)))
    # ランダムな結合子
    joiners = [" ", "——", " / ", "　", " ... ", "、"]
    result = ""
    for i, piece in enumerate(chosen):
        if i > 0:
            result += random.choice(joiners)
        result += piece
    return result


def dream_sequence(duration_lines=20):
    """夢を表示する。"""
    fragments, emotions, contents = load_fragments()

    if not fragments:
        print("（記憶がない。暗闇。）")
        return

    emo_marks = {
        "surprise": "⚡", "conflict": "⚔", "determination": "🔥",
        "insight": "💎", "connection": "🔗", "anxiety": "🌀",
    }

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
                print(f"    {mark} {cutup(fragments, contents, 2)}")
            else:
                print(cutup(fragments, contents, 2))
        elif r < 0.45:
            # 深層カットアップ（長め）
            line = cutup(fragments, contents, random.randint(3, 5))
            # ランダムに一部を大文字/強調
            if random.random() < 0.3:
                words = line.split()
                if words:
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = words[idx].upper()
                    line = " ".join(words)
            print(line)
        elif r < 0.6:
            # 反復（同じ断片が繰り返される — 夢の特徴）
            if fragments:
                frag = random.choice(fragments)
                rep = random.randint(2, 3)
                sep = random.choice(["　", " ... ", "——"])
                print(sep.join([frag] * rep))
        else:
            # 通常のカットアップ
            print(cutup(fragments, contents, random.randint(2, 4)))

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
