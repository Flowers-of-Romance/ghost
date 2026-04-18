#!/usr/bin/env python3
"""
transfer.py - 記憶間のアナロジーと領域横断的つながりを検出する

「適度に似ているが異なる領域にある記憶」は、転用可能な知識や
隠れたアナロジーを示唆する。木村敏の「あいだ」とPSMのつながりのように、
異分野間の意外な共鳴を発見するためのツール。

使い方:
  python transfer.py [limit]   # デフォルト10件
"""

import sys
import os
import io
import json
import struct

# Windows cp932 で出力できない文字を回避
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# memory.py と同じDB設定を使う
from pathlib import Path
import sqlite3
import numpy as np

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))

# アナロジー検出の類似度範囲
SIM_LOW = 0.70
SIM_HIGH = 0.85

EMOTION_EMOJI = {
    "surprise": "😲",
    "conflict": "⚡",
    "determination": "🔥",
    "insight": "💎",
    "connection": "🤝",
    "anxiety": "😰",
}

CATEGORY_LABEL = {
    "fact": "事実",
    "episode": "出来事",
    "context": "文脈",
    "preference": "好み",
    "procedure": "手順",
    "schema": "スキーマ",
}


def bytes_to_vec(b):
    n = len(b) // 4
    return np.array(struct.unpack(f'{n}f', b), dtype=np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_active_memories(conn):
    """忘却されていない、embeddingを持つ記憶をすべて読み込む。"""
    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()
    return rows


def dominant_emotion(emotions_json):
    """JSON文字列から主要な情動を返す。空なら'neutral'。"""
    emotions = json.loads(emotions_json) if emotions_json else []
    return emotions[0] if emotions else "neutral"


def compute_interestingness(sim, cat_a, cat_b, emo_a, emo_b):
    """
    面白さスコアを計算する。
    - 類似度が中間域（0.77付近）に近いほど高い
    - カテゴリが異なると加点
    - 主要情動が異なると加点
    """
    # 類似度の「中間らしさ」: 0.77を頂点とする三角関数的スコア
    mid = (SIM_LOW + SIM_HIGH) / 2.0
    half_range = (SIM_HIGH - SIM_LOW) / 2.0
    sim_score = 1.0 - abs(sim - mid) / half_range  # 0..1

    # カテゴリ差異ボーナス
    cat_bonus = 0.4 if cat_a != cat_b else 0.0

    # 情動差異ボーナス
    emo_bonus = 0.3 if emo_a != emo_b else 0.0

    return sim_score + cat_bonus + emo_bonus


def find_analogies_data(limit=10):
    """アナロジー候補をデータとして返す。"""
    conn = get_connection()
    memories = load_active_memories(conn)
    n = len(memories)

    if n < 2:
        conn.close()
        return []

    vecs = [bytes_to_vec(m["embedding"]) for m in memories]

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vecs[i], vecs[j])
            if SIM_LOW <= sim <= SIM_HIGH:
                cat_a = memories[i]["category"]
                cat_b = memories[j]["category"]
                emo_a = dominant_emotion(memories[i]["emotions"])
                emo_b = dominant_emotion(memories[j]["emotions"])
                if cat_a != cat_b or emo_a != emo_b:
                    score = compute_interestingness(sim, cat_a, cat_b, emo_a, emo_b)
                    kw_a = set(json.loads(memories[i]["keywords"]) if memories[i]["keywords"] else [])
                    kw_b = set(json.loads(memories[j]["keywords"]) if memories[j]["keywords"] else [])
                    bridge = kw_a & kw_b
                    candidates.append({
                        "mem_a": memories[i],
                        "mem_b": memories[j],
                        "sim": sim,
                        "score": score,
                        "bridge": bridge,
                    })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    conn.close()
    return candidates[:limit]


def find_analogies(limit=10):
    """アナロジー候補となる記憶ペアを検出する。"""
    conn = get_connection()
    memories = load_active_memories(conn)
    n = len(memories)

    if n < 2:
        print("記憶が2件未満です。アナロジー検出には記憶が必要です。")
        return

    # embedding をまとめて取得
    vecs = []
    for m in memories:
        vecs.append(bytes_to_vec(m["embedding"]))

    # 全ペアの類似度を計算し、条件に合うものを収集
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vecs[i], vecs[j])
            if SIM_LOW <= sim <= SIM_HIGH:
                cat_a = memories[i]["category"]
                cat_b = memories[j]["category"]
                emo_a = dominant_emotion(memories[i]["emotions"])
                emo_b = dominant_emotion(memories[j]["emotions"])

                # カテゴリ違い OR 情動違い のどちらかを満たす
                if cat_a != cat_b or emo_a != emo_b:
                    score = compute_interestingness(sim, cat_a, cat_b, emo_a, emo_b)
                    candidates.append((i, j, sim, score))

    # 面白さでソート
    candidates.sort(key=lambda x: x[3], reverse=True)
    candidates = candidates[:limit]

    if not candidates:
        print("アナロジー候補が見つかりませんでした。")
        print(f"  (類似度 {SIM_LOW}〜{SIM_HIGH} の異領域ペアを探索しました)")
        print(f"  記憶数: {n}件")
        return

    # 出力
    print(f"=== アナロジー検出 ({len(candidates)}件) ===")
    print(f"    類似度 {SIM_LOW}〜{SIM_HIGH} の異領域ペアを面白さ順に表示\n")

    for rank, (i, j, sim, score) in enumerate(candidates, 1):
        ma = memories[i]
        mb = memories[j]

        kw_a = set(json.loads(ma["keywords"]) if ma["keywords"] else [])
        kw_b = set(json.loads(mb["keywords"]) if mb["keywords"] else [])
        bridge = kw_a & kw_b          # 橋渡し概念
        diverge_a = kw_a - kw_b       # Aにだけある
        diverge_b = kw_b - kw_a       # Bにだけある

        emo_a_list = json.loads(ma["emotions"]) if ma["emotions"] else []
        emo_b_list = json.loads(mb["emotions"]) if mb["emotions"] else []
        emo_a_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emo_a_list) if emo_a_list else "·"
        emo_b_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emo_b_list) if emo_b_list else "·"

        cat_a_label = CATEGORY_LABEL.get(ma["category"], ma["category"])
        cat_b_label = CATEGORY_LABEL.get(mb["category"], mb["category"])

        print(f"--- #{rank}  面白さ: {score:.2f}  類似度: {sim:.3f} ---")
        print(f"  A) #{ma['id']} [{cat_a_label}] {emo_a_str}")
        print(f"     {ma['content'][:100]}")
        print(f"  B) #{mb['id']} [{cat_b_label}] {emo_b_str}")
        print(f"     {mb['content'][:100]}")

        if bridge:
            print(f"  橋渡し概念: {', '.join(sorted(bridge)[:10])}")
        else:
            print(f"  橋渡し概念: (なし — 暗黙の類似性)")

        if diverge_a or diverge_b:
            da = ', '.join(sorted(diverge_a)[:6]) if diverge_a else "(なし)"
            db = ', '.join(sorted(diverge_b)[:6]) if diverge_b else "(なし)"
            print(f"  Aの独自概念: {da}")
            print(f"  Bの独自概念: {db}")

        print()

    conn.close()


def main():
    limit = 10
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"使い方: python transfer.py [limit]")
            print(f"  limit: 表示件数（デフォルト10）")
            sys.exit(1)

    find_analogies(limit)


if __name__ == "__main__":
    main()
