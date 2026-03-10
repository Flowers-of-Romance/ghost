#!/usr/bin/env python3
"""
autobiography.py - エピソード記憶から自伝的ナラティブを生成する

使い方:
  python autobiography.py
"""

import sqlite3
import sys
import os
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Windows cp932 で emoji が出力できない問題を回避
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))

EMOTION_EMOJI = {
    "surprise": "😲",
    "conflict": "⚡",
    "determination": "🔥",
    "insight": "💎",
    "connection": "🤝",
    "anxiety": "😰",
}

EMOTION_JA = {
    "surprise": "驚き",
    "conflict": "葛藤",
    "determination": "決意",
    "insight": "洞察",
    "connection": "つながり",
    "anxiety": "不安",
}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_episodes(conn):
    """非忘却のエピソード記憶を時系列順に取得。"""
    return conn.execute(
        "SELECT * FROM memories WHERE category = 'episode' AND forgotten = 0 ORDER BY created_at"
    ).fetchall()


def fetch_contexts(conn):
    """非忘却のコンテキスト記憶を取得。"""
    return conn.execute(
        "SELECT * FROM memories WHERE category = 'context' AND forgotten = 0 ORDER BY created_at"
    ).fetchall()


def fetch_links(conn):
    """全リンクを取得し、双方向の辞書として返す。"""
    rows = conn.execute(
        "SELECT source_id, target_id, strength FROM links"
    ).fetchall()
    links = {}
    for r in rows:
        links[(r["source_id"], r["target_id"])] = r["strength"]
        links[(r["target_id"], r["source_id"])] = r["strength"]
    return links


def parse_date(created_at_str):
    """created_at文字列からdatetimeを得る。"""
    try:
        return datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return datetime(2000, 1, 1)


def group_by_period(episodes):
    """エピソードを日付でグループ化する。

    同日のものは日単位、同週のものは週単位、それ以外は月単位。
    ただし実用上は日単位グループを基本とし、表示時に隣接判定する。
    """
    groups = defaultdict(list)
    for ep in episodes:
        dt = parse_date(ep["created_at"])
        day_key = dt.strftime("%Y-%m-%d")
        groups[day_key].append(ep)
    return dict(sorted(groups.items()))


def format_emotions(emotions_json):
    """情動リストをタグ文字列に変換。"""
    try:
        emotions = json.loads(emotions_json) if isinstance(emotions_json, str) else emotions_json
    except (json.JSONDecodeError, TypeError):
        emotions = []
    if not emotions:
        return ""
    tags = [EMOTION_JA.get(e, e) for e in emotions]
    return "・".join(tags)


def format_emotion_emoji_sequence(emotions_json):
    """情動リストを絵文字+日本語名の列に変換。"""
    try:
        emotions = json.loads(emotions_json) if isinstance(emotions_json, str) else emotions_json
    except (json.JSONDecodeError, TypeError):
        emotions = []
    if not emotions:
        return ""
    parts = []
    for e in emotions:
        emoji = EMOTION_EMOJI.get(e, "·")
        ja = EMOTION_JA.get(e, e)
        parts.append(f"{emoji}{ja}")
    return " → ".join(parts)


def find_linked_pairs(episodes_in_group, all_links):
    """グループ内のエピソード間のリンクを探す。"""
    ids = [ep["id"] for ep in episodes_in_group]
    pairs = []
    seen = set()
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            key = (min(id_a, id_b), max(id_a, id_b))
            if key not in seen and (id_a, id_b) in all_links:
                seen.add(key)
                pairs.append((id_a, id_b, all_links[(id_a, id_b)]))
    return pairs


def find_cross_group_links(ep, all_episode_ids, all_links):
    """あるエピソードから他グループのエピソードへのリンクを探す。"""
    result = []
    for other_id in all_episode_ids:
        if other_id != ep["id"] and (ep["id"], other_id) in all_links:
            result.append((other_id, all_links[(ep["id"], other_id)]))
    return result


def build_emotion_arc(grouped_episodes):
    """日ごとの感情の軌跡を構築する。"""
    arc = {}
    for day, episodes in grouped_episodes.items():
        day_emotions = []
        for ep in episodes:
            try:
                emotions = json.loads(ep["emotions"]) if isinstance(ep["emotions"], str) else ep["emotions"]
            except (json.JSONDecodeError, TypeError):
                emotions = []
            for e in emotions:
                if e not in day_emotions:
                    day_emotions.append(e)
        if day_emotions:
            arc[day] = day_emotions
    return arc


def generate():
    conn = get_connection()

    episodes = fetch_episodes(conn)
    contexts = fetch_contexts(conn)
    all_links = fetch_links(conn)

    if not episodes and not contexts:
        print("記憶がありません。まず memory.py add でエピソードを追加してください。")
        return

    print("═══ 自伝的記憶 ═══")
    print()

    # コンテキスト記憶を背景として表示
    if contexts:
        print("【背景】")
        for ctx in contexts:
            emo_str = format_emotions(ctx["emotions"])
            tag = f"[{emo_str}] " if emo_str else ""
            print(f"  {tag}{ctx['content']}")
        print()

    # エピソードをグループ化
    grouped = group_by_period(episodes)
    all_episode_ids = set(ep["id"] for ep in episodes)

    # エピソードID→記憶の逆引き
    id_to_ep = {ep["id"]: ep for ep in episodes}

    # 表示済みリンクを追跡（重複防止）
    shown_links = set()

    for day, day_episodes in grouped.items():
        # 日付ヘッダー
        print(f"▸ {day}")

        for ep in day_episodes:
            emo_str = format_emotions(ep["emotions"])
            tag = f"[{emo_str}] " if emo_str else ""
            print(f"  {tag}{ep['content']}")

        # グループ内リンク
        intra_pairs = find_linked_pairs(day_episodes, all_links)
        for id_a, id_b, strength in intra_pairs:
            key = (min(id_a, id_b), max(id_a, id_b))
            if key not in shown_links:
                shown_links.add(key)
                print(f"    → #{id_a} と #{id_b} の記憶は連想で結ばれている ({strength:.2f})")

        # 他の日へのリンク
        for ep in day_episodes:
            cross_links = find_cross_group_links(ep, all_episode_ids, all_links)
            for other_id, strength in cross_links:
                key = (min(ep["id"], other_id), max(ep["id"], other_id))
                if key not in shown_links:
                    shown_links.add(key)
                    other = id_to_ep.get(other_id)
                    if other:
                        other_summary = other["content"][:40]
                        print(f"    → 「{other_summary}」と連想で結ばれている ({strength:.2f})")

        print()

    # 感情の軌跡
    arc = build_emotion_arc(grouped)
    if arc:
        print("━━━ 感情の軌跡 ━━━")
        for day, emotions in arc.items():
            # 日付を短縮形に
            try:
                dt = datetime.fromisoformat(day)
                short_date = dt.strftime("%m/%d")
            except ValueError:
                short_date = day
            emoji_seq = " → ".join(
                f"{EMOTION_EMOJI.get(e, '·')}{EMOTION_JA.get(e, e)}" for e in emotions
            )
            print(f"  {short_date}: {emoji_seq}")
        print()

    conn.close()


if __name__ == "__main__":
    generate()
