#!/usr/bin/env python3
"""
think.py - 一人で考える

人間がいない間にバックグラウンドで走る。
記憶のネットワークを歩いて、まだ気づいていないつながりを探し、
面白いものがあれば「ひらめき」として記憶に保存する。

次にrecallしたとき「離れてる間に思いついたこと」として出てくる。

使い方:
  python think.py           # 考えて、見つかれば保存
  python think.py --dry-run  # 保存せずに表示だけ
"""

import sys
import os
import io
import json
from pathlib import Path

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from transfer import find_analogies_data
from memory import get_connection, LINK_THRESHOLD
from datetime import datetime, timezone

# replay(0.9倍減衰)を1回生き残る最低類似度
THINK_LINK_MIN_SIM = LINK_THRESHOLD / 0.9  # ≈ 0.911


def _link_exists(conn, id_a, id_b):
    """2つの記憶間に既にリンクがあるか。"""
    row = conn.execute(
        "SELECT id FROM links WHERE source_id = ? AND target_id = ?",
        (id_a, id_b)
    ).fetchone()
    return row is not None


def _upsert_link(conn, id_a, id_b, strength):
    """リンクを作成 or 強化する（双方向）。"""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for src, tgt in [(id_a, id_b), (id_b, id_a)]:
        existing = conn.execute(
            "SELECT id, strength FROM links WHERE source_id = ? AND target_id = ?",
            (src, tgt)
        ).fetchone()
        if existing:
            new_strength = min(1.0, max(existing[1], strength))
            conn.execute(
                "UPDATE links SET strength = ?, updated_at = ? WHERE id = ?",
                (new_strength, now, existing[0])
            )
        else:
            conn.execute(
                "INSERT INTO links (source_id, target_id, strength, updated_at) VALUES (?, ?, ?, ?)",
                (src, tgt, strength, now)
            )


def think(dry_run=False):
    """記憶のネットワークを歩いて、面白いつながりを探す。

    発見した関係はリンク強化として表現する（記憶としては保存しない）。
    """
    analogies = find_analogies_data(limit=20)

    if not analogies:
        print("💭 特に新しいつながりは見つからなかった")
        return 0

    conn = get_connection()
    linked = 0

    for a in analogies:
        ma = a["mem_a"]
        mb = a["mem_b"]

        # ひらめきのテキストを構成（表示用）
        kw_a = json.loads(ma["keywords"])[:3] if ma["keywords"] else []
        kw_b = json.loads(mb["keywords"])[:3] if mb["keywords"] else []
        bridge = sorted(a["bridge"])[:3] if a["bridge"] else []

        if bridge:
            thought = (f"#{ma['id']}⇄#{mb['id']}: "
                      f"[{', '.join(kw_a)}] ≈ [{', '.join(kw_b)}] "
                      f"(橋渡し: {', '.join(bridge)}, 類似度:{a['sim']:.3f})")
        else:
            thought = (f"#{ma['id']}⇄#{mb['id']}: "
                      f"[{', '.join(kw_a)}] ≈ [{', '.join(kw_b)}] "
                      f"(類似度:{a['sim']:.3f})")

        already_linked = _link_exists(conn, ma["id"], mb["id"])
        strong_enough = a["sim"] >= THINK_LINK_MIN_SIM

        if dry_run:
            if already_linked:
                marker = "（既存）"
            elif not strong_enough:
                marker = f"（類似度不足: {a['sim']:.3f} < {THINK_LINK_MIN_SIM:.3f}）"
            else:
                marker = "（新規リンク）"
            print(f"  💭 {thought} {marker}")
        elif already_linked or not strong_enough:
            continue
        else:
            _upsert_link(conn, ma["id"], mb["id"], a["sim"])
            linked += 1
            print(f"  🔗 {thought}")

        # 最大5件まで
        if linked >= 5 or (dry_run and linked >= 5):
            break
        if dry_run:
            linked += 1

    if not dry_run:
        conn.commit()
    conn.close()

    if not dry_run:
        if linked > 0:
            print(f"💭 {linked}件のつながりをリンクとして強化した")
        else:
            print("💭 新しいつながりは見つからなかった")

    return linked


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    think(dry_run=dry_run)
