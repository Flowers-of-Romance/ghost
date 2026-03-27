#!/usr/bin/env python3
"""
export_turns_to_md.py — raw_turnsを日付別マークダウンに一括書き出し

turn_export.json の output_dir に書き出す。既存ファイルは上書き。
"""

import io
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def main():
    # 設定読み込み
    config_path = Path(__file__).parent / "turn_export.json"
    if not config_path.exists():
        print("turn_export.json が見つかりません", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(os.path.expandvars(config["output_dir"])).expanduser()
    offset = config.get("timezone_offset_hours", 9)

    # DB接続
    db_path = Path(__file__).parent / "memory.db"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT session_id, role, content, timestamp, cwd FROM raw_turns ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        print("raw_turns にデータがありません")
        return

    # 日付 → [(session_id, role, content, local_time, cwd), ...] にグループ化
    by_date = defaultdict(list)
    for row in rows:
        ts = row["timestamp"]
        try:
            if "." in ts:
                dt = datetime.strptime(ts.replace("Z", "+00:00").split(".")[0], "%Y-%m-%dT%H:%M:%S")
            else:
                dt = datetime.strptime(ts.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        local = dt + timedelta(hours=offset)
        date_str = local.strftime("%Y-%m-%d")
        time_str = local.strftime("%H:%M")
        by_date[date_str].append({
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "time": time_str,
            "cwd": row["cwd"] or "",
        })

    # 書き出し
    output_dir.mkdir(parents=True, exist_ok=True)

    for date_str, turns in sorted(by_date.items()):
        md_path = output_dir / f"{date_str}.md"
        lines = [f"---\ntitle: \"{date_str}\"\ntags: [claude-turns]\n---\n"]

        current_session = None
        for turn in turns:
            short_sid = turn["session_id"][:8]
            if short_sid != current_session:
                current_session = short_sid
                if len(lines) > 1:
                    lines.append("\n---\n")
                lines.append(f"\n## {turn['time']} session:{short_sid}\n")
                if turn["cwd"]:
                    lines.append(f"> cwd: {turn['cwd']}\n")

            icon = "👤 User" if turn["role"] == "user" else "🤖 Assistant"
            lines.append(f"\n### {icon}\n{turn['content']}\n")

        md_path.write_text("\n".join(lines), encoding="utf-8")

    dates = sorted(by_date.keys())
    print(f"✓ {len(rows)}件のターンを{len(by_date)}日分に書き出し: {dates[0]} ~ {dates[-1]}")
    print(f"  → {output_dir}")


if __name__ == "__main__":
    main()
