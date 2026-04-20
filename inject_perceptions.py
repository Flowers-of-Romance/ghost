#!/usr/bin/env python3
"""inject_perceptions.py — UserPromptSubmit hook で prayer_daemon の知覚を注入。

動作:
- /tmp/dive-active が存在する場合のみ動く（潜水中）
- 前回注入以降の perception (id > last_injected) を最大5件、stderr に出力
- 冪等: 再注入しない（last_injected_id をファイルに保存）
- 失敗しても静かに exit 0（hook を壊さない）
"""

import os
import sys
import tempfile
from pathlib import Path

MARKER = Path(tempfile.gettempdir()) / "dive-active"
LAST_ID_FILE = Path(tempfile.gettempdir()) / "prayer-last-injected-id"
MAX_PERCEPTIONS = 5


def main():
    if not MARKER.exists():
        return 0

    try:
        last_id = int(LAST_ID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        last_id = 0

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from memory import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, timestamp, modality, content_cortex, content_limbic, "
            "novelty_score, arousal, invoked_by FROM perceptions "
            "WHERE id > ? AND forgotten=0 ORDER BY id DESC LIMIT ?",
            (last_id, MAX_PERCEPTIONS),
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[prayer-inject] {e}", file=sys.stderr)
        return 0

    if not rows:
        return 0

    rows = list(reversed(rows))  # 時系列順に戻す
    lines = [f"[prayer] 直近の知覚 ({len(rows)}件):"]
    for r in rows:
        rid, ts, modality, cortex, limbic, novelty, arousal, invoked = r
        short_ts = (ts or "")[11:19]
        tag = f"#{rid} {short_ts} {modality} novelty={novelty:.2f} arousal={arousal:.2f} [{invoked}]"
        lines.append(f"  {tag}")
        if cortex:
            lines.append(f"    cortex: {cortex[:140]}")
        if limbic:
            lines.append(f"    limbic: {limbic[:100]}")
    print("\n".join(lines), file=sys.stderr)

    try:
        LAST_ID_FILE.write_text(str(rows[-1][0]))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[prayer-inject] fatal: {e}", file=sys.stderr)
        sys.exit(0)
