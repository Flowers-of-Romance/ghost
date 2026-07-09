#!/usr/bin/env python3
"""
sleep.py - 記憶の睡眠処理を一括実行

全ステップを1プロセスで順次実行し、結果をまとめて出力する。
Claude Codeから1回のBash呼び出しで完了させるためのラッパー。

使い方:
  python sleep.py [dream_lines]   # dream_lines: dream.pyの行数 (default: 30)
"""

import subprocess
import sys
import io
import time
from pathlib import Path
from datetime import datetime, timedelta

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DREAM_LINES = sys.argv[1] if len(sys.argv) > 1 else "30"

# v30: catalog は schema 直後に挿入（cluster_abstract の素材が出揃うため）。
# catalog は navigability の整理（sleep の神経学的 fitness 整理とは目的が別）。
# nap() には入れない — catalog は full sleep のみ。
# ghost記憶v2（2026-07-08）: 埋め込み補完と増分重複整理を夜間の最初に回す。
# backfill は sentence-transformers が要るので ghost venv の python を使う（無ければスキップされ、系は劣化動作で継続）。
_GHOST = Path(__file__).parent
_VENV_PY = _GHOST / ".venv" / "bin" / "python3"
# reconcile Stage B は numpy を使うため、venv があればそちらの python で揃える
_MEMORY_RUNNER = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
_YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

steps = []
if _VENV_PY.exists():
    # sentence-transformers 入りの venv がある時だけ夜間バックフィルを回す
    steps.append(("backfill_embed",
                  [str(_VENV_PY), str(_GHOST / "memory.py"), "backfill-embeddings", "--missing-only"]))

steps += [
    ("reconcile",      [_MEMORY_RUNNER, str(_GHOST / "memory.py"), "reconcile", "--since", _YESTERDAY, "--execute"]),
    ("memo_index",    [sys.executable, "memory.py", "memo", "index"]),
    ("promote",       [sys.executable, "memory.py", "promote"]),
    ("dream",         [sys.executable, "dream.py", DREAM_LINES]),
    ("replay",        [sys.executable, "memory.py", "replay"]),
    ("consolidate",   [sys.executable, "memory.py", "consolidate"]),
    ("schema",        [sys.executable, "memory.py", "schema"]),
    ("mentions",      [sys.executable, "memory.py", "extract-mentions", "--limit", "100"]),
    ("catalog",       [sys.executable, "memory.py", "catalog", "build"]),
    ("proceduralize", [sys.executable, "memory.py", "proceduralize"]),
    ("think",         [sys.executable, "think.py"]),
    ("self_tune",     [sys.executable, "memory.py", "self-tune"]),
    ("meta_memory",   [sys.executable, "memory.py", "meta-memory"]),
    ("stats",         [sys.executable, "memory.py", "stats"]),
]

results = {}
t0 = time.time()

for name, cmd in steps:
    t1 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t1
    output = (r.stdout or "") + (r.stderr or "")
    results[name] = {"output": output.strip(), "time": elapsed, "ok": r.returncode == 0}

total = time.time() - t0

# --- 出力 ---
for name, r in results.items():
    if r["output"]:
        print(r["output"])
    if not r["ok"]:
        print(f"⚠ {name} failed")
    print()

print(f"--- sleep完了: {total:.1f}s ---")
