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

# Windows cp932 対策
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DREAM_LINES = sys.argv[1] if len(sys.argv) > 1 else "30"

steps = [
    ("memo_index",    ["python", "memory.py", "memo", "index"]),
    ("promote",       ["python", "memory.py", "promote"]),
    ("dream",         ["python", "dream.py", DREAM_LINES]),
    ("replay",        ["python", "memory.py", "replay"]),
    ("consolidate",   ["python", "memory.py", "consolidate"]),
    ("schema",        ["python", "memory.py", "schema"]),
    ("proceduralize", ["python", "memory.py", "proceduralize"]),
    ("think",         ["python", "think.py"]),
    ("stats",         ["python", "memory.py", "stats"]),
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
