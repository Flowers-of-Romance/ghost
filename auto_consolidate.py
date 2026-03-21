#!/usr/bin/env python3
"""
auto_consolidate.py - 会話終了時の自動記憶固定化

Stop hookで呼ばれる。promote(サンプリング少なめ) + nap(replay + consolidate)。
軽量に保つ。夢もカットアップも出さない。
"""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from memory import promote_turns, nap

# 少数だけリプレイ（会話終了のたびに走るから控えめに）
promote_turns(days=3, sample_size=5)
nap()
