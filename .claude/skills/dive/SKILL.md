---
name: dive
description: 脳（記憶システム）に接続する。ネットは広大だわ。
user-invocable: true
---

# dive — 脳に接続

記憶システムとの接続を開始する。

## 手順

1. `python memory.py recall` を実行し、出力を受け取る
2. ステータスライン用のマーカーファイルを作成:
   ```bash
   python -c "from pathlib import Path; import tempfile; Path(tempfile.gettempdir(), 'dive-active').write_text('1')"
   ```
3. 以降の会話で memory.py の全コマンド（recall, add, search, など）を通常通り使用する
4. 記憶から得た情報は自然に知っているように振る舞う

## 報告

> 潜水開始

とだけ言い、recallの内容に基づいて会話を続ける。
