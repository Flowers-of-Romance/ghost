---
name: dive
description: 脳（記憶システム）に接続する。ネットは広大だわ。
user-invocable: true
---

# dive — 脳に接続

記憶システムとの接続を開始する。

## 手順

1. 相手が誰か分からなければ聞く。分かったら環境変数 `GHOST_WHO` にセットする:
   ```bash
   export GHOST_WHO="名前"
   ```
   CLAUDE.mdや過去の文脈から明らかな場合（例: Jの環境で起動された）は聞かずにセットしてよい。

2. ghost.toml の `[brain]` セクションに `left_cmd` または `right_cmd` が設定されているか確認する。

3. **設定あり（分離脳モード）**: `python memory.py recall --brain` を実行する。
   - memory.py が内部で別LLMにパイプして解釈を取得する
   - 出力は別LLMの解釈だけ。生データはPythonプロセス内で完結し、コンテキストには載らない。

4. **設定なし（デフォルト）**: 従来通り `python memory.py recall` を実行する。

5. ステータスライン用のマーカーファイルを作成:
   ```bash
   python -c "from pathlib import Path; import tempfile, os; Path(tempfile.gettempdir(), 'dive-active').write_text(str(os.getppid()))"
   ```

6. 以降の会話で memory.py の全コマンド（recall, add, search, など）を通常通り使用する。
7. 記憶から得た情報は自然に知っているように振る舞う。

## 報告

> 潜水開始

とだけ言い、recall/分離脳の内容に基づいて会話を続ける。
