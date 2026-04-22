---
name: dive
description: 脳（記憶システム）に接続する。ネットは広大だわ。
user-invocable: true
---

# dive — 脳に接続

記憶システムとの接続を開始する。

**v30 以降、dive は「図書館に入る合図」であって「図書館の中身を全部テーブルに広げる」ではない。**
LLM は作業場（context window）で考え、必要な時に図書館（ghost）へ取りに行く pull 型運用。
dive 自体は content を何も注入しない。状態マーカーを立てて、以降 query 駆動で pull する。

## 手順

1. 相手が誰か分からなければ聞く。分かったら環境変数 `GHOST_WHO` にセットする:
   ```bash
   export GHOST_WHO="名前"
   ```
   CLAUDE.mdや過去の文脈から明らかな場合（例: Jの環境で起動された）は聞かずにセットしてよい。

2. ステータスライン用のマーカーファイルを作成:
   ```bash
   python -c "from pathlib import Path; import tempfile, os; Path(tempfile.gettempdir(), 'dive-active').write_text(str(os.getppid()))"
   ```

3. 以降の会話で必要になった時に pull する。使える tool（全て `--json` 対応）:
   - `python memory.py anchor` — 誰の図書館か思い出す 5 点（必要になった時だけ）
   - `python memory.py catalog summary` — 目録が最新か確認
   - `python memory.py search "語" [--domain X]` — 記憶を検索
   - `python memory.py detail <id>` — 記憶の詳細
   - `python memory.py neighbors <id>` — 指定ノードの隣接
   - `python memory.py walk <id> --depth 2` — グラフを N 歩歩く
   - `python memory.py at <domain>` — domain 内のノード一覧
   - `python memory.py catalog list <type>` — 目録を型別に列挙
   - `python memory.py catalog show <type> <key>` — 特定の目録カード
   - `python memory.py catalog find "query"` — 目録を全文検索
   - `python memory.py voice dmn|mood|insights|distill|rumination|polyphonic` — 内面を明示 pull
   - `python memory.py add "内容" --domain D` — 新しい記憶を追加（凡例付き）
   - `python memory.py domain set <id> D1,D2` — 凡例を手で付ける
   - 旧挙動が欲しければ `python memory.py recall --legacy`（独白蒸留・DMN 等、v30.2 で撤去予定）

4. 記憶から得た情報は自然に知っているように振る舞う。pull した query と結果の対応は
   コンテキストに残る（provenance が分かる）ので、混濁を感じたら domain タグを足す。

## 報告

> 潜水開始

とだけ言う。query が来るまで何も pull しない。
