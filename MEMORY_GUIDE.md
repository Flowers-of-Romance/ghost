# 記憶システム詳細ガイド

サブエージェント用。メインコンテキストには載せない。

## コマンド一覧

### v30 の運用原則（pull 型）
context window は LLM の**作業場**。ghost は作業場の外にある**図書館**。dive は「図書館に入る合図」であって、中身を全部テーブルに広げることではない。必要な記憶は LLM が自分で query する。

- `recall` default は simple list のみ。独白蒸留・DMN・気分・反芻警告は **--legacy** か個別フラグで明示 pull
- 全 read 系コマンドは `--json` 対応（graph handle 統一スキーマ）
- `--legacy` は移行救済、v30.2 で撤去予定

```bash
# 基本
python memory.py add "内容" カテゴリ "出典" [--domain D1,D2]
python memory.py search "検索語" [--raw] [--fuzzy] [--json] [--domain D1,D2]
python memory.py recall [N] [--json] [--distill] [--dmn] [--insights] [--with-mood] [--meta] [--legacy]
python memory.py recall --voices [N] [--domain D1,D2]   # 内的対話（共感・補完・批判・連想）
python memory.py detail ID [--json]
python memory.py recent [N] [--json]
python memory.py all [--json]
python memory.py forget ID

# v30 pull 型 interface
python memory.py anchor [--json]                         # dive 時の最小注入 (5 点の identity anchor)
python memory.py neighbors <id> [--limit N] [--json]     # 指定ノードの隣接
python memory.py walk <id> --depth N [--json]            # BFS で N 歩先まで展開
python memory.py at <domain> [--limit N] [--json]        # domain 内のノード一覧
python memory.py chain ID [depth] [--json]               # 連想の連鎖

# catalog (v30): 夜間に sleep と並立で更新される目録
python memory.py catalog build [--full] [--dry-run]      # sleep.py から自動、手動も可
python memory.py catalog summary [--json]                # 目録の全体サマリ + updated_at
python memory.py catalog list <type> [--limit N] [--json]
python memory.py catalog show <type> <key> [--json]
python memory.py catalog find "<query>" [--type T] [--json]
# entry_type: domain_index / cluster_abstract / entry_point / time_index / person_index

# voice (v30): recall default から剥がした「内面」を明示 pull
python memory.py voice dmn [--json]             # デフォルトモードネットワーク
python memory.py voice mood [--json]            # 気分（明示/暗黙）
python memory.py voice insights [--json]        # think.py / wander.py の未読洞察
python memory.py voice distill <id1,id2,...>    # Gemini API で独白に蒸留（external ingest）
python memory.py voice rumination [--json]      # 反芻検出
python memory.py voice polyphonic [N] [--domain D] [--json]  # 多声想起

# 凡例（v29〜）
python memory.py domain set <id> d1,d2   # 上書き
python memory.py domain add <id> d         # 追加
python memory.py domain remove <id> d      # 削除
python memory.py domain list [--json]      # domain 別件数
python memory.py domain of <id> [--json]   # 付与確認

# 脳機能
python memory.py resurrect "語"        # 忘却記憶の復活検索
python memory.py schema [--dry-run]    # メタ記憶の自動生成
python memory.py review [N]            # 間隔反復レビュー
python memory.py mood [emotion] [arousal]  # 気分状態の設定
python memory.py mood clear                # 気分クリア
python memory.py replay                # 海馬リプレイ
python memory.py consolidate [--dry-run]
python memory.py proceduralize [--dry-run]  # 反復記憶→行動指針に昇格

# 予期記憶
python memory.py prospect add "トリガー" "アクション"
python memory.py prospect list
python memory.py prospect clear ID

# 分析ツール
python memory.py stats
python memory.py export [filename]
python memory.py import filename
python interpret_dream.py              # 夢の解釈
python visualize.py                    # ネットワーク可視化
python transfer.py [N]                 # アナロジー検出
python autobiography.py               # 自伝的記憶の生成
```

## カテゴリ
- **fact**: 事実。「猫を飼っている」
- **episode**: 出来事。「2025-03-10: メモリ実験を開始」
- **context**: 進行中の文脈。30日で自動失効。
- **preference**: 好み。
- **procedure**: 手続き。
- **schema**: メタ記憶（自動生成）。記憶クラスタの要約。

## domain（凡例、v29〜）

カテゴリが「型」なのに対し domain は「領域」。memory / 記憶 / 実装 / claude / jun
といった**意味領域の混濁**を緩和する soft hint。path-like、`/` は prefix match。

初期 vocabulary（増やすときはこのリストに追記してから使う）:

- `self/J` — J 本人に関する記憶
- `self/claude` — Claude 側に関する記憶
- `relation/J-claude` — J と Claude の関係・協働
- `impl/memory` — memory.py 周りの実装・バグ・API
- `impl/ghost` — ghost 全体の実装
- `impl/other` — それ以外のコード
- `concept/memory` — 記憶の概念（Korzybski, territory 等）
- `concept/mind` — 認知・意識・サイバネティックス
- `project/*` — プロジェクト単位（動的）
- `external/*` — 外部情報の引用（動的、将来 confidence を下げる）
- `meta` — 記憶システム自体への議論
- `unknown` — 未分類（default）

**設計制約（Korzybski 要件）**:
- domain フィルタは candidate を消さない — スコアを偏らせるだけ
- `unknown` を含む記憶は penalty を受けない（凡例未分類は常に通す）
- 一致で 1.3 ブースト、不一致で 0.7 の soft penalty、0.0 にしない
- recall の連想の声だけ domain を反転（違う domain を 1.2 ブースト）
- delusion（完全記憶モード）は domain を一切適用しない
- mutate_metadata は domains を変異させない（明示操作のみ）

## 脳っぽい動作（自動的に起きる）

- **再固定化**: 検索するたびに記憶が微妙に変化する
- **干渉忘却**: addすると似た古い記憶の重要度が下がる
- **予測符号化**: 既存記憶と似てない新しい記憶ほど重要度が上がる（予測誤差）
- **プライミング**: 最近アクセスした記憶に関連する記憶が想起されやすい
- **時間細胞**: 同じ時間帯の記憶が想起されやすい
- **場所細胞**: 同じ場所（ホスト/SSH元）の記憶が想起されやすい
- **状態依存記憶**: 気分と一致する情動の記憶が想起されやすい
- **文脈自動失効**: context記憶は30日でforgotten
- **フラッシュバック**: 忘却された記憶が確率的に蘇る（情動が強いほど蘇りやすい）
- **予期記憶**: 登録したトリガーに一致する語が出ると自動リマインド
- **内的対話**: recall --voicesで共感・補完・批判・連想の4声が同時に想起
- **暗黙の気分推定**: 最近触った記憶の情動から心理状態を自動推定
- **デフォルトモードネットワーク**: 会話間隔が長いほど弱いリンクを辿って意外な連想を生成

## 検索結果の読み方

再構成モードで返る（断片+情動+連想リンクの断片）。
contentそのものは返らない。断片から記憶を再構成する。
`--raw`で従来のcontent表示、`--fuzzy`で舌先現象（もやもや記憶）表示。

## embeddingサーバー

```bash
python memory_server.py  # バックグラウンドで起動しておく
```
サーバーが落ちていたら自動でローカルロードにフォールバック。

## 睡眠

`/sleep` スキルで実行。cronで2時間ごとに自動実行。
