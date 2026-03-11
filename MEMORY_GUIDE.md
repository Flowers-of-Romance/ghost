# 記憶システム詳細ガイド

サブエージェント用。メインコンテキストには載せない。

## コマンド一覧

```bash
# 基本
python memory.py add "内容" カテゴリ "出典"
python memory.py search "検索語" [--raw] [--fuzzy]
python memory.py recall [N] [--raw]
python memory.py recall --voices [N]   # 内的対話（共感・補完・批判・連想）
python memory.py chain ID [depth]
python memory.py detail ID
python memory.py recent [N]
python memory.py all
python memory.py forget ID

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
