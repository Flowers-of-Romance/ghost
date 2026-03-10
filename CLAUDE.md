# CLAUDE.md - 記憶するClaude

## 記憶システム

このプロジェクトには脳の仕組みを模した記憶システムがある。
会話をまたいで文脈を保持する。必ず使うこと。

### 会話の開始時

```bash
# 1. 重要な記憶を自動想起（情動 × 鮮度 × 強化度 × プライミング × 時間帯 × 気分でランク）
python memory.py recall

# 2. ユーザーの発言に関連する記憶を連想検索
python memory.py search "キーワードや概念"

# 3. 気になる記憶があれば連想の連鎖をたどる
python memory.py chain ID
```

検索結果は**再構成モード**で返る（断片+情動+連想リンクの断片）。
contentそのものは返らない。断片から記憶を再構成して応答する。
これは脳のパターン補完と同じ——記憶は保存されたテキストではなく、
断片から毎回生成されるパターン。完全に正確でなくていい。
「以前の記憶によると」とは言わない。

`--raw`で従来のcontent表示、`--fuzzy`で舌先現象（もやもや記憶）表示。

### 会話の終了時

覚えるべきことを保存する:

```bash
python memory.py add "内容" カテゴリ "出典メモ"
```

重要度は自動。情動（驚き、葛藤、決意、洞察、つながり、不安）を
テキストとトーン（!の数、?、大文字、文の長さ、...）から検出して決まる。

#### カテゴリ
- **fact**: 事実。「猫を飼っている」
- **episode**: 出来事。「2025-03-10: メモリ実験を開始」
- **context**: 進行中の文脈。30日で自動失効。「無職になりたい」
- **preference**: 好み。「嫌い」
- **procedure**: 手続き。「デバッグはまずログを見る」
- **schema**: メタ記憶（自動生成）。記憶クラスタの要約。

### 脳っぽい動作（自動的に起きる）

- **再固定化**: 検索するたびに記憶が微妙に変化する
- **干渉忘却**: addすると似た古い記憶の重要度が下がる
- **プライミング**: 最近アクセスした記憶に関連する記憶が想起されやすい
- **時間細胞**: 同じ時間帯の記憶が想起されやすい
- **状態依存記憶**: 気分と一致する情動の記憶が想起されやすい
- **文脈自動失効**: context記憶は30日でforgotten
- **フラッシュバック**: 忘却された記憶が確率的に蘇る（情動が強いほど蘇りやすい）
- **予期記憶**: 登録したトリガーに一致する語が出ると自動リマインド

### 睡眠（リプレイ+統合）

`/sleep` スキルで実行。cronで2時間ごとに自動実行。

### embeddingサーバー

```bash
python memory_server.py  # バックグラウンドで起動しておく
```
サーバーが落ちていたら自動でローカルロードにフォールバック。

### コマンド一覧

```bash
# 基本
python memory.py add "内容" カテゴリ "出典"
python memory.py search "検索語" [--raw] [--fuzzy]
python memory.py recall [N] [--raw]
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
