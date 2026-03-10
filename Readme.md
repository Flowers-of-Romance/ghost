# Ghost — LLMのための脳

LLMに脳の仕組みを模した長期記憶を実装する。

## 脳の構造

```
ghost/
├── memory.py           # 記憶システム本体 — 海馬+新皮質
├── Extract.py          # 会話ログからの記憶抽出 — 海馬の取り込み
├── dream.py            # バロウズ式カットアップ夢 — 睡眠中の脳内イメージ
├── interpret_dream.py  # 夢の解釈 — 断片の出典・情動分析
├── autobiography.py    # 自伝的ナラティブ生成 — エピソード記憶の物語化
├── memory_server.py    # embeddingモデル常駐サーバー — 高速化用
├── CLAUDE.md           # Claude Code統合ルール
└── memory.db           # SQLiteデータベース（init後に生成）
```

## セットアップ

```bash
git clone https://github.com/Flowers-of-Romance/ghost.git
cd ghost
pip install sentence-transformers numpy
python memory.py init
```

初回は embeddingモデル（intfloat/multilingual-e5-small, 約90MB）が自動ダウンロードされる。多言語対応、日本語OK。

## 記憶の仕組み

### 生物学的メカニズム

memory.pyは脳の記憶メカニズムを再現する:

| メカニズム | 説明 |
|-----------|------|
| 情動タグ | テキストから情動を自動推定。強い情動の記憶ほど残る |
| 連想リンク | 記憶同士がネットワーク化。芋づる式に想起 |
| 断片保存 | キーワードの束として保存し、想起時に再構成する |
| 減衰と強化 | 時間で薄れ、使うと強まる |
| 再固定化 | 想起するたびに記憶が微妙に変化する |
| 統合・圧縮 | 似た記憶がスキーマ（抽象知識）に統合される |
| 干渉忘却 | 新しい記憶が類似する古い記憶を弱める |
| プライミング | 最近アクセスした記憶が関連記憶の想起を促進 |
| 状態依存記憶 | 気分と一致する情動の記憶が想起されやすい |
| フラッシュバック | 忘却された記憶が確率的に蘇る |
| 予期記憶 | トリガー語に反応して自動リマインド |
| Hebbian learning | 反復された記憶が行動指針に昇格（CLAUDE.mdに自動書込み） |
| 外傷的記憶 | arousalが極端に高い記憶は馴化・統合・減衰に抵抗する |
| 情動重み付き減衰 | 情動が強い記憶ほど忘れにくい（半減期が変動） |

### 検索

sentence-transformersがあれば **ベクトル検索**（384次元、コサイン類似度）。なければ **LIKE検索** にフォールバック。

検索結果はデフォルトで **再構成モード** — 断片+情動+連想リンクから記憶を再構成する。脳のパターン補完と同じ。`--raw`で原文表示、`--fuzzy`で舌先現象（もやもや記憶）表示。

## 使い方

### 基本操作

```bash
# 記憶を追加（情動・重要度は自動推定）
python memory.py add "内容" カテゴリ "出典メモ"

# 連想検索
python memory.py search "ペット"       # ベクトル検索（意味の近さ）
python memory.py search "猫" --raw     # 原文表示
python memory.py search "猫" --fuzzy   # 舌先現象モード

# 自動想起（情動 × 鮮度 × 強化度 × プライミング × 時間帯 × 気分でランク）
python memory.py recall

# 連想の連鎖をたどる
python memory.py chain ID [depth]

# 記憶の詳細
python memory.py detail ID

# 一覧
python memory.py recent [N]
python memory.py all

# 忘却（削除ではなくフラグ）
python memory.py forget ID
```

### 脳機能

```bash
# 忘却された記憶を復活検索
python memory.py resurrect "語"

# メタ記憶（スキーマ）の自動生成
python memory.py schema [--dry-run]

# 間隔反復レビュー
python memory.py review [N]

# 気分状態の設定
python memory.py mood [emotion] [arousal]
python memory.py mood clear

# 海馬リプレイ + 統合・圧縮
python memory.py replay
python memory.py consolidate [--dry-run]

# 予期記憶（トリガーベースのリマインダー）
python memory.py prospect add "トリガー" "アクション"
python memory.py prospect list
python memory.py prospect clear ID

# 手続き化（Hebbian learning）
python memory.py proceduralize [--dry-run]
```

### カテゴリ

| カテゴリ | 用途 | 例 |
|---------|------|-----|
| fact | 事実 | 「猫を飼っている」 |
| episode | 出来事 | 「2025-03-10: メモリ実験を開始」 |
| context | 進行中の文脈（30日で自動失効） | 「無職になりたい」 |
| preference | 好み | 「嫌い」 |
| procedure | 手続き | 「デバッグはまずログを見る」 |
| schema | メタ記憶（自動生成） | 記憶クラスタの要約 |

## 睡眠

脳の夜間バッチ処理。

### 会話ログからの記憶抽出 (Extract.py)

海馬のシミュレーション。Claude Codeの会話ログを読み、「何を覚えるべきか」を自動判断してmemory.dbに保存する。

```bash
# 最新セッションから抽出
python Extract.py

# 全セッションから抽出（初回の大量取り込み）
python Extract.py --all

# ドライラン
python Extract.py --dry-run

# claude.aiの会話テキストから抽出（コピペ対応）
python Extract.py --chat conversation.txt
```

### 夢 (dream.py)

バロウズのカットアップ技法で記憶の断片を表示する。arousalが高い記憶ほど夢に出やすく、リンクが強い記憶同士は一緒に出現する。外傷的記憶は反復する。

```bash
python dream.py        # 20行の夢
python dream.py 30     # 30行の夢
```

### 夢の解釈 (interpret_dream.py)

夢を生成し、各断片の出典を特定。情動テーマ別分析、反復する記憶の検出、意外なつながりの発見を行う。

```bash
python interpret_dream.py
```

出力:
- **断片の出典** — 各断片がどの記憶から来たか
- **情動テーマ別分析** — 驚き・葛藤・決意・洞察・つながり・不安
- **夢に取り憑く記憶** — 繰り返し現れる記憶（未処理の心理的負荷）
- **意外なつながり** — 本来無関係な記憶の異質な結合

### 自伝的記憶 (autobiography.py)

エピソード記憶を時系列で並べ、情動の弧と記憶間のリンクを可視化し、ナラティブとして出力する。

```bash
python autobiography.py
```

## Claude Codeとの統合

`CLAUDE.md` に統合ルールが定義されている。Claude Codeはこのディレクトリで起動すると、会話の開始時に `recall` と `search` で記憶を想起し、終了時に `add` で記憶を保存する。

## データ

```bash
python memory.py stats              # 統計
python memory.py export [filename]  # JSONエクスポート
python memory.py import filename    # JSONインポート
```

## スキーマ

| カラム | 説明 |
|--------|------|
| id | 自動採番 |
| content | 記憶の内容 |
| category | fact / episode / context / preference / procedure / schema |
| importance | 1-5（自動推定） |
| keywords | キーワード断片（JSON） |
| emotions | 情動タグ（JSON） |
| arousal | 覚醒度 |
| created_at | 記録日時 |
| last_accessed | 最後にアクセスした日時 |
| access_count | アクセス回数 |
| forgotten | 忘却フラグ |
| source_conversation | 出典 |
| embedding | ベクトル表現（BLOB, 384次元） |
