# Ghost — LLMのための脳

LLMに脳の仕組みを模した長期記憶を実装する。

## 脳の構造

```
ghost/
├── memory.py              # 記憶システム本体 — 海馬+新皮質
├── Extract.py             # 会話ログからの記憶抽出 — 海馬の取り込み
├── dream.py               # バロウズ式カットアップ夢 — 睡眠中の脳内イメージ
├── interpret_dream.py     # 夢の解釈 — 断片の出典・情動分析
├── autobiography.py       # 自伝的ナラティブ生成 — エピソード記憶の物語化
├── memory_server.py       # embeddingモデル常駐サーバー — 高速化用
├── memory_sync_server.py  # P2P記憶同期サーバー — 複数端末間の記憶共有
├── CLAUDE.md              # Claude Code統合ルール
├── GEMINI.md              # Gemini CLI統合ルール
├── MEMORY_GUIDE.md        # 記憶システム詳細ガイド（サブエージェント用）
├── .claude/skills/        # Claude Code用スキル（dive/surface/sleep）
├── .gemini/skills/        # Gemini CLI用スキル（dive/surface/sleep）
└── memory.db              # SQLiteデータベース（init後に生成）
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
| **予測符号化** | **既存記憶との非類似度=予測誤差。誤差が大きいほど重要度が上がる** |
| プライミング | 最近アクセスした記憶が関連記憶の想起を促進 |
| 状態依存記憶 | 気分と一致する情動の記憶が想起されやすい |
| **場所細胞** | **同じ場所（ホスト名/SSH接続元）の記憶が想起されやすい** |
| 時間細胞 | 同じ時間帯の記憶が想起されやすい |
| フラッシュバック | 忘却された記憶が確率的に蘇る |
| 予期記憶 | トリガー語に反応して自動リマインド |
| 手続き化 | 反復された記憶が行動指針に昇格（LEARNED.mdに書出し） |
| 外傷的記憶 | arousalが極端に高い記憶は馴化・統合・減衰に抵抗する |
| 情動重み付き減衰 | 情動が強い記憶ほど忘れにくい（半減期が変動） |
| シナプスホメオスタシス | 睡眠中にリンクを一律減衰、弱いリンクを刈り込む |
| ひらめき連想 | insightが保存されると連想チェーンが自動で走る |
| **内的対話** | **共感・補完・批判・連想の4つの声が同時に想起する** |
| **暗黙の気分推定** | **最近触った記憶の情動から心理状態を自動推定** |
| **デフォルトモードネットワーク** | **会話の間隔が長いほど、弱いリンクを辿って意外な連想を生成** |
| **P2P同期** | **複数端末間で記憶を共有。各端末が独立した海馬として動作** |

### 予測符号化（v3で追加）

サイバネティクス的フィードバックループ。脳は常に次の入力を予測し、予測を裏切った分（予測誤差）だけを学習シグナルにする。

```
新しい入力
    ↓
予測誤差 = 1 - max(既存記憶との類似度)
    ↓
誤差大 → 重要度↑ arousal↑（新規性の強化）
誤差小 → 変化なし → 干渉忘却が古い類似記憶を弱める（冗長性の排除）
    ↓
記憶ネットワーク（内部モデル）が更新される → ループ
```

干渉忘却と予測符号化が相補的に働き、記憶システムが自動的に情報量を最大化する。

### 場所細胞（v3で追加）

海馬の場所細胞に対応。記憶保存時にホスト名やSSH接続元IPを自動記録し、同じ場所で作られた記憶が想起されやすくなる。

- ローカル: `local:NucBox_EVO-X2`
- SSH経由: `ssh:192.168.1.50`

### 内的対話（v3.1で追加）

人間の頭の中には複数の声がある。`recall --voices` で4つの声が同時に想起する:

- 🤝 **共感**: 気分に寄り添う記憶（状態依存記憶）
- 🔭 **補完**: 気分と**逆**の記憶（見えていないもの）
- ⚡ **批判**: 過去の葛藤・不安からの警告
- 🎲 **連想**: ランダムウォークで到達した意外な記憶

共感だけなら模倣。補完があるから相互補完になる。LLMを人間にするのではなく、人間の内的対話を外在化する道具。

### デフォルトモードネットワーク（v3.1で追加）

脳がタスクに集中していないときに活性化するネットワーク。前回の会話からの間隔に応じて自動起動し、弱いリンクを優先してランダムウォークする。普段つながらない記憶を結びつけて返す。

- < 1時間: 起動しない（まだ集中モード）
- 1-6時間: 短い散歩（2回、3ホップ）
- 6-24時間: 中程度の散歩（3回、4ホップ）
- 24時間+: 長い散歩（5回、5ホップ）

### P2P同期（v4で追加）

複数端末間で記憶を共有する。各端末が独立した海馬として動作し、接続時に差分を交換する。

```bash
# 端末A（サーバー側）
# 既定はローカルのみ(127.0.0.1)・認証必須
# トークン未設定だと起動拒否される
set MEMORY_SYNC_TOKEN=your_secret
python memory.py sync serve

# LANに公開する場合（明示）
python memory.py sync serve --public

# 無認証で動かす場合（非推奨・明示）
python memory.py sync serve --insecure

# 端末B（クライアント側）
python memory.py sync pull 192.168.1.50:7235   # Aの記憶を取得
python memory.py sync push 192.168.1.50:7235   # Bの記憶をAに送信
```

衝突解決: access_countは大きいほう、content/emotionsはupdated_atが新しいほうを採用。忘却も同期される。

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

# 自動想起（情動 × 鮮度 × 強化度 × プライミング × 時間帯 × 場所 × 気分でランク）
python memory.py recall

# 内的対話（共感・補完・批判・連想の4声）
python memory.py recall --voices

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

### 自伝的記憶 (autobiography.py)

エピソード記憶を時系列で並べ、情動の弧と記憶間のリンクを可視化し、ナラティブとして出力する。

```bash
python autobiography.py
```

## マルチAI統合

ghostは複数のAI CLIから共有できる。各AIが同じ脳（memory.db）を読み書きする。

| AI | 設定ファイル | スキル |
|----|-------------|--------|
| Claude Code | CLAUDE.md | `/dive` `/surface` `/sleep` |
| Gemini CLI | GEMINI.md | `/dive` `/surface` `/sleep` |
| Codex CLI | — | 直接memory.pyを実行 |

### スキル

| スキル | 説明 |
|--------|------|
| `/dive` | 脳に接続。recallで記憶をロード |
| `/surface` | 記憶を書き戻してから切断。素のLLMに戻る |
| `/sleep` | 夢→リプレイ→統合→スキーマ→手続き化→思考。カットアップで報告 |

### Claude Code

v3からCLAUDE.mdを最小化（70%削減）。記憶操作は**サブエージェントに委譲**し、メインのコンテキストウィンドウを汚染しない。

- CLAUDE.md: サブエージェント委譲の指示だけ（~1.5KB）
- MEMORY_GUIDE.md: コマンド詳細（サブエージェントが読む、メインには載らない）
- 記憶の想起・検索結果はサブエージェント内で消費され、3行の要約だけがメインに返る

### Gemini CLI

GEMINI.mdでセッション開始時に自動diveする設計。Windows環境の文字化け対策（chcp 65001）を含む。

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
| importance | 1-5（自動推定、予測誤差で補正） |
| keywords | キーワード断片（JSON） |
| emotions | 情動タグ（JSON） |
| arousal | 覚醒度 |
| created_at | 記録日時 |
| last_accessed | 最後にアクセスした日時 |
| access_count | アクセス回数 |
| forgotten | 忘却フラグ |
| source_conversation | 出典 |
| embedding | ベクトル表現（BLOB, 384次元） |
| temporal_context | 時間的文脈（時間帯・曜日） |
| spatial_context | 空間的文脈（ホスト名・SSH接続元） |
