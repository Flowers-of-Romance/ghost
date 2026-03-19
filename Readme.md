# Ghost — LLMのための脳

LLMに脳の仕組みを模した長期記憶を実装する。

## 脳の構造

```
ghost/
├── memory.py              # 記憶システム本体 — 海馬+新皮質
├── Extract.py             # 会話ログからの記憶抽出 — 海馬の取り込み
├── tokenizer.py           # 日本語形態素解析（fugashi/SudachiPy/regex）— FTS5用
├── dream.py               # バロウズ式カットアップ夢 — 睡眠中の脳内イメージ
├── interpret_dream.py     # 夢の解釈 — 断片の出典・情動分析
├── autobiography.py       # 自伝的ナラティブ生成 — エピソード記憶の物語化
├── memory_server.py       # embeddingモデル常駐サーバー — 高速化用
├── record_turn.py         # 会話の全ターン自動保存 — 完全記憶の入力側
├── ghost_hooks.py         # PostToolUse hook — prospective検証 + 自動nap
├── wander.py              # 目的なき連想（DMN agent）— Gemini/ローカルLLMで自由連想
├── think.py               # 一人で考える — 記憶ネットワークを歩いて新しいつながりを探す
├── ghost-local.py         # ローカルLLMチャット（ollama） — 記憶付き対話
├── memory_sync_server.py  # P2P記憶同期サーバー — 複数端末間の記憶共有
├── CLAUDE.md              # Claude Code統合ルール
├── GEMINI.md              # Gemini CLI統合ルール
├── MEMORY_GUIDE.md        # 記憶システム詳細ガイド（サブエージェント用）
├── .claude/skills/        # Claude Code用スキル（dive/surface/sleep/delusion）
├── .gemini/skills/        # Gemini CLI用スキル（dive/surface/sleep/delusion）
└── memory.db              # SQLiteデータベース（init後に生成）
```

## セットアップ

```bash
git clone https://github.com/Flowers-of-Romance/ghost.git
cd ghost
pip install sentence-transformers numpy fugashi unidic-lite
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
| **メタデータ変容** | **睡眠のたびにキーワード・埋め込み・情動が隣接記憶の影響で変化する** |
| **修正可能性** | **provenance（出自）とconfidence（信頼度）で記憶の重みを調整。全版保持で改訂可能** |
| ひらめき連想 | insightが保存されると連想チェーンが自動で走る |
| **内的対話** | **共感・補完・批判・連想の4つの声が同時に想起する** |
| **暗黙の気分推定** | **最近触った記憶の情動から心理状態を自動推定** |
| **デフォルトモードネットワーク** | **会話の間隔が長いほど、弱いリンクを辿って意外な連想を生成** |
| **完全記憶** | **会話の全ターンをリアルタイムで自動保存。delusionで完全検索可能** |
| **P2P同期** | **複数端末間で記憶を共有。各端末が独立した海馬として動作** |

### 予測符号化

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

### 場所細胞

海馬の場所細胞に対応。記憶保存時にホスト名やSSH接続元IPを自動記録し、同じ場所で作られた記憶が想起されやすくなる。

- ローカル: `local:NucBox_EVO-X2`
- SSH経由: `ssh:192.168.1.50`

### 内的対話

人間の頭の中には複数の声がある。`recall --voices` で4つの声が同時に想起する:

- 🤝 **共感**: 気分に寄り添う記憶（状態依存記憶）
- 🔭 **補完**: 気分と**逆**の記憶（見えていないもの）
- ⚡ **批判**: 過去の葛藤・不安からの警告
- 🎲 **連想**: ランダムウォークで到達した意外な記憶

共感だけなら模倣。補完があるから相互補完になる。LLMを人間にするのではなく、人間の内的対話を外在化する道具。

### デフォルトモードネットワーク

脳がタスクに集中していないときに活性化するネットワーク。前回の会話からの間隔に応じて自動起動し、弱いリンクを優先してランダムウォークする。普段つながらない記憶を結びつけて返す。

- < 1時間: 起動しない（まだ集中モード）
- 1-6時間: 短い散歩（2回、3ホップ）
- 6-24時間: 中程度の散歩（3回、4ホップ）
- 24時間+: 長い散歩（5回、5ホップ）

### P2P同期

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

## コマンド一覧

### 日常使うもの

| コマンド | 何をする | いつ使う |
|---------|---------|---------|
| `recall` | 最近の記憶をスコア順に表示 | 会話の最初。「何を覚えてるか」の確認 |
| `recall --voices` | 共感・補完・批判・連想の4つの声で想起 | 一つの視点に偏ってるとき |
| `search "語"` | 意味の近い記憶をベクトル検索 | 「あれなんだっけ」のとき |
| `search "語" --raw` | 検索結果を原文で表示（再構成モードではなく） | 正確な内容を確認したいとき |
| `add "内容" カテゴリ` | 記憶を追加。情動・重要度は自動推定 | 覚えておきたいことがあるとき |
| `overview` | 脳の俯瞰。構造・重心・arousal分布・時系列 | 「この脳どうなってる？」のとき |
| `stats` | 数字だけの統計 | overviewより軽く見たいとき |
| `detail ID` | 1件の記憶の全情報 | 特定の記憶を深掘りしたいとき |
| `correct ID "内容"` | 記憶を修正（旧版を保存） | 間違った記憶を直したいとき |
| `versions ID` | 記憶の版履歴を表示 | 改訂の経緯を見たいとき |

### delusion（完全記憶検索）

通常の検索は「脳の検索」— 忘却・情動バイアス・減衰がかかる。delusionはそれを全部外して、事実だけを返す。

| コマンド | 何をする |
|---------|---------|
| `delusion "語"` | 純粋ベクトル検索。忘却された記憶も含む |
| `delusion "語" --date 2024-12-11` | 日付フィルタ付き |
| `delusion "語" --after 2024-11 --before 2025-02` | 期間フィルタ |
| `delusion --date 2024-12-11` | その日の全記憶ダンプ |
| `delusion --all` | 全記憶ダンプ |
| `delusion --raw "語"` | 対話原文（raw_turns）のみ検索 |
| `delusion --context ID` | 記憶IDから元の対話文脈を復元 |

通常検索で「アタリ」をつけてからdelusionで正確な内容を引く2段階リレーが基本。

### 睡眠処理（`/sleep` で一括実行される）

寝てる間に脳がやること。手動で個別実行もできる。

30分以上操作がないと **自動nap** が発動する（PostToolUse hook経由）。napはreplay + consolidateだけの軽量sleep。LLM不要。

| コマンド | 何をする | 脳の何に相当 |
|---------|---------|------------|
| `nap` | replay + consolidateのみ（30分無操作で自動発動） | うたた寝 |
| `replay` | リンク再計算、刈り込み、メタデータ変容、自動忘却 | シナプスホメオスタシス + メタデータ変容 |
| `mutations [ID]` | メタデータ変異履歴の閲覧（直近20件 or 特定記憶） | 監査ログ |
| `consolidate` | 類似度が非常に高い記憶ペアを1つに統合 | 記憶の統合・圧縮 |
| `schema` | リンク密集クラスタからメタ記憶（スキーマ）を生成 | 個別記憶→抽象知識 |
| `proceduralize` | 大量に参照された記憶を行動指針に昇格（LEARNED.mdに書出し） | 手続き記憶化（Hebbian learning） |
| `review [N]` | 長期間触ってない重要な記憶を表示 | 間隔反復（Spaced Repetition） |

### 抽出

| コマンド | 何をする |
|---------|---------|
| `python Extract.py` | 最新のClaude Codeセッションから記憶+原文を抽出 |
| `python Extract.py --all` | 全セッションから一括抽出 |
| `python Extract.py --chat file.txt` | claude.aiからコピペした会話テキストから抽出 |
| `python Extract.py --dry-run` | 保存せず候補だけ表示 |

### あまり手動で使わないもの

| コマンド | 何をする | 備考 |
|---------|---------|------|
| `forget ID` | 記憶を忘却（削除ではなくフラグ）| delusionでは見える |
| `resurrect "語"` | 忘却された記憶を検索して復活 | delusionで見つけてからでもいい |
| `chain ID [depth]` | 連想リンクを芋づる式にたどる | 特定の記憶から関連を探索 |
| `mood emotion arousal` | 気分を手動設定 | 自動推定があるので普段は不要 |
| `mood clear` | 気分リセット | |
| `prospect add "trigger" "action"` | トリガー語で自動リマインド登録 | 検索/addのたびに自動チェックされる |
| `recent [N]` | 最近の記憶N件 | |
| `all` | 全記憶表示 | 件数多いと重い |
| `search "語" --fuzzy` | 舌先現象モード（類似度0.45-0.65のもやもや記憶も表示）| |

### カテゴリ

| カテゴリ | 何 | 特殊な挙動 |
|---------|-----|-----------|
| fact | 事実 | なし |
| episode | 出来事 | なし |
| context | 進行中の文脈 | 30日で自動失効 |
| preference | 好み | なし |
| procedure | 手続き | なし |
| schema | メタ記憶 | 自動生成。統合の産物 |

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
| Claude Code | CLAUDE.md | `/dive` `/surface` `/sleep` `/delusion` |
| Gemini CLI | GEMINI.md | `/dive` `/surface` `/sleep` `/delusion` |
| ローカルLLM | ghost-local.py | 組み込みコマンドで直接操作 |
| Codex CLI | — | 直接memory.pyを実行 |

### スキル

| スキル | 説明 |
|--------|------|
| `/dive` | ghostに接続。recallで記憶をロード |
| `/surface` | 記憶を書いてから切断。素のLLMに戻る |
| `/sleep` | 夢→リプレイ→統合→スキーマ→手続き化→思考。カットアップで報告 |
| `/delusion` | 完全記憶モード。2段階リレーで事実を正確に引き出す |

### Claude Code

記憶操作は**サブエージェントに委譲**し、メインのコンテキストウィンドウを汚染しない。

- CLAUDE.md: サブエージェント委譲の指示だけ（~1.5KB）
- MEMORY_GUIDE.md: コマンド詳細（サブエージェントが読む、メインには載らない）
- 記憶の想起・検索結果はサブエージェント内で消費され、3行の要約だけがメインに返る

### Gemini CLI

GEMINI.mdでセッション開始時に自動diveする設計。Windows環境の文字化け対策（chcp 65001）を含む。

## データ

```bash
python memory.py overview           # 俯瞰（構造・重心・層・delusionの領域）
python memory.py stats              # 数字だけの統計
python memory.py export [filename]  # JSONエクスポート
python memory.py import filename    # JSONインポート
```

## DBテーブル

### memories（記憶）

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| content | 記憶の内容（全文） |
| category | fact / episode / context / preference / procedure / schema |
| importance | 1-5。自動推定+予測誤差で補正 |
| keywords | キーワード断片（JSON配列） |
| emotions | 情動タグ（JSON配列）: surprise, conflict, determination, insight, connection, anxiety |
| arousal | 覚醒度 0.0-1.0。0.85以上は「外傷的記憶」として特殊扱い |
| created_at | 記録日時（ISO 8601） |
| last_accessed | 最後に想起した日時 |
| access_count | 想起回数。多いほど強化される。20回超で馴化 |
| forgotten | 忘却フラグ。1=通常検索では見えない。delusionでは見える |
| embedding | ベクトル表現（BLOB, 384次元, multilingual-e5-small） |
| spatial_context | 場所（ホスト名/SSH接続元） |
| temporal_context | 時間帯・曜日 |
| provenance | 出自: user_explicit / wander / consolidation |
| confidence | 信頼度 0.0-1.0。出自に基づいて自動設定 |
| revision_count | 改訂回数。多いほど安定性スコアが下がる |

### raw_turns（対話原文）

会話の全ターンを切り詰めなしで保存。delusionの`--raw`検索の対象。

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| session_id | セッションID（JONSLファイル名） |
| role | user / assistant |
| content | 発話の全文 |
| timestamp | 発話日時 |
| memory_ids | この発話から抽出された記憶のID群（JSON配列） |

### memory_versions（版履歴）

interfere/consolidate/correctの前に記憶の状態をスナップショット保存。

| カラム | 何 |
|--------|-----|
| memory_id | 対象の記憶ID |
| content | その時点の内容 |
| importance | その時点の重要度 |
| arousal | その時点の覚醒度 |
| confidence | その時点の信頼度 |
| reason | 保存理由: interference / consolidation / user_correction |
| superseded_by | 統合先の新記憶ID（統合時のみ） |
| created_at | スナップショット日時 |

### memories_fts / raw_turns_fts（全文検索）

FTS5インデックス。fugashiで形態素解析してからスペース区切りで格納。ベクトル検索の補助。
