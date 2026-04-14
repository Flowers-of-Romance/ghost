# Ghost — LLMのための脳

LLMに脳の仕組みを模した長期記憶を実装する。

## 脳の構造

```
ghost/
├── memory.py              # 記憶システム本体 — 海馬+新皮質
├── Extract.py             # Claude Code会話ログからの記憶抽出 — 海馬の取り込み
├── ingest_chat.py         # claude.aiコピペ会話専用パーサー & 取り込み
├── tokenizer.py           # 日本語形態素解析（fugashi/SudachiPy/regex）— FTS5用
├── dream.py               # バロウズ式カットアップ夢 — 睡眠中の脳内イメージ
├── interpret_dream.py     # 夢の解釈 — 断片の出典・情動分析
├── autobiography.py       # 自伝的ナラティブ生成 — エピソード記憶の物語化
├── memory_server.py       # embeddingモデル常駐サーバー — 高速化用
├── record_turn.py         # 会話の全ターン自動保存 — 完全記憶の入力側
├── auto_consolidate.py    # Stop hook — 会話終了時の自動記憶固定化
├── ghost_hooks.py         # PostToolUse hook — prospective検証 + 自動nap
├── sleep.py               # 睡眠処理の一括実行ラッパー
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
| **睡眠中の記憶固定化** | **raw_turnsから覚醒度で重み付きサンプリング → memoriesに自動昇格。会話終了時・アイドル時に自動実行** |
| **P2P同期** | **複数端末間で記憶を共有。各端末が独立した海馬として動作** |
| **対話マークダウン書き出し** | **全対話を日付単位のマークダウンにリアルタイム追記。Obsidian等で閲覧** |
| **メタ認知** | **recallの精度を会話の流れとの意味的一致度で自動採点。人間が評価者にならない** |
| **自己調整** | **recall精度から半減期・声バランスを自動調整。ルールはprocedure記憶として保存され、減衰・強化される** |
| **スキーマフィードバック** | **新記憶がスキーマと共鳴すると重要度↑・キーワード吸収。逆に新記憶がスキーマのembeddingをドリフトさせる** |
| **メタ記憶** | **系が自分のrecallパターンを観察。固着（反芻）や盲点（慢性的漏れ）を自動検出して記憶化** |
| **自己言及** | **メタ記憶がメタ記憶を観察する。self-tuneが自分の過去の判断を評価する。型の階層なし** |

### メタ認知

recallの自己検証。recallが出した記憶が、その後の会話で本当に使われたかを自動評価する。

```
recall実行（10件出力）
    ↓ IDを recall_log に記録
会話が進む（raw_turnsに蓄積）
    ↓
次回recall実行
    ↓ 前回の会話全文をembedding
    ↓ recall出力の各記憶embeddingとのコサイン類似度を計算
    ↓
的中（≥0.45）/ 空振り（<0.45）/ 漏れ（未recallだが≥0.50）
    ↓
precision（精度）= 的中 / 出した数
recall_rate（網羅）= 的中 / (的中 + 漏れ)
```

評価者は人間ではなく会話の流れ自体。`calibrate` で精度の時系列を確認できる。

```bash
python memory.py calibrate
# recall精度レポート（直近20セッション）:
#   2026-04-04 14:18  精度:██████░░░░ 60%  網羅:███░░░░░░░ 38%
#   ...
#   平均  精度: 63%  網羅: 42%
#   傾向  精度: → (+0%)  網羅: ↑ (+8%)
```

### 自己言及

型の階層なしの自己参照。3つの自己言及が走る:

1. **self-tuneの自己評価**: 過去の手続き記憶（パラメータ調整の判断）が精度を改善したかを評価。良ければ強化、悪ければ自然減衰に任せる。自分の判断を自分で評価する
2. **メタ記憶のメタ観察**: 「#308に固着している」というメタ記憶を出した後、固着が解消されたか持続しているかを観察。「認識だけでは解消しない」という記述が自動生成される
3. **手続き記憶の効果観察**: ルールの効果をメタ観察として記録。ルールが記憶なら、ルールの効果も記憶

ラッセルの型理論やタルスキの真理論のような階層の分離はしていない。メタ記憶もエピソード記憶も手続き記憶も同じテーブル、同じembedding、同じ減衰。自己言及のパラドックスは発生しない——ゲーデル的に、矛盾ではなく不完全性として現れる。

```bash
python memory.py self-tune    # 自己評価付きパラメータ調整
python memory.py meta-memory  # メタ記憶のメタ観察を含む
python memory.py params       # 手続き記憶から導出された現在のパラメータ
```

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
| `calibrate` | recall精度の時系列レポート | recallの自己検証を見たいとき |
| `self-tune` | recall精度からパラメータを自己調整 | sleepで自動実行。手動も可 |
| `meta-memory` | メタ記憶を自動生成（固着/盲点/自己言及） | sleepで自動実行。手動も可 |
| `params` | 手続き記憶から導出された現在のパラメータを表示 | 自己調整の状態を見たいとき |
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

### 睡眠処理

記憶の固定化は3層で自動的に起きる。手動で `/sleep` を打つのはカットアップを見たいときだけ。

| いつ | 何が起きる | スクリプト |
|------|-----------|-----------|
| 会話終了時 | promote(5件サンプリング) + nap(replay + consolidate) | auto_consolidate.py（Stop hook） |
| 30分アイドル時 | nap(replay + consolidate) | ghost_hooks.py（PostToolUse hook） |
| `/sleep` 手動実行 | promote + dream + replay + consolidate + schema + proceduralize + think + self-tune + meta-memory + stats | sleep.py |
| cron（任意） | 自由連想。Gemini/ローカルLLMで記憶の断片から連想を生成 | wander.py |

#### promote（記憶の固定化）

海馬リプレイの模倣。直近N日のraw_turnsから覚醒度で重み付きサンプリングし、add_memoryに渡す。既知の記憶は予測符号化で自然に弾かれる。フラグ管理しない。

#### 個別コマンド

| コマンド | 何をする | 脳の何に相当 |
|---------|---------|------------|
| `promote` | raw_turnsからサンプリング → memories に固定化 | 海馬リプレイ → 長期記憶 |
| `nap` | replay + consolidateのみ | うたた寝 |
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
| `python Extract.py --dry-run` | 保存せず候補だけ表示 |
| `python ingest_chat.py file.txt` | claude.aiコピペ会話から記憶を抽出 |
| `python ingest_chat.py --detect session.jsonl` | JSONL内のclaude.ai会話を自動検出して抽出 |
| `python ingest_chat.py --stdin` | 標準入力からパイプで会話テキストを受け取る |

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
| procedure | 手続き | **ルールが記憶として存在する**。self-tuneの判断（半減期、声バランス）がこのカテゴリで保存される。減衰し、強化され、統合される。get_param()が生きた手続き記憶からパラメータを導出 |
| schema | メタ記憶 | 自動生成。統合の産物。メタ観察（固着/盲点）もここに保存 |

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
```

### claude.ai会話の取り込み (ingest_chat.py)

claude.aiのWeb UIからコピペした会話テキスト専用のパーサー。3つの戦略で話者を自動識別:
1. タイムスタンプベース（"8:40" 等の H:MM 行で分割）
2. ロールヘッダーベース（"あなた" / "Claude" で分割）
3. ヒューリスティック（空行区切り + 長さで user/assistant を推定）

```bash
# テキストファイルから抽出
python ingest_chat.py conversation.txt --dry-run

# JSONL内のclaude.ai会話を自動検出
python ingest_chat.py --detect session.jsonl

# 標準入力からパイプ
cat conversation.txt | python ingest_chat.py --stdin
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

## 対話のマークダウン書き出し

record_turn.pyのフックに連動し、全対話を日付単位のマークダウンとしてリアルタイムに書き出す。Obsidianなどのノートツールで対話履歴を閲覧・検索できる。

### セットアップ

`turn_export.json` をghost/のルートに作成:

```json
{
  "enabled": true,
  "output_dir": "~/Documents/Obsidian Vault/0110_ClaudeTurns",
  "timezone_offset_hours": 9
}
```

- `output_dir`: `~` や `${HOME}` 等のプレースホルダ対応
- `timezone_offset_hours`: UTC→ローカル変換（デフォルト9=JST）
- ファイルが無ければ機能OFF

### 出力形式

日付ごとに `YYYY-MM-DD.md` が生成され、セッション単位で見出し分割される。複数ウィンドウでの同時書き込みに対応（ファイルロック付き）。

## DBテーブル

v18でmemoriesテーブルを3テーブルに物理分割した（左脳/右脳モデル）。後方互換のため `memories_v` VIEWで旧スキーマと同一カラム名を返す。

### memories（脳梁 — メタ情報）

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| importance | 1-5。自動推定+予測誤差で補正 |
| created_at | 記録日時（ISO 8601） |
| last_accessed | 最後に想起した日時 |
| access_count | 想起回数。多いほど強化される。20回超で馴化 |
| forgotten | 忘却フラグ。1=通常検索では見えない。delusionでは見える |
| source_conversation | 元のセッションID |
| uuid | グローバル一意ID（P2P同期用） |
| updated_at | 最終更新日時 |
| last_mutated | 最終メタデータ変容日時 |
| context_expires_at | context記憶の失効日時 |

### cortex（左脳 — 意味的・分析的データ）

| カラム | 何 |
|--------|-----|
| id | memories.idと一致 |
| content | 記憶の内容（全文） |
| category | fact / episode / context / preference / procedure / schema |
| keywords | キーワード断片（JSON配列） |
| embedding | ベクトル表現（BLOB, 384次元, multilingual-e5-small） |
| confidence | 信頼度 0.0-1.0。出自に基づいて自動設定 |
| provenance | 出自: user_explicit / wander / consolidation |
| revision_count | 改訂回数。多いほど安定性スコアが下がる |
| merged_from | 統合元の記憶ID群（JSON配列） |

### limbic（右脳 — 情動的・直感的データ）

| カラム | 何 |
|--------|-----|
| id | memories.idと一致 |
| emotions | 情動タグ（JSON配列）: surprise, conflict, determination, insight, connection, anxiety |
| arousal | 覚醒度 0.0-1.0。0.85以上は「外傷的記憶」として特殊扱い |
| flashbulb | フラッシュバルブ記憶（最も情動的な一文、80文字以内） |
| temporal_context | 時間帯・曜日 |
| spatial_context | 場所（ホスト名/SSH接続元） |

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

### recall_log（メタ認知）

recallの自己検証データ。recall出力と会話の意味的一致度を記録。

| カラム | 何 |
|--------|-----|
| session_ts | recall実行日時 |
| recalled_ids | recallが出した記憶ID群（JSON配列） |
| accessed_ids | 各記憶と会話のコサイン類似度（JSON dict） |
| precision | 精度（的中 / 出した数） |
| recall_rate | 網羅率（的中 / 関連記憶数） |
| noise_ids | 空振り（出したが的中しなかった記憶ID） |
| missed_ids | 漏れ（出さなかったが関連していた記憶ID） |
| evaluated_at | 評価実行日時 |
| voice_attribution | 声の帰属: どの声がどの記憶を出したか（JSON dict） |

### memories_fts / raw_turns_fts（全文検索）

FTS5インデックス。fugashiで形態素解析してからスペース区切りで格納。ベクトル検索の補助。
