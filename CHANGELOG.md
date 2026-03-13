# Changelog

## [v8] - 2026-03-13

### Added
- **delusionモード（完全記憶）**: 忘却・情動バイアス・再固定化を全て無効化した純粋検索。日付・期間フィルタ、全件ダンプ、対話文脈復元に対応
- **raw_turnsテーブル**: Claude Codeの全対話ターンを原文のまま保存。セッション・タイムスタンプで索引
- **FTS5全文検索**: fugashi形態素解析によるインデックス。memories/raw_turns両テーブルに対応。ベクトル検索との併用で高精度な日本語検索
- **tokenizer.py**: fugashi → SudachiPy → 正規表現の3段フォールバック形態素解析
- **planカテゴリ**: 減衰しない・忘却されない・統合されない特殊カテゴリ
- **overviewコマンド**: 脳の俯瞰表示（ハブ記憶、覚醒度分布、タイムライン、FTS統計等）
- **ghost-local.pyにdelusion/overview追加**: `/delusion`と`/overview`チャットコマンド
- **/delusionスキル**: Claude Code/Gemini CLI両対応。2段階リレー検索の対話戦略

### Changed
- **Extract.py**: 記憶抽出と同時にraw_turnsへ全ターン保存
- **README**: コマンドの用途別整理、ghost-local.py追加、マルチAI統合テーブル更新

## [v7] - 2026-03-12

### Security (Codex review)
- **SYNC_TOKEN必須化**: トークン未設定で同期サーバーが起動しない。無認証には`--insecure`を明示的に要求
- **デフォルト127.0.0.1バインド**: `--public`で明示しない限りローカルのみ。トークン必須化と二重防御
- **JSON検証・サイズ制限**: memory_server.py（256KB/20000文字）、memory_sync_server.py（5MB）にバリデーション追加
- **prediction_error Noneガード**: embedding無効時に予測誤差が0.5固定で重要度が過剰に上がる問題を修正。Noneを返して補正をスキップ
- **search_memories フォールバック**: embed_text()失敗時にLIKE検索にフォールバック
- **sync_import入力検証**: 必須フィールドチェック、categoryホワイトリスト、行単位try/exceptで不正レコードをスキップ
- **sync merge漏れ修正**: UPDATEにcategory/merged_from/context_expires_atを追加

### Added
- **GEMINI.md**: Gemini CLI統合ガイド。セッション開始時に自動dive、文字化け対策
- **スキル（dive/surface）**: 脳への接続・切断。Claude Code/Gemini CLI両対応
  - `/dive`: recallで記憶をロード、脳と同期
  - `/surface`: 記憶を書き戻してから切断、素のLLMに戻る
- **VALID_CATEGORIES定数**: DB CHECK制約とsyncバリデーションで共有

## [v6] - 2026-03-11

### Added
- **ghost-local**: ローカルLLM（llama.cpp等）にghost記憶を統合するチャットインターフェース
  - 会話開始時にrecallで記憶をロード
  - 会話終了時に重要な発話を自動保存
  - ローカルLLMとクラウドLLMが同じ脳を共有

## [v5] - 2026-03-11

### Changed
- **recallのコンテキスト汚染を大幅削減**: デフォルト出力をコンパクトモード（1記憶1行）に変更。15件×3行→10件×1行で約75%削減
- **recall件数**: デフォルト15件→10件に削減
- **`--full`フラグ**: 従来の再構成モード（連想リンク・情動詳細つき3行表示）を使いたい場合に指定

### Added
- **`format_memory_compact()`**: ID + 情動2つ + 重要度 + キーワード4つ + スコアを1行に凝縮
- **ひらめき表示**: recall時にthink.pyが保存した未表示の洞察を自動表示

## [v4] - 2026-03-11

### Added
- **P2P同期**: 複数端末間で記憶を同期。各端末が独立した海馬として動作し、接続時に差分を交換する
  - `sync push <host:port>` — ローカルの変更をリモートに送信
  - `sync pull <host:port>` — リモートの変更を取得してマージ
  - `sync serve` — 同期サーバーを起動
  - `sync status` — 接続確認と同期履歴
- **UUID**: 全記憶・リンクにUUIDを付与。端末間でIDが衝突しない
- **updated_at**: 全テーブルに更新タイムスタンプ。SQLiteトリガーで自動更新。差分同期の基盤
- **node_id**: 端末識別用UUID
- **memory_sync_server.py**: 同期専用HTTPサーバー（port 7235）
- **俯瞰トリガー**: コンテキスト疲労を感じたらrecall --voicesを実行する指示をCLAUDE.mdに追加

### Changed
- **衝突解決**: access_countは大きいほう、last_accessedは新しいほう、content/emotionsはupdated_atが新しいほうを採用
- **proceduralize**: 書き込み先をCLAUDE.md → LEARNED.mdに分離。fact/schemaカテゴリを除外

## [v3.3] - 2026-03-11

### Changed
- **proceduralize**: 書き込み先をCLAUDE.md → LEARNED.mdに分離。CLAUDE.mdを自動生成物で汚染しない
- **proceduralize**: fact/schemaカテゴリを除外。事実やメタ記憶が行動指針に昇格するのを防止

## [v3.2] - 2026-03-11

### Added
- **俯瞰の声（birds-eye view）**: 5つ目の声 🦅 が記憶全体の構造をメタレベルで観察。カテゴリ偏り、情動の欠落、アクセス集中、孤立ノード、中心テーマ、鮮度低下を検出
- **反芻検出（rumination detection）**: search後に同じ記憶ばかり触っていると警告。`recall --voices`で別視点を提案
- **自動内的対話**: 前回の会話から6時間以上空くと自動でvoicesモードに切り替え（軽量版: 2件/声）

## [v3.1] - 2026-03-11

### Added
- **内的対話（polyphonic recall）**: `recall --voices` で4つの声が同時に想起する
  - 🤝 共感: 気分に寄り添う記憶（状態依存記憶）
  - 🔭 補完: 気分と逆の記憶（見えていないもの）
  - ⚡ 批判: 過去の葛藤・不安からの警告
  - 🎲 連想: ランダムウォークで到達した意外な記憶（DMN的）
- **暗黙の気分推定**: mood未設定でも最近アクセスした記憶の情動から心理状態を推定。補完の声が気分設定なしでも機能する
- **デフォルトモードネットワーク（DMN）**: 前回の会話からの間隔に応じて起動。弱いリンクを優先してランダムウォークし、普段つながらない記憶を結びつける。間隔が長いほど多く歩く

### Changed
- **recall**: DMNの結果を自動表示（間隔1時間以上で起動）
- **気分不一致ブースト**: 明示mood → 暗黙mood（最近触った記憶の情動）の順でフォールバック

## [v3] - 2026-03-11

### Added
- **予測符号化（predictive coding）**: 新しい記憶の保存時に既存記憶との予測誤差を計算。予測を裏切る情報ほど重要度・arousalが上がる。干渉忘却と相補的に働き、記憶システムが自動的に情報量を最大化するサイバネティクス的フィードバックループ
- **場所細胞（place cells）**: 記憶保存時にホスト名/SSH接続元IPを`spatial_context`に自動記録。同じ場所で作られた記憶が想起されやすくなる場所依存記憶
- **MEMORY_GUIDE.md**: 記憶システムの詳細ガイドをCLAUDE.mdから分離。サブエージェントが読む用

### Changed
- **CLAUDE.md最小化**: 4983→1490 bytes（70%削減）。コマンド一覧・脳動作説明を全てMEMORY_GUIDE.mdに移動
- **サブエージェント委譲**: 記憶操作をサブエージェントに委譲し、メインコンテキストに要約だけ返す設計に変更。コンテキスト汚染を防止
- **search/recall**: 場所ブースト（spatial_boost）をスコア計算に追加
- **stats**: 場所別の記憶数と現在の場所を表示
- **detail**: 記憶の場所を表示
- **export**: spatial_contextを含めてエクスポート

## [v2] - 2026-03-10

### Added
- **ヘブ学習（手続き化）**: 反復された記憶（access_count × リンク数が閾値超え）がLEARNED.mdの行動指針に自動昇格。`python memory.py proceduralize [--dry-run]`
- **ひらめき連想**: insightでarousal >= 0.5の記憶が保存されると、連想チェーンが自動で走って関連記憶を提案する
- **シナプスホメオスタシス（Tononi SHY）**: replay時に全リンクのstrengthを0.9倍。閾値以下は刈り込み。外傷的記憶のリンクは免除。リンク数が自然に平衡に達する
- **外傷的記憶**: arousal >= 0.85 の記憶は通常の処理パイプラインに抵抗する
  - 再固定化で馴化しない（想起のたびに再刻印）
  - 統合を拒否する（凍結）
  - 時間減衰が極端に遅い（半減期が通常の4-5倍）
  - 夢に頻出する（arousal²で重み付け）
- **情動重み付き減衰**: `effective_half_life(arousal)` — arousalが高い記憶ほど減衰が遅い

### Changed
- **dream.py**: 一様サンプリング → arousal重み付きサンプリング。連想クラスタ出力（リンクが強い記憶同士が一緒に出る）。外傷的記憶の反復優先
- **search/recall**: freshnessがarousalに応じた半減期を使用

## [v1] - 2026-03-09

初回リリース。

- 情動タグ（6種: surprise, conflict, determination, insight, connection, anxiety）
- 連想リンク（コサイン類似度 > 0.82 で自動結線）
- 断片保存と再構成モード
- 時間減衰（半減期14日）
- 再固定化（想起時の確率的arousalドリフト + 馴化）
- 統合・圧縮（類似度 > 0.94 のペアを統合）
- スキーマ生成（Bron-Kerboschクリーク検出）
- 干渉忘却（新しい記憶が似た古い記憶を弱める）
- プライミング（最近アクセスした記憶が関連記憶を促進）
- 状態依存記憶（気分一致性ブースト）
- フラッシュバック（忘却記憶の確率的復活）
- 予期記憶（トリガーベースのリマインド）
- 時間細胞（時間帯ブースト）
- 間隔反復レビュー
- 舌先現象（fuzzy recall）
- バロウズ式カットアップ夢（dream.py）
- 夢の解釈（interpret_dream.py）
- 自伝的記憶（autobiography.py）
- 会話ログからの記憶抽出（Extract.py）
- embeddingサーバー常駐化（memory_server.py）
