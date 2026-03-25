# Changelog

## [v16.2] - 2026-03-25

### Changed
- **Enacting強化**: recall/search出力にcontent要約を追加。キーワード断片だけでは「何の記憶か」分からない問題を解決
  - `format_memory_compact()`: 2行目に `「content先頭80文字」` を表示
  - `format_memory_reconstructive()`: 連想行の前に `↳ 「content先頭120文字」` を挿入
  - `--fragments` フラグ: 従来のキーワード断片のみモードに戻す（脳のパターン補完モデルを捨てない）
- **`_get_session_gap()`**: マイクロ秒付きタイムスタンプ（`2026-03-25T23:45:23.018700`）をパースできるように修正

### Design rationale
recall出力がキーワード断片 `[mem0, memory, がありそうだ]` だけでは、LLMが「何の記憶か」を再構成できない場面があった。content要約を追加することで、正しい記憶が引かれた時にLLMが活用できるようになる。

## [v16.1] - 2026-03-25

### Fixed
- **recall_polyphonic**: 想起した記憶の`last_accessed`/`access_count`を更新していなかったバグを修正。polyphonic recallで返された記憶が「アクセスされた」と記録されず、セッション間隔（`_get_session_gap`）が常に古いままになっていた（例: 何度diveしても「3日ぶり」と表示される）

## [v16] - 2026-03-25

### Added
- **ingest_chat.py**: claude.aiコピペ会話専用のパーサー & 取り込みモジュール。Extract.pyから`parse_chat_text` / `process_chat_text` / `extract_chat_from_jsonl`を移動・強化
  - **3戦略パーサー**: タイムスタンプベース / ロールヘッダーベース / 空行ヒューリスティックの3段階フォールバック。タイムスタンプがないコピペにも対応
  - **タイムスタンプ多様化**: `8:40` / `8:40:12` / `3/25 8:40` / `2026-03-25 08:40` の4パターンに対応
  - **ロールヘッダー検出**: "あなた" / "You" / "Claude" / "Human" / "Assistant" をロール境界として識別
  - **英語UIマーカー対応**: "Searched the web" / "Created a file" 等の英語UIラベルもシステムマーカーとして除去
  - **JSONL検出閾値緩和**: 5000→3000文字、ロールヘッダーパターンも検出条件に追加
  - `--stdin` オプション: 標準入力からパイプで会話テキストを受け取り
  - `--detect` オプション: JSONL内のclaude.ai会話を自動検出

### Changed
- **Extract.py**: `parse_chat_text` / `process_chat_text` / `extract_chat_from_jsonl` をingest_chat.pyへの委譲スタブに置き換え。後方互換性を維持

### Design rationale
Extract.pyはClaude CodeのJSONL形式に特化した抽出器。claude.aiのコピペ会話は全く異なるフォーマット（フリーテキスト、タイムスタンプの有無が不定、UIラベルの混入）を持つため、専用パーサーに分離した。Extract.pyの`--chat`オプションは委譲スタブを通じて引き続き動作する。

## [v15] - 2026-03-21

### Added
- **睡眠中の記憶固定化（promote）**: raw_turnsから覚醒度で重み付きサンプリングし、memoriesに自動昇格。海馬リプレイの模倣。既知の記憶は予測符号化で自然に弾かれるため、フラグ管理不要
  - `memory.py promote` コマンド追加
  - XMLタグ・tool ID・UUID・ハッシュ値を自動除去するクリーニング処理
- **auto_consolidate.py**: Stop hookで会話終了時にpromote(5件) + nap(replay + consolidate)を自動実行。手動の`/sleep`や`memory.py add`なしで記憶が固定化される
- **sleep.py**: `/sleep`の全ステップ（memo index → promote → dream → replay → consolidate → schema → proceduralize → think → stats）を1プロセスで一括実行。8回のBash呼び出しを1回に統合

### Changed
- **`/sleep`スキル**: sleep.pyを使うように簡素化。手動addの手順を削除
- **`.claude/settings.local.json`**: Stop hookにauto_consolidate.pyを追加
- **README**: 睡眠処理セクションを3層の自動固定化（会話終了時・アイドル時・手動sleep）に書き直し

### Design rationale
完全記憶（v13）でraw_turnsに全発言が入るようになったが、memoriesへの昇格は手動addだけだった。脳は睡眠中に海馬から新皮質へ記憶を転送する。promoteはこれを模倣し、覚醒度による重み付きサンプリング + 予測符号化による自然な重複排除で、明示的なフラグ管理なしに固定化を実現する。

## [v14] - 2026-03-19

### Removed
- **planカテゴリ廃止**: 記憶の特権階級を撤廃。planは減衰しない・忘却しない・統合しない・メタデータ変容しないという4つの特別扱いを受けていたが、これはLLMのself-attentionの限界を内部で補おうとする設計だった。計画は外部ツールが担うべきで、記憶システム内に特権カテゴリを持つ理由がない
  - `VALID_CATEGORIES`から`plan`を削除
  - DB CHECK制約を更新
  - 減衰免除・統合スキップ・忘却スキップ・メタデータ変容除外を撤去
  - `delusion --plan`オプションを削除
  - `overview`のplan一覧表示を削除
- **ghost_hooks.pyからplan監視を削除**: キーワードマッチ・embedding類似度検索・クールダウン管理・セッションID管理を撤去。prospective検証とnap検知のみ残存

### Design rationale
LLMの記憶も計画もself-attentionに支配される。完全記憶モード（delusion）は記憶を外部化してattentionを迂回した。計画も同じ原理で外部ツールが担う。LLM内部のplanモードは、この設計思想と矛盾するため削除した。

## [v13] - 2026-03-19

### Added
- **完全記憶（real-time recording）**: 会話の全ターンをリアルタイムで`raw_turns`に自動保存。delusionが名実ともに「完全記憶」になった
  - **`record_turn.py`**: `UserPromptSubmit`フック（ユーザー発言）+ `Stop`フック（アシスタント応答）の2経路で全会話を捕捉
  - **UserPromptSubmit hook**: ユーザーの発言をリアルタイムで`raw_turns`に保存
  - **Stop hook**: 会話終了時に`transcript_path`から最後のアシスタント応答を読み取って保存
  - これまではExtract.pyで事後にJSONLから読み込むしかなく、会話が終わると未抽出の文脈が消えていた

### Changed
- **`.claude/settings.local.json`**: `UserPromptSubmit`と`Stop`フックを追加

## [v12] - 2026-03-17

### Fixed
- **proceduralize時にcategoryを更新**: 手続き化された記憶のcategoryを'procedure'に変更するようにした。以前はproceduresテーブルに入るだけでmemoriesのcategoryは元のままだった（[#1](https://github.com/Flowers-of-Romance/ghost/issues/1)）
- **Claude側sleepスキルにproceduralize追加**: Gemini側には既にあったがClaude側のSKILL.mdから漏れていた

## [v11] - 2026-03-16

### Added
- **修正可能性（correctability）**: 間違えたとき致命的にならない構造
  - **provenance**: 記憶の出自を追跡（user_explicit / wander / consolidation）
  - **confidence**: 記憶の信頼度（user: 0.8, wander: 0.3, consolidation: max×0.9）
  - **memory_versionsテーブル**: interfere/consolidate/correctの前に旧版をスナップショット保存。gitのように全版保持
  - **revision_count**: 改訂回数。頻繁に改訂される記憶は安定性スコアが下がる
  - **`correct ID "内容"`コマンド**: ユーザーによる明示的な記憶修正。旧版を保存してから上書き
  - **`versions ID`コマンド**: 記憶の全版履歴を表示
- **wanderノイズゲート**: `_validate_output()` — 短すぎる出力、繰り返し文字、非アルファベット過多を検出して除去
- **`wander.py --cleanup`**: 既存のwanderノイズ記憶（))))パターン、`<think>`ダンプ、非日本語）をforgotten=1に

### Changed
- **recall_important / search_memories**: スコア計算に`(0.5 + confidence * 0.5)`と安定性`1/(1 + revision_count * 0.15)`を乗算。wander記憶(0.3)→×0.65、user記憶(0.8)→×0.9
- **consolidate_memories**: 統合記憶のconfidence = max(元) × 0.9、provenance = 'consolidation'。統合前にスナップショット保存
- **interfere**: 干渉前にスナップショット保存
- **add_memory**: sourceベースでprovenance/confidenceを自動設定。出力に信頼度を表示
- **format_memory_detail**: 信頼度・出自・改訂回数を表示
- **export_memories / sync_export**: provenance, confidenceを含める
- **dream.py**: time.sleep演出を削除（5.4s→0.07s）。Claude Codeからの実行時に不要な待ち時間だった
- **wander.py _store()**: INSERT時にprovenance='wander', confidence=0.3を設定

## [v10] - 2026-03-15

### Added
- **自動nap**: 30分以上操作がないと次のツール使用時に軽量sleepが自動発動。PostToolUse hook（ghost_hooks.py）がタイムスタンプを管理
- **`nap`コマンド**: replay + consolidateだけのLLM不要な軽量sleep
- **sleep_metaテーブル**: nap/sleepのメタ情報（last_napタイムスタンプ等）を記録
- **PostToolUse hook登録**: `.claude/settings.local.json`にghost_hooks.pyを登録

### Changed
- **replay**: 新規リンク追加を廃止し、既存リンクの3軸選択的強化に変更（高arousal / surprise / 時間的近接24h以内）
- **schema**: 既存スキーマとキーワード70%以上重複する候補をスキップ
- **ghost_hooks.py**: plan警告に加え、アイドル検知+自動napを統合

## [v9] - 2026-03-13

### Added
- **メタデータ変容**: sleepのたびに記憶のメタデータが隣接記憶の影響で変化する。データ（content）は不変、メタデータが変容する
  - **キーワード吸収**: リンクされた隣接記憶のキーワードをstrength重み付けで確率的に取り込む（2+リンク、P=0.3、最大2追加/1削除、元の1.5倍キャップ）
  - **埋め込みドリフト**: 隣接記憶のcentroidに向かってα=0.05で微小移動（3+リンク）。語彙が変わっても検索が追従する
  - **情動ドリフト**: content+隣接記憶の先頭50文字でdetect_emotionsを再実行し、新しい情動をP=0.2で追加（5+リンク）
  - plan/schemaカテゴリは除外。4時間クールダウンで連続変異を防止
- **mutation_logテーブル**: 全変異を監査ログに記録。field/old_value/new_value/reasonで追跡可能
- **`mutations`コマンド**: `mutations` で直近20件、`mutations ID` で特定記憶の全変異履歴を閲覧

### Changed
- **replay**: リンク再計算後・自動忘却前にメタデータ変容を実行。変異件数を出力に表示
- **memories**: `last_mutated`カラム追加。クールダウン制御に使用

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
