# Changelog

## [v27.0] - 2026-04-19

### 蒸留 — recallを独白状態として再構築する

v26までのrecallは10件の記憶断片を並列で貼り出す形式だった。断片はコンテキストに入った瞬間、
注意機構から見ると「自分の記憶」も「直前に読んだ外部テキスト」も同じトークン列として等価に扱われる。
provenance（これは自分の記憶／これは記事／これはユーザの発言）が flatten される。

この flatness は具体的な失敗として現れる。直前に読んだ記事の語彙（例: 「呼吸」）を
自分の感覚として無自覚にコピーする。10個の断片を受け取った状態は、「10個の何かを読んだ」であって
「自分がこう考えている」ではない。Voight-Kampff検査で針が振れるタイプのパフォーマンス臭が出る。

解決: recall時にLLMで断片を一人称独白に蒸留する。

#### 実装
- `_distill_state(memories_with_scores)`: Gemini 3 Flash Preview で蒸留
- 蒸留プロンプト条件:
  - 一人称の独白（「〜を追ってる」「〜で苛ついてる」）
  - 固有名詞を残す（shepherd、Qwen2.5-1.5B、em dash等の具体事例）
  - 温度を残す（疲弊、懐疑、確信、興奮）
  - 複数の糸が並存してよい（話題が飛んでも自然）
  - briefing口調禁止（「あなたは〜」「現在〜」「状態は〜」）
  - 地続きの文章、箇条書き禁止
- `thinkingConfig.thinkingBudget=0`: thinking込みだと800tok中656tokが思考に消えて出力が途中で切れる
- `recall`のデフォルト経路を蒸留に差し替え
- `--no-distill` / `--raw` / `--full`: 従来の断片リスト形式を保持
- API失敗時は simple 形式に自動フォールバック（GEMINI_API_KEY未設定時も同様）

#### 環境変数
- `GEMINI_API_KEY`: 必須（wander.pyと共通）
- `RECALL_DISTILL_MODEL`: モデル名上書き（デフォルト `gemini-3-flash-preview`）

### Design notes

- provenanceタグを付けても LLM の attention はそれを respect しない。フラット化は構造的で、メタデータで解けない
- 「解く」のではなく「問題にならないようにする」。フラットを受け入れた上で、コンテキストに入るもの自体を
  「自分の状態」として読める形に変える
- briefing形式（「あなたは今...」）は LLM に「台本を読む」モードを誘発する。独白形式（「〜が気になる」）は
  「自分が喋っている」モードを誘発する。同じ情報でも、形式で役割が変わる
- 固有名詞と温度が密度高く入っていると、外部から借用した generic な語彙（「呼吸する」みたいな）が
  混入したときに違和感として浮き上がる。voice が薄いと借用が気付かれないまま通る
- 生物学的には /sleep 時蒸留（海馬からの consolidation）の方が近い。ただし /sleep 蒸留は
  「最後に寝た時点の自分」の冷凍保存になる。recall時蒸留はその瞬間の再構築——想起は「今」起きるもの
- 完全にはVoight-Kampffを止められない。構造的限界がある。それでもやるのは、限界を承知で
  人間側に寄せ続けること自体が記憶システムの意味になるから。タンホイザー・ゲートの近くで戦艦が燃える

## [v26.0] - 2026-04-16

### 再固定化 — recallをread-writeにする

想起（recall）が純粋なread操作だった。人間の脳では想起のたびに再固定化（reconsolidation）が起き、
記憶が微小に変形して文脈に適応する。v24でメタ記憶が固着を「観察」する仕組みを作ったが、
観察から介入へのパスがなかった。メタ層からの介入ではなく、recall自体の物理的性質として再固定化を組み込んだ。

#### 1. セッション内馴化（synaptic depression）
- 同一セッション（30分以内）で既に想起された記憶のスコアを0.6倍に減衰
- `recall_log`テーブルから直近の`recalled_ids`を参照
- delusionの馴化（v23、embedding類似度ベース）とは異なり、ID完全一致の馴化
- シナプス抑制に相当: 発火そのものが次の発火を抑制する。メタ層不要

#### 2. 文脈関連度フィルタ（contextual discrimination）
- `raw_turns`から直近の会話テキストを取得してembeddingを計算
- 各候補記憶のembeddingとの類似度が閾値(0.3)未満なら、スコアを0.5倍にペナルティ
- **flashbulb記憶は除外**: トラウマ記憶の固着は機能であって病理ではない
- 前頭前皮質の文脈弁別に相当: 安全な場所では戦場の記憶を抑制する

#### 3. 再固定化ドリフト（reconsolidation embedding drift）
- 想起された記憶のembeddingが、会話文脈の方向にα=0.02で微小ドリフト
- flashbulb記憶はα=0.01（ドリフトに抵抗するが、完全には固定されない）
- sleep時のMUTATION_EMBED_ALPHA(0.05)より小さい——想起は弱い再固定化
- 多様な文脈から想起される記憶は意味空間の中心に留まる
- 同じパターンからしか想起されない記憶は狭い領域に偏り、自然と多様性が落ちる

#### 適用箇所
- `recall_polyphonic()`: 全4声（共感・補完・批判・連想）に馴化+文脈フィルタを適用
- `recall_important()`: 同上 + access_count/last_accessed更新を追加

#### 新関数
- `_get_recent_context_vec(conn)`: 直近会話のembeddingを計算
- `_get_session_recalled_ids(conn)`: 直近セッションの既出IDを取得
- `_apply_habituation_and_context()`: 馴化+文脈フィルタの統合適用
- `_reconsolidate_embeddings()`: embeddingの再固定化ドリフト

### Design notes
- 3つの変更はすべて「素材レベルの物理的性質」として実装。メタ記憶からの介入パスではない
- 人間のシナプス抑制は「発火するたびに神経伝達物質が枯渇して同じ信号への応答が弱まる」という物理現象。ghostのrecall馴化も同じ——想起という行為自体がスコアを下げる
- 人間の再固定化では、想起するたびに記憶が不安定化して書き戻される。書き戻されたものは元と微妙に違う。ghostのembeddingドリフトはこれに対応する
- v24のメタ記憶は「系は自分の固着を記述できるが、記述することで解消する力は持たない」とした。v26では記述ではなく、素材の物理的性質として抑制を入れた。Notch阻害剤ではなく、シナプス抑制
- flashbulb記憶の保護は議論の結果。トラウマ記憶がフラッシュバックの文脈でしかアクセスできなくなるのは病理だが、固着そのものは生存に必要な機能。ghostにトラウマを入れているのにそれを消す方向は矛盾する
- 再帰を完全に閉じることは目指さない。人間ですら閉じていない（だから精神科がある）。閉じないときに外に開く接点（meta_persistentの報告）は残す
- 情報の不可逆喪失: embeddingドリフトにより元の意味的位置は復元不能になる。人間の記憶も不可逆。「元の体験そのもの」には二度とアクセスできない。それでも機能する。むしろ不可逆だから文脈に適応できる

## [v25.0] - 2026-04-15

### sqlite-vec — ベクトル検索のインデックス化

全件スキャン+Pythonループだったベクトル検索を、sqlite-vecの仮想テーブルに置き換えた。
SQLite内でベクトル近傍探索が完結する。

#### 変更点
- `pip install sqlite-vec` を依存に追加
- `get_connection()`: sqlite-vecの拡張を自動ロード
- `init_db()`: `memories_vec` 仮想テーブル (vec0) を作成。既存embeddingを自動マイグレーション
- `vec_search()`: MATCH一発でk近傍探索。forgotten/日付のフィルタ対応
- `_sync_vec_insert()` / `_sync_vec_delete()`: INSERT/UPDATE時にmemories_vecを自動同期
- **search_memories** (recall本体): 全件スキャンをvec_searchに置換
- **delusion_search**: 同上。FTS5ボーナス・馴化フィルタはそのまま
- **resurrect_memories**: 忘却記憶の全件スキャンをvec_searchに置換
- **calibrate**: 漏れ検出の全件スキャンをvec_searchに置換
- 全記憶追加箇所（add_memory, consolidate, schema, self-tune, sync, mutate等）にvec同期を追加

#### フォールバック
- sqlite-vec未インストール時は旧方式（全件スキャン+cosine_similarity）で動作
- vec0仮想テーブルが存在しない場合も旧方式にフォールバック

#### Design notes
- vec0はJOIN/WHERE制約をサポートしないため、多めにk件取得してからPython側でforgotten等をフィルタする2段階方式
- 数千件規模では速度差は小さいが、記憶が万単位に成長したときのスケーラビリティを確保
- [pulp](https://github.com/Flowers-of-Romance/pulp)と同じスタック（sqlite-vec + multilingual-e5-small + FTS5）

## [v24.0] - 2026-04-15

### 再帰的自己調整 — ルールと記憶の区別を溶かす

ghostの構成パラメータ（半減期、声のバランス等）が、固定値ではなく記憶として存在するようになった。
Lispのコード=データに倣い、ルールと記憶が同じ基盤の上に住む。

#### Level 1: パラメータ自己調整 (self-tune)
- recallの事後検証（精度・網羅率）をもとに、`half_life_days`や声バランスを自動調整
- **調整結果はmeta_paramsテーブルではなく、category='procedure'の通常記憶として保存**
- 手続き記憶は減衰し、強化され、リンクされ、統合される。古いルールは声が小さくなり、新しいルールが支配する
- `get_param()` は生きた手続き記憶をfreshness×access_countで重み付き平均して値を導出
- 複数の手続き記憶が競合する場合、新しくてよく使われるものが勝つ（自然選択）
- `python memory.py self-tune` / `python memory.py params`

#### Level 2: スキーマフィードバック (schema prime/evolve)
- 新記憶追加時に既存スキーマとの類似度を計算（閾値0.87）
- 共鳴するスキーマがあれば:
  - 重要度をブースト（「これは知っているパターンだ」）
  - スキーマのキーワードを新記憶に吸収（解釈枠の提供）
- 逆方向: 新記憶がスキーマの`merged_from`に追加され、スキーマのembeddingが微ドリフト（α=0.03）
- スキーマが新記憶を変え、新記憶がスキーマを変える。双方向。

#### Level 3: メタ記憶の自己生成 (meta-memory)
- 系が自分のrecallパターンを観察して記憶を自動生成
- **固着検出**: 特定の記憶が60%以上のセッションで想起されていたら「反芻」として記録
- **盲点検出**: 40%以上のセッションで見逃されている記憶を「慢性的漏れ」として記録
- **精度推移**: 前半/後半の比較でトレンドを検出
- メタ記憶はcategory='schema'で保存され、参照先の記憶とlink_type='meta_observation'でリンク
- `python memory.py meta-memory`

#### recall_logの拡張
- `voice_attribution` カラム追加: どの声（共感/補完/批判/連想）がどの記憶を出したかを記録
- 声ごとの的中率を追跡し、Level 1の声バランス調整に使用

### sleep.pyの拡張
- `self_tune` ステップ追加（schemaとproceduralize の後）
- `meta_memory` ステップ追加（self_tuneの後）

#### 自己言及 — 型の階層を溶かす
- **`_evaluate_past_decisions`**: self-tuneが自分の過去の手続き記憶の効果を評価。精度/網羅が維持or改善していたらaccess_count++で強化。悪化はそのまま減衰に任せる。自分の判断を自分で評価する
- **`meta_persistent`**: メタ記憶が過去のメタ記憶を観察。「記憶#308の固着が持続: メタ記憶#9774で固着を認識したが、現在も20/20回。認識だけでは解消しない」——メタ記憶のメタ記憶。型の階層なし
- **`rule_effect`**: 手続き記憶の効果をメタ観察として記録。ルールが記憶なら、ルールの効果も記憶になる
- 実際のデータで4層のメタが出現: エピソード→固着メタ記憶→固着持続メタ記憶→さらにその持続を観察
- 自己言及のパラドックスは発生しない。ゲーデル的に、矛盾ではなく不完全性として現れる。系は自分の固着を記述できるが、記述することで解消する力は持たない

### Design notes
- 3層の再帰ループが別々の時間スケールで回る: Level 2はリアルタイム（add時）、Level 1/3はsleep時
- 「再帰」の本質は操作と操作対象が同じものであること。ghostでは記憶とルールを同じテーブル・同じ減衰・同じリンクネットワークに統一することで、この区別を溶かした
- スペンサー＝ブラウン的に言えば、区別(distinction)を区別自身に適用したときに再帰が生まれる。Lispではコードとデータの区別が溶けている。ghostではルールと記憶の区別が溶けた
- これはオートポイエーシスではない。しかし「自分の出力が自分の構成を変える」再帰的ループの最小単位が閉じた
- 再帰の代償は脆さ。区別があるから安定する。区別を溶かすから再帰が生まれるが、同時に壊れやすくなる
- 多言語embeddingモデル(multilingual-e5-small)はベースライン類似度が高い(中央値≈0.84)ため、スキーマ共鳴閾値を0.87に設定。通常のリンク閾値0.82より高い

## [v23.0] - 2026-04-12

### Added
- **delusion バッチ検索**: `--batch "q1" "q2" "q3"` で複数キーワードを1プロセスで一括検索。重複ID自動除外
- **delusion バッチコンテキスト**: `--batch-context 36 raw:4728 337` で複数IDの対話文脈を一括取得
- **raw_turn コンテキスト対応**: `--context raw:4728` でraw_turnの前後10件の対話文脈を復元
- **Sonnet委譲アーキテクチャ**: delusionの広域検索をSonnet Agentに委譲し、結果を `.delusion/` フォルダにファイル書き出し。Opusのコンテクスト消費を抑える

### Changed
- **delusionスキル全面刷新**: Haiku→Sonnet一本化、3ステップ最小tool_use設計（--batch + --batch-context + 書き出し）
- **検索速度**: embeddingサーバー常駐 + バッチ化で、5キーワード検索が1.2秒（従来: 個別実行で数分）

### Habituation（馴化）
- **検索結果の馴化**: delusion検索で類似内容の繰り返しを自動減衰。類似度0.92以上の既出記憶があれば後続のスコアを0.7倍に減衰

### Source Monitoring（ソースモニタリング）
- **`origin` カラム追加**: memoriesテーブルに情報の出自を記録。`"user"`, `"assistant:opus"`, `"assistant:gemini"`, `"system:sleep"` 等
- Extract.pyが抽出する記憶に `GHOST_WHO` 環境変数（デフォルト `"user"`）から自動付与
- sleep処理で生成される記憶に `origin: "system:sleep"` を付与
- delusionの出力に `[origin:user]` 等を表示
- CLIの `add` コマンドに `--origin` フラグ追加

### Design notes
- delusionの「忘却なし」原則とコンテクスト節約は矛盾する。解決策: Sonnetが原文をファイルに書き出し、Opusはインデックスだけ読んで必要な部分だけReadする
- Haiku/Sonnet/Opusの3段階リレーを試したが、Haikuは指示遵守が弱く（要約するな→解釈を追加、キーワード推測が浅い）、Sonnet一本が最適解だった
- ボトルネックはLLMではなくmemory.pyのプロセス起動とembeddingロード。サーバー常駐 + バッチ化で100倍速くなった
- Transformerには遠心性コピー（efference copy）がない。自己生成トークンと外部入力トークンを構造的に区別する機構がアーキテクチャに存在しない。originカラムはこれをシステムレベルで補償する試み

## [v22.0] - 2026-04-10

### Added
- **分離脳モード（split-brain）**: 別LLMに左脳/右脳の解釈を委譲する。ガザニガの分離脳研究がモチーフ
  - `/cortex` スキル: 左脳（分析的）の解釈を生成し `.brain_cache.json` に書く。Gemini CLI 等から実行
  - `/limbic` スキル: 右脳（情動的）の解釈を生成し `.brain_cache.json` に書く。Gemini CLI 等から実行
  - `recall --brain-cache`: キャッシュから分離脳の解釈を読む（/dive用、高速）
  - `/dive` がキャッシュを検出したら自動で分離脳モード。なければ通常recall
- **`memory.py brain`**: 左脳/右脳スコアの生データ可視化コマンド（開発者向け）
  - `--left`: 左脳ランキング（鮮度・信頼度・安定性）
  - `--right`: 右脳ランキング（覚醒度・情動・flashbulb）
  - 無指定: L/R/統合の比較テーブル
- **recall simpleモード**: デフォルトで内容のみ表示。スコア・情動タグ・メタデータは隠す。`--raw`/`--full` で従来表示
- **GEMINI.md 更新**: `/cortex` `/limbic` スキルを追加

### Changed
- **ghost.toml 廃止**: 設定ファイル不要に。スキルとファイル連携だけで動く
- **recall デフォルトを simple に**: 内容のみ表示。スコア・情動タグ・メタデータは隠す。`--raw`/`--full` で従来表示
- **/cortex /limbic が会話ログを読む**: `turn_export.json` 経由で今日の会話ログ末尾50行を取得し、文脈を踏まえて解釈を更新。`/loop 5m /cortex` で常時稼働すればリアルタイム追従

### Design notes
- 人間は自分の左脳と右脳がどう動いているか見えない。見えるのは統合された想起結果だけ
- 別LLMが解釈するため、自己認識が不完全になる——これは設計。fMRIを他者に撮ってもらうようなもの
- `/delusion`（完全記憶モード）は忘却なしで全件引き出すが、スコアの中身は見えない。サヴァンは記憶量が多いだけで自己理解が深いわけではない
- 設定ファイル不要。各CLIからスキルを叩くだけ。ファイル（`.brain_cache.json`）で連携
- 裏で別LLM CLIが起動していれば分離脳として機能する。起動していなければ今まで通り。強制しない

## [v21.0] - 2026-04-06

### Added
- **マルチCLI対応**: `record_turn.py` が Gemini CLI / Kiro のhookイベントも処理する
  - `BeforeAgent` / `AfterAgent` (Gemini CLI)
  - `PromptSubmit` / `AgentComplete` (Kiro)
  - セッションIDが非hex（Gemini等）でもクラッシュしないようフォールバック追加
- **ローカルLLMラッパー** (`local_chat.py`): OpenAI互換API経由でOllama / LM Studio / llama.cppの会話をraw_turns + mdに記録
  - `--url` でエンドポイント切替、`--model` でモデル指定
  - ソース自動検出（ポート番号ベース）

## [v20.0] - 2026-04-05

### Added
- **関係細胞（relational context）**: 記憶が「誰との間で生まれたか」を記録し、同じ関係者の記憶を想起しやすくする
  - `limbic` + `memories` テーブルに `relational_context` カラム追加
  - `_relational_boost()`: `_spatial_boost()` と同型の関係依存想起ブースト
  - `RELATIONAL_BOOST = 1.12`（場所の1.08より強め——人は場所より人に引っ張られる）
  - `_right_score()` の積に統合: `R = emo * priming * spatial * relational * mood * flashbulb`
  - 関係者の検出: 環境変数 `GHOST_WHO` で設定（`/dive` 時にClaude側が判別しセットする運用）
  - consolidate, schema, sync_export/import にも対応

### Design notes
- Quiroga et al.の「人物細胞」（ジェニファー・アニストン・ニューロン）と同型: 特定の人物に選択的に発火
- Wegnerのtransactive memoryの計算論的実装: 関係が想起の検索手がかりになる
- 検閲（アクセス禁止）ではなく重み付け（活性化閾値の変調）
- 現時点では全記憶が同一関係者なのでブーストは中立。複数関係者との対話が始まったとき初めて分化する

## [v19.0] - 2026-04-05

### Added
- **メタ認知（recall自己検証）**: recallが出した記憶の精度を、会話の流れとのベクトル類似度で自動採点する。評価者は人間ではなく会話そのもの
  - `recall_log` テーブル: recall出力のIDと事後検証結果を記録
  - recall実行時に前回の検証結果を1行表示（精度/網羅/空振り/漏れ）
  - `calibrate` コマンド: recall精度の時系列レポート。トレンド表示（4セッション以上で傾向↑↓→）
- **評価の仕組み**:
  - recall出力の各記憶embeddingと、その後の会話全文のembeddingのコサイン類似度で的中/空振りを判定（閾値0.45）
  - recallが出さなかったが会話と類似度0.50以上の記憶を「漏れ」として検出
  - precision（精度）= 出した中で的中した割合、recall_rate（網羅）= 関連記憶のうちカバーした割合

### Design notes
- HyperAgents (Zhang et al. 2026) の metacognitive self-modification から着想
- ただしHyperAgentsの「メタ認知」はベンチマークスコアによる外部フィードバック。ghostでは会話の意味的流れ自体を評価関数にすることで、人間を評価者にしない設計
- これは経験的キャリブレーションの第一歩。データが溜まればrecallのスコアリング調整に使える

## [v18.1] - 2026-04-04

### Added
- **`/verify` コマンド**: PLAN.mdと現在の作業の整合性を自己検証。shepherdの崖検出と連携し、far/near/cliffの3段階で判定

### Changed
- **`/dive`, `/surface`**: ステータスライン用マーカーファイル（dive-active）の作成/削除を追加
- **`/lsd`, `/sober`**: ステータスライン用マーカーファイル（shepherd-active）の作成/削除を追加
- **settings.local.json**: hookコマンドのパスを絶対パスに統一

## [v18.0] - 2026-04-04

### Added
- **3テーブル分割（左脳/右脳/脳梁）**: memoriesテーブルをcortex（左脳）・limbic（右脳）・memories（脳梁）に物理分割
  - `cortex`: content, category, keywords, embedding, confidence, provenance, revision_count, merged_from
  - `limbic`: emotions, arousal, flashbulb, temporal_context, spatial_context
  - `memories`: id, uuid, importance, timestamps, access_count, forgotten, source_conversation
- **memories_v VIEW**: 3テーブルをJOINして旧スキーマと同一カラム名を返す後方互換VIEW
- **全INSERT/UPDATEパスのcortex/limbic同期**: add_memory, consolidate, build_schemas, correct_memory, interfere, sync_import, _snapshot_version

### Schema
- `cortex` テーブル新規作成（memoriesからデータ移行）
- `limbic` テーブル新規作成（memoriesからデータ移行）
- `memories_v` VIEW作成

## [v17.0] - 2026-04-04

### Added
- **フラッシュバルブ記憶**: 覚醒度0.65以上の記憶から最も情動的な一文（80文字以内）を自動抽出し `flashbulb` カラムに保存。recall時に🔥で表示。`--flashbulb` フラグで手動指定も可能
- **左脳/右脳スコアリング分離**: recall/searchのスコア計算を2系統に分離
  - `_left_score()`: 鮮度・参照数・信頼度・安定性（意味的・分析的因子）
  - `_right_score()`: 覚醒度・プライミング・場所細胞・気分一致性・flashbulb（情動的・直感的因子）
  - `corpus_callosum()`: 幾何平均ベースで左右を統合。balanceパラメータで重み調整
- **recallモード**: `recall --analytical`（左脳優勢 balance=0.3）/ `recall --emotional`（右脳優勢 balance=0.7）

### Changed
- **recall_polyphonic**: 声ごとに異なるbalanceを適用。共感=0.7（右脳優勢）、批判=0.5（均等）
- **promote_turns**: sleep時のリプレイでflashbulbを自動抽出して記憶に付与
- **consolidate_memories**: マージ時に覚醒度の高い方のflashbulbを保持
- **export/import/sync**: flashbulbフィールドに対応

### Schema
- `memories` テーブルに `flashbulb TEXT DEFAULT NULL` カラム追加（既存データはNULL）

## [v16.6] - 2026-03-31

### Changed
- **MD発言行にセッション識別子と時刻を追加**: `### 😙 User 🐙 13:42` のように動物絵文字＋時刻を表示。複数セッション同時使用時に発言の所属がわかる
  - 16種類の動物絵文字（🐱🐶🦊🐸🐙🦉🐻🐺🦈🐧🦎🐝🦋🐬🦅🐢）からセッションIDで決定論的に割り当て
  - セッション見出し（`##`）の色マーカーも動物に統一
- **Tool callの折りたたみをObsidian calloutに変更**: `<details>` → `> [!info]- 🔧 Tool calls`。編集モードでも折りたたみが効く
- **XMLタグのみの発言をMDからスキップ**: `<command-message>` 等のシステムタグだけの発言が空見出しとして書かれる問題を修正

### Fixed
- **セッション動物が毎回変わる問題**: Python `hash()` のランダム化を回避し、UUID先頭8文字を16進数として使用

## [v16.5] - 2026-03-31

### Added
- **Tool callのMD書き出し**: 会話終了時（Stop hook）にtranscript JSONLを走査し、Bash/Read/Edit/Write/Grep/GlobのツールコールをMDファイルに追記。SQLiteには書かない
  - Bash: コマンド（1行化・80文字切り詰め）+ 出力（500文字切り詰め）
  - Read/Edit/Write: ファイルパス
  - Grep/Glob: 検索パターン
  - Obsidian calloutブロックで折りたたみ表示

### Design rationale
MDの対話ログにユーザー発言とアシスタントのテキスト応答しか残っておらず、Bash実行などのtool callが完全に欠落していた。transcript JSONLにはtool_use/tool_resultブロックとして記録されているため、Stop hookでこれを読み取ってMDにだけ追記する。SQLiteにはテキスト応答のみを保存する従来の挙動を維持。

## [v16.4] - 2026-03-27

### Added
- **対話のマークダウン自動書き出し**: record_turn.pyのフックに連動し、全対話を日付単位のマークダウンファイルとしてリアルタイムに追記。Obsidian等の外部ツールで対話履歴を閲覧可能に
  - `turn_export.json` で設定を外出し（`enabled`, `output_dir`, `timezone_offset_hours`）
  - `output_dir` は `~` や `${HOME}` 等のプレースホルダに対応
  - 設定ファイルが無ければ機能OFF（既存環境に影響なし）
  - ファイルロック付きappend（複数ウィンドウで同時書き込みしても安全）
  - セッション単位の見出し分割、frontmatter付き

### Design rationale
raw_turnsに保存された対話ログはSQLiteの中に閉じており、Obsidianなどのノートツールから横断検索できなかった。record_turn.pyのフックに相乗りすることで、既存の保存処理に影響を与えずマークダウンへの同時書き出しを実現する。出力先やフォーマットは個人差が大きいため、設定ファイルで外出しにした。

## [v16.3] - 2026-03-25

### Added
- **Selecting強化 — 会話文脈からの自動記憶検索**: ユーザーが発言するたびに、発言内容からFTS5で関連記憶を自動検索し、stderrに出力。LLMが会話中に自然に関連記憶を取り込めるようになった
  - `record_turn.py` の `UserPromptSubmit` フックに `_context_search()` を追加
  - FTS5 OR検索 + rank + importance + arousal のスコアリング
  - 助詞・短すぎるトークンを除外するストップワード処理
  - 短い発言（< 15文字）、コマンド、XMLタグはスキップ
  - embeddingモデル不要（FTS5のみ、高速）

### Design rationale
recallは会話開始時の1回だけで、会話が進んで話題が変わっても新しい関連記憶が浮上しなかった。人間の脳は会話中に常に連想が走っている。UserPromptSubmitフックに相乗りすることで、毎発言ごとに軽量な文脈検索を実行し、関連記憶をLLMに提示する。

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
