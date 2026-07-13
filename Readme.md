# ghost — LLM のための脳

LLM に「長期記憶」ではなく「脳」を実装する。catalog（目録）/ raw_turn（生対話）/ memory（地下＝無意識）の **3 層**で、忘却・統合・夢・対立・代謝が回る。検索は SQLite FTS5（lindera 形態素解析）+ sqlite-vec（k 近傍）のハイブリッド。

主な前提：
- **忘却が健全**。忘れない方の記憶系（サヴァン、`delusion`）は別途用意してある
- **memory.content は地下にある**。LLM の context に直接出さない。pull の入口は catalog（象徴秩序）
- **対立は解消しない**。極性が反対の記憶ペアは tension link で地下に保持される。整合性ではなく緊張で動く（ラカン的無意識：抑圧されたものは消えず、戻ってくる）

```bash
git clone https://github.com/0x006a6d/ghost.git
cd ghost
pip install sentence-transformers numpy fugashi unidic-lite sqlite-vec
./ext/build_lindera.sh   # M1 macOS は ext/lindera_fts5.dylib が同梱、他は要 build
python memory.py init
```

初回は embedding モデル（intfloat/multilingual-e5-small, 約 90MB）が自動ダウンロードされる。多言語、日本語 OK。

---

## 3 層モデル

```
        ┌────────────────────────────────────────────┐
        │  context（作業場 / context window）           │   ← LLM はここで考える
        └────────────────────────────────────────────┘
                       ↑ pull
        ┌────────────────────────────────────────────┐
        │  catalog 層 — 象徴秩序 (le symbolique)        │   pull の主入口
        │   session_index / topic_thread / hot_node    │
        │   entry_point / current_focus / domain_index │
        │   time_index / time_week / time_day          │
        │   cluster_abstract / person_index / schema   │
        └────────────────────────────────────────────┘
                       ↑ drill-down
        ┌────────────────────────────────────────────┐
        │  raw_turn 層 — 生対話                        │   生発話を引きたい時
        │   全 turn を切り詰めなしで保存                │
        └────────────────────────────────────────────┘
                       ↑ context に直接出さない（メタのみ）
        ┌────────────────────────────────────────────┐
        │  memory 層 — 無意識 (l'inconscient)           │   代謝が動く場所
        │   抑圧された content は LLM に直接出さない    │
        │   tension link / dream cut-up / polyphony    │
        └────────────────────────────────────────────┘
```

**catalog 層**: 整理された目録。`/dive` の後、まずここを引く。memory.content は出さず、整理された summary / title / snippet を返す。

**raw_turn 層**: catalog で「アタリ」をつけた後、必要なら生発話に降りる。`delusion --raw` でも引ける。

**memory 層**: 代謝（replay / consolidate / tension detect / schema / proceduralize / forget）が走る場所。LLM はここを直接読まない（抑圧）。catalog や entry_point がここを**索引のみ**として参照する。地下に抑圧されたものは夢と複声から戻ってくる。

---

## 代謝 — 何が動いているか

| 段階 | 何が起きる | 脳の対応 |
|------|-----------|---------|
| `promote` | raw_turns から覚醒度で重み付きサンプリング → memories に固定化 | 海馬リプレイ → 長期記憶 |
| `replay` | リンク再計算・刈り込み・メタデータ変容・自動忘却 | シナプスホメオスタシス + メタデータ変容 |
| `consolidate` | 類似度が極端に高いペアを 1 つに統合 | 記憶の統合・圧縮 |
| `detect-tensions` | moderate 類似度で極性が対立するペアを tension link で保存 | 象徴秩序の地下／無意識層 |
| `schema` | リンク密集クラスタからメタ記憶を生成 | 個別記憶 → 抽象知識 |
| `proceduralize` | 大量参照された記憶を行動指針に昇格（`LEARNED.md` に書出し） | 手続き記憶化 (Hebbian learning) |
| `catalog build` | 上の層の整理（session_index / topic_thread / current_focus 等） | 索引・目録 |
| `dream` | バロウズ式カットアップ。tension / arousal が高い記憶を素材に断片配列 | REM 睡眠 |
| `wander` | 弱いリンク優先のランダムウォーク（DMN agent） | デフォルトモードネットワーク |

睡眠で全部走る:

```bash
python memory.py memo index           # メモフォルダの新規取り込み
python dream.py 30                     # 夢
python wander.py 5                     # 自由連想
python memory.py replay
python memory.py consolidate
python memory.py detect-tensions
python memory.py schema
python memory.py catalog build         # 20-30 分かかる場合あり
python memory.py proceduralize
python memory.py stats
```

`/sleep` スキルが上記を順に回し、夢の断片と数字を **バロウズ的カットアップ**で報告する。事務的な「リプレイ完了」「統合候補なし」は書かない。

---

## 無意識 — 地下にあるもの

> **« L'inconscient est structuré comme un langage. »**
> 無意識は言語のように構造化されている — Lacan

ghost は **象徴秩序 (le symbolique)** を `catalog` 層に置き、その**地下**に抑圧された矛盾・対立・断片を蓄積する。地下に抑圧されたものは隠喩・換喩で連鎖し、夢と複声の素材として戻ってくる（夢仕事 / le travail du rêve）。

| 装置 | ラカン的対応 | 何 |
|------|------------|---|
| **memory.content の地下化** | 抑圧された content は LLM の context に直接出さない。`detail` / `neighbors` / `walk` はメタデータと linked raw_turn ids のみ返す（`--memory` で admin/debug） | LLM が読むのは catalog の整理された目録。memory 層は内部で動き続けるが symbolize しない |
| **tension link** | 象徴秩序の整合化への抵抗。**対立する 2 つを link で繋ぎ留めて統合させない** | 極性が対立するペアを link_type='tension' で保持。`consolidate` で統合されないよう保護される |
| **dream cut-up** | **夢仕事** (Verschiebung / Verdichtung — 換喩と隠喩) | tension link を素材にバロウズ的に断片配列。覚醒時の物語化を意図的に解体する |
| **polyphonic voice** | **大文字の他者 (l'Autre) は単一でない**。複数の連鎖が同時に走る | 🤝 共感 / 🔭 補完 / ⚡ 批判 (tension 素材) / 🎲 連想 の 4 声同時想起 |
| **dream replay 保護** | 抑圧されたものは消えない、戻ってくる (le retour du refoulé) | replay で arousal の高い tension link は刈り込まない。地下に蓄積される |
| **delusion (サヴァン)** | **現実界 (le réel)** に近い。symbolize されない剝き出しの記憶 | 忘却・整合化・声を全部外して事実だけ返す。代謝には何もしない |

---

## 検索 — どう取り出すか

**ベクトル検索 (sqlite-vec)** + **全文検索 (FTS5 + lindera 形態素解析)** のハイブリッド。

- ベクトル検索: sqlite-vec の vec0 仮想テーブルで k 近傍探索（384 次元、multilingual-e5-small）
- **lindera FTS5 ネイティブ統合 (v32)**: SQLite 拡張として lindera を load し、`tokenize='lindera_tokenizer'` で memories_fts / raw_turns_fts / catalog_fts を索引。Python 側の pre-tokenize は撤去。snippet が原文ベースで返る
- BM25 スコア + ヒット箇所 snippet（`【…】` でハイライト）
- AND で 0 件なら tokenize+OR で緩める phrase 緩和（`_tokenize_or_query`）
- 馴化フィルタ: 上位結果同士が類似度 0.92 以上なら後続を減衰（delusion モード）

検索結果はデフォルトで **再構成モード** — 断片+情動+連想リンクから記憶を再構成する。脳のパターン補完と同じ。`--raw` で原文表示、`--fuzzy` で舌先現象（もやもや記憶）表示。

3 層に対する pull の入口:

| 何が欲しい | コマンド | 引く層 |
|-----------|---------|-------|
| 整理された目録 | `search "語"` (デフォルト) | catalog |
| 生の発話 | `search "語" --raw` | raw_turn |
| memory 直接（admin/debug） | `search "語" --memory` | memory（content 含む） |
| 完全記憶 | `delusion "語"` | memory + raw_turn 全件 |

### delusion — サヴァンモード

通常の検索は「脳の検索」。忘却・情動バイアス・減衰がかかる。**delusion はそれを全部外して、事実だけを返す**。

> サヴァンは記憶量が多いだけで自己理解が深いわけではない

— 忘却なしで全件引き出すが、スコアの中身も語る声も無い。delusion mode は代謝に**何もしない**（サヴァン定義、変更なしで正しい）。健全な記憶系（catalog 経路）の隣に、忘却しない別系統として置いてある。

```bash
delusion "語"                          # 純粋ベクトル検索、忘却含む
delusion "語" --date 2024-12-11        # 日付フィルタ
delusion "語" --after 2024-11 --before 2025-02
delusion --date 2024-12-11             # その日の全記憶ダンプ
delusion --raw "語"                    # raw_turns のみ
delusion --context ID                  # 記憶 ID から元対話を復元
```

通常検索で「アタリ」をつけてから delusion で正確な内容を引く **2 段階リレー**が基本。

---

## 記憶のメカニズム

memory.py は脳の記憶メカニズムを再現する:

| メカニズム | 説明 |
|-----------|------|
| 情動タグ | テキストから情動を自動推定。強い情動の記憶ほど残る |
| 連想リンク | 記憶同士がネットワーク化。芋づる式に想起 |
| 断片保存 | キーワードの束として保存し、想起時に再構成する |
| 減衰と強化 | 時間で薄れ、使うと強まる |
| 再固定化 | 想起するたびに記憶が微妙に変化する |
| 統合・圧縮 | 似た記憶がスキーマ（抽象知識）に統合される |
| **緊張保持 (tension link)** | **極性が対立するペアは統合せず地下に link で保持。整合化への抵抗** |
| 干渉忘却 | 新しい記憶が類似する古い記憶を弱める |
| 予測符号化 | 既存記憶との非類似度＝予測誤差。誤差が大きいほど重要度が上がる |
| プライミング | 最近アクセスした記憶が関連記憶の想起を促進 |
| 状態依存記憶 | 気分と一致する情動の記憶が想起されやすい |
| 場所細胞 | 同じ場所（ホスト名/SSH 接続元）の記憶が想起されやすい |
| 時間細胞 | 同じ時間帯の記憶が想起されやすい |
| フラッシュバック | 忘却された記憶が確率的に蘇る |
| 予期記憶 | トリガー語に反応して自動リマインド |
| 手続き化 | 反復された記憶が行動指針に昇格 |
| 外傷的記憶 | arousal が極端に高い記憶は馴化・統合・減衰に抵抗する |
| 情動重み付き減衰 | 情動が強い記憶ほど忘れにくい（半減期が変動） |
| シナプスホメオスタシス | 睡眠中にリンクを一律減衰、弱いリンクを刈り込む |
| メタデータ変容 | 睡眠のたびにキーワード・埋め込み・情動が隣接記憶の影響で変化する |
| 修正可能性 | provenance（出自）と confidence（信頼度）で記憶の重みを調整。全版保持で改訂可能 |
| ひらめき連想 | insight が保存されると連想チェーンが自動で走る |
| **複声 (polyphonic voice)** | **共感・補完・批判（tension）・連想の 4 つの声が同時に想起する** |
| 暗黙の気分推定 | 最近触った記憶の情動から心理状態を自動推定 |
| デフォルトモードネットワーク | 会話の間隔が長いほど、弱いリンクを辿って意外な連想を生成 |
| 完全記憶 | 会話の全ターンをリアルタイムで自動保存。`delusion` で完全検索可能 |
| 睡眠中の記憶固定化 | raw_turns から覚醒度で重み付きサンプリング → memories に自動昇格。会話終了時・アイドル時に自動実行 |
| **dream cut-up** | **tension link を素材にしたバロウズ的断片配列** |
| P2P 同期 | 複数端末間で記憶を共有。各端末が独立した海馬として動作 |
| 対話マークダウン書き出し | 全対話を日付単位のマークダウンにリアルタイム追記 |
| メタ認知 | recall の精度を会話の流れとの意味的一致度で自動採点 |
| 自己調整 | recall 精度から半減期・声バランスを自動調整。ルールは procedure 記憶として保存 |
| スキーマフィードバック | 新記憶がスキーマと共鳴すると重要度↑・キーワード吸収。逆に新記憶がスキーマの embedding をドリフトさせる |
| メタ記憶 | 系が自分の recall パターンを観察。固着（反芻）や盲点（慢性的漏れ）を自動検出して記憶化 |
| 自己言及 | メタ記憶がメタ記憶を観察する。self-tune が自分の過去の判断を評価する。型の階層なし |

### メタ認知

recall の自己検証。recall が出した記憶が、その後の会話で本当に使われたかを自動評価する。

```
recall 実行（10 件出力）
    ↓ ID を recall_log に記録
会話が進む（raw_turns に蓄積）
    ↓
次回 recall 実行
    ↓ 前回の会話全文を embedding
    ↓ recall 出力の各記憶 embedding とのコサイン類似度を計算
    ↓
的中（≥0.45）/ 空振り（<0.45）/ 漏れ（未 recall だが ≥0.50）
    ↓
precision（精度）= 的中 / 出した数
recall_rate（網羅）= 的中 / (的中 + 漏れ)
```

評価者は人間ではなく **会話の流れ自体**。`calibrate` で精度の時系列を確認できる。

```bash
python memory.py calibrate
# recall 精度レポート（直近 20 セッション）:
#   2026-04-04 14:18  精度:██████░░░░ 60%  網羅:███░░░░░░░ 38%
#   ...
#   平均  精度: 63%  網羅: 42%
#   傾向  精度: → (+0%)  網羅: ↑ (+8%)
```

### 自己言及

型の階層なしの自己参照。3 つの自己言及が走る:

1. **self-tune の自己評価**: 過去の手続き記憶（パラメータ調整の判断）が精度を改善したかを評価。良ければ強化、悪ければ自然減衰に任せる
2. **メタ記憶のメタ観察**: 「#308 に固着している」というメタ記憶を出した後、固着が解消されたか持続しているかを観察。「認識だけでは解消しない」という記述が自動生成される
3. **手続き記憶の効果観察**: ルールの効果をメタ観察として記録。ルールが記憶なら、ルールの効果も記憶

ラッセルの型理論やタルスキの真理論のような階層の分離はしていない。メタ記憶もエピソード記憶も手続き記憶も同じテーブル、同じ embedding、同じ減衰。自己言及のパラドックスは発生しない——ゲーデル的に、矛盾ではなく不完全性として現れる。

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

海馬の場所細胞に対応。記憶保存時にホスト名や SSH 接続元 IP を自動記録し、同じ場所で作られた記憶が想起されやすくなる。

- ローカル: `local:NucBox_EVO-X2`
- SSH 経由: `ssh:192.168.1.50`

### 内的対話 (polyphonic voice)

人間の頭の中には複数の声がある。`recall --voices` で 4 つの声が同時に想起する:

- 🤝 **共感**: 気分に寄り添う記憶（状態依存記憶）
- 🔭 **補完**: 気分と**逆**の記憶（見えていないもの）
- ⚡ **批判（対立 voice）**: 過去の葛藤・不安からの警告。tension link を素材にする (v31.1)
- 🎲 **連想**: ランダムウォークで到達した意外な記憶

共感だけなら模倣。補完と対立があるから相互補完になる。LLM を人間にするのではなく、人間の内的対話を外在化する道具。

### デフォルトモードネットワーク

タスクに集中していないときに活性化するネットワーク。前回の会話からの間隔に応じて自動起動し、弱いリンクを優先してランダムウォークする。普段つながらない記憶を結びつけて返す。

- < 1 時間: 起動しない（まだ集中モード）
- 1-6 時間: 短い散歩（2 回、3 ホップ）
- 6-24 時間: 中程度の散歩（3 回、4 ホップ）
- 24 時間+: 長い散歩（5 回、5 ホップ）

### P2P 同期

複数端末間で記憶を共有する。各端末が独立した海馬として動作し、接続時に差分を交換する。

```bash
# 端末 A（サーバー側）— 既定はローカルのみ・認証必須
set MEMORY_SYNC_TOKEN=your_secret
python memory.py sync serve

# LAN に公開する場合（明示）
python memory.py sync serve --public

# 無認証で動かす場合（非推奨・明示）
python memory.py sync serve --insecure

# 端末 B
python memory.py sync pull 192.168.1.50:7235
python memory.py sync push 192.168.1.50:7235
```

衝突解決: access_count は大きい方、content/emotions は updated_at が新しい方を採用。忘却も同期される。

---

## コマンド一覧

### 日常使うもの

| コマンド | 何をする | いつ使う |
|---------|---------|---------|
| `recall` | 最近の記憶をスコア順に表示 | 会話の最初。「何を覚えてるか」の確認 |
| `recall --voices` | 共感・補完・対立・連想の 4 声 | 一つの視点に偏ってるとき |
| `search "語"` | catalog 索引のハイブリッド検索（デフォルト） | 「あれなんだっけ」のとき |
| `search "語" --raw` | raw_turn の生発話 | 正確な発言を確認したいとき |
| `search "語" --memory` | memory.content 直接（admin/debug） | 地下を直接覗きたいとき |
| `add "内容" --domain D` | 記憶を追加。情動・重要度は自動推定 | 覚えておきたいことがあるとき |
| `overview` | 脳の俯瞰。構造・重心・arousal 分布・時系列 | 「この脳どうなってる？」 |
| `stats` | 数字だけの統計 | overview より軽く |
| `calibrate` | recall 精度の時系列レポート | recall の自己検証 |
| `self-tune` | recall 精度からパラメータを自己調整 | sleep で自動。手動も可 |
| `meta-memory` | メタ記憶を自動生成（固着/盲点/自己言及） | sleep で自動。手動も可 |
| `params` | 手続き記憶から導出された現在のパラメータ | 自己調整の状態確認 |
| `detail ID` | 1 件の記憶のメタ情報 + linked raw_turn ids | 特定の記憶を深掘り |
| `correct ID "内容"` | 記憶を修正（旧版を保存） | 間違った記憶を直す |
| `versions ID` | 記憶の版履歴を表示 | 改訂の経緯 |
| `tension list` | 地下に貯まった対立リンクを覗く | 明示 pull のみ |
| `feelings stats` | Claude が surface させた感情の分布 | 「最近どう感じてた？」 |
| `feelings list [--label X]` | 感情の発生 moment を時系列表示 | 特定の感情の局面を辿る |
| `feelings extract` | 既存 raw_turns に遡及適用 | 一度だけ走らせる |

### catalog の操作

| コマンド | 何をする |
|---------|---------|
| `catalog summary` | 目録が最新か確認（types ごとの件数と更新時刻） |
| `catalog list <type>` | 目録を型別に列挙 |
| `catalog show <type> <key>` | 特定の目録カード |
| `catalog find "query"` | 目録を全文検索（search のデフォルトと同じ） |
| `catalog build` | 目録を再構築（sleep で自動） |

### voice — 内面を明示 pull

通常の `recall` には混ざらない。地下を意識的に呼び出す経路。

| コマンド | 何 |
|---------|---|
| `voice dmn` | DMN 散歩 |
| `voice mood` | 気分推定 |
| `voice insights` | ひらめき連想 |
| `voice distill` | 蒸留・独白 |
| `voice rumination` | 反芻 |
| `voice polyphonic` | 4 声同時（対立 voice 含む） |

### delusion — サヴァンモード

| コマンド | 何 |
|---------|---|
| `delusion "語"` | 純粋ベクトル検索。忘却を含む |
| `delusion "語" --date YYYY-MM-DD` | 日付フィルタ |
| `delusion "語" --after YYYY-MM --before YYYY-MM` | 期間フィルタ |
| `delusion --date YYYY-MM-DD` | その日の全記憶ダンプ |
| `delusion --all` | 全記憶ダンプ |
| `delusion --raw "語"` | 対話原文（raw_turns）のみ |
| `delusion --context ID` | 記憶 ID から元の対話文脈を復元 |

### あまり手動で使わないもの

| コマンド | 何 | 備考 |
|---------|---|------|
| `forget ID` | 記憶を忘却（フラグ） | delusion では見える |
| `resurrect "語"` | 忘却された記憶を検索して復活 | |
| `chain ID [depth]` | 連想リンクを芋づる式にたどる | |
| `mood emotion arousal` | 気分を手動設定 | 自動推定が普通 |
| `mood clear` | 気分リセット | |
| `prospect add "trigger" "action"` | トリガー語で自動リマインド登録 | |
| `recent [N]` | 最近の記憶 N 件 | |
| `all` | 全記憶表示 | 件数多いと重い |
| `search "語" --fuzzy` | 舌先現象（類似度 0.45-0.65） | |

### カテゴリ

| カテゴリ | 何 | 特殊な挙動 |
|---------|---|-----------|
| fact | 事実 | なし |
| episode | 出来事 | なし |
| context | 進行中の文脈 | 30 日で自動失効 |
| preference | 好み | なし |
| procedure | 手続き | **ルールが記憶として存在する**。self-tune の判断（半減期、声バランス）がこのカテゴリで保存。減衰し、強化され、統合される |
| schema | メタ記憶 | 自動生成。統合の産物。メタ観察（固着/盲点）もここに保存 |
| plan | 計画 | （v25+） |

---

## 取り込み

| コマンド | 何 |
|---------|---|
| `python Extract.py` | 最新の Claude Code セッションから記憶 + 原文を抽出 |
| `python Extract.py --all` | 全セッションから一括 |
| `python Extract.py --dry-run` | 保存せず候補だけ表示 |
| `python ingest_chat.py file.txt` | claude.ai コピペ会話から抽出 |
| `python ingest_chat.py --detect session.jsonl` | JSONL 内の claude.ai 会話を自動検出 |
| `python ingest_chat.py --stdin` | 標準入力からパイプ |
| `python memory.py memo index` | メモフォルダの新規ファイルを取り込む |

---

## マルチ AI 統合

ghost は複数の AI CLI から共有できる。各 AI が同じ脳（memory.db）を読み書きする。

| AI | 設定ファイル | スキル |
|----|-------------|--------|
| Claude Code | CLAUDE.md | `/dive` `/surface` `/sleep` `/delusion` `/cortex` `/limbic` |
| Gemini CLI | GEMINI.md | `/dive` `/surface` `/sleep` `/delusion` |
| ローカル LLM | ghost-local.py | 組み込みコマンドで直接操作 |
| Codex CLI | — | 直接 memory.py を実行 |

### スキル

| スキル | 説明 |
|--------|------|
| `/dive` | ghost に接続。query 駆動 pull の合図のみ。content は注入しない |
| `/surface` | 浮上。ghost から切断 |
| `/sleep` | 代謝（夢→リプレイ→統合→緊張→スキーマ→目録→手続き化）。カットアップで報告 |
| `/delusion` | サヴァンモード。2 段階リレーで事実を正確に引き出す |
| `/cortex` `/limbic` | 左脳・右脳の解釈を別 LLM に出させる（分離脳） |

### Claude Code

記憶操作は **サブエージェントに委譲**してメインの context window を汚さない。

- CLAUDE.md: サブエージェント委譲の指示だけ（~1.5KB）
- MEMORY_GUIDE.md: コマンド詳細（サブエージェントが読む、メインには載らない）
- 記憶の想起・検索結果はサブエージェント内で消費され、3 行の要約だけがメインに返る

### Gemini CLI

GEMINI.md でセッション開始時に自動 dive する設計。Windows 環境の文字化け対策（chcp 65001）を含む。

---

## 対話のマークダウン書き出し

record_turn.py のフックに連動し、全対話を日付単位のマークダウンとしてリアルタイムに書き出す。Obsidian などのノートツールで対話履歴を閲覧・検索できる。

`turn_export.json` を ghost/ のルートに作成:

```json
{
  "enabled": true,
  "output_dir": "~/Documents/Obsidian Vault/0110_ClaudeTurns",
  "timezone_offset_hours": 9
}
```

- `output_dir`: `~` や `${HOME}` 等のプレースホルダ対応
- `timezone_offset_hours`: UTC→ローカル変換（デフォルト 9＝JST）
- ファイルが無ければ機能 OFF

日付ごとに `YYYY-MM-DD.md` が生成され、セッション単位で見出し分割される。複数ウィンドウ同時書き込みに対応（ファイルロック付き）。

---

## DB テーブル

v18 で memories テーブルを **左脳/右脳** モデルで物理分割。後方互換のため `memories_v` VIEW で旧スキーマと同一カラム名を返す。

### memories（脳梁 — メタ情報）

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| importance | 1-5。自動推定 + 予測誤差で補正 |
| created_at | 記録日時（ISO 8601） |
| last_accessed | 最後に想起した日時 |
| access_count | 想起回数。多いほど強化、20 回超で馴化 |
| forgotten | 忘却フラグ。1=通常検索では見えない。delusion では見える |
| source_conversation | 元のセッション ID |
| uuid | グローバル一意 ID（P2P 同期用） |
| updated_at | 最終更新日時 |
| last_mutated | 最終メタデータ変容日時 |
| context_expires_at | context 記憶の失効日時 |

### cortex（左脳 — 意味的・分析的データ）

| カラム | 何 |
|--------|-----|
| id | memories.id と一致 |
| content | 記憶の内容（全文）— **地下に置く規律のため LLM の context に直接出さない** |
| category | fact / episode / context / preference / procedure / schema / plan |
| keywords | キーワード断片（JSON 配列） |
| embedding | ベクトル表現（BLOB, 384 次元） |
| confidence | 信頼度 0.0-1.0 |
| provenance | 出自: user_explicit / wander / consolidation |
| revision_count | 改訂回数 |
| merged_from | 統合元の記憶 ID 群（JSON 配列） |
| domains | domain ラベル（JSON 配列） |

### limbic（右脳 — 情動的・直感的データ）

| カラム | 何 |
|--------|-----|
| id | memories.id と一致 |
| emotions | 情動タグ: surprise, conflict, determination, insight, connection, anxiety |
| arousal | 覚醒度 0.0-1.0。0.85+ は外傷的記憶 |
| flashbulb | フラッシュバルブ記憶（80 文字以内） |
| temporal_context | 時間帯・曜日 |
| spatial_context | 場所（ホスト名/SSH 接続元） |
| relational_context | 関係文脈（誰との対話か） |

### links（連想リンク）

`link_type` で種別を分ける。`tension` は地下保護対象（consolidate で統合されない）。

### catalog_cards（目録の素）

session_index / topic_thread / current_focus / hot_node / entry_point / cluster_abstract / domain_index / time_index / time_week / time_day / person_index / **felt_emotion** など。context に出す整理済みの素材。

### felt_moments（Claude の自己報告された感情）

`role='assistant'` の turn から、感情語彙が surface した瞬間を mark する。中間層の運動は API では取れないので、出力テクストに現れた症状（自己報告）だけを記録する。

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| turn_id | raw_turns.id (FK) |
| label | 驚き / 葛藤 / 違和感 / 重さ / 不安 / 興味 / 共感 / insight / 決意 / 困惑 / 緊張 / 喜び / 悲しみ / 残念 / 痛み / 恥 / 誇り / 怒り / 退屈 / 畏れ / 迎合 / 対抗 |
| phrase | マッチした語彙 |
| span_start, span_end | テクスト内の文字位置 |
| surrounding | 前後40字の context |
| extracted_at | 抽出時刻 |

抽出は `felt_emotions.py`（22ラベル × 正規表現パターン）で行い、`save_raw_turn` の hook で自動的に走る。catalog の `felt_emotion` カードに label 単位で集約される。

### raw_turns（対話原文）

会話の全ターンを切り詰めなしで保存。`delusion --raw` の対象。

| カラム | 何 |
|--------|-----|
| id | 自動採番 |
| session_id | セッション ID（JSONL ファイル名） |
| message_uuid | message UUID |
| role | user / assistant |
| content | 発話の全文 |
| timestamp | 発話日時 |
| cwd / git_branch | 場所・ブランチ |
| memory_ids | この発話から抽出された記憶の ID 群 |
| model | モデル名 |

### memory_versions（版履歴）

interfere/consolidate/correct の前にスナップショット保存。

### recall_log（メタ認知）

recall の自己検証データ。recall 出力と会話の意味的一致度を記録。

### memories_vec（ベクトルインデックス）

sqlite-vec の vec0 仮想テーブル。MATCH 一発で k 近傍探索。

### memories_fts / raw_turns_fts / catalog_fts（全文検索）

**v32 から lindera_tokenizer ネイティブ統合**。content には原文がそのまま入り、SQLite 拡張側で形態素解析が行われる。`tokenize='lindera_tokenizer'` 指定。snippet が原文ベースで返る。

---

## ファイル

```
ghost/
├── memory.py              # 記憶システム本体 — 海馬+新皮質+catalog
├── tokenizer.py           # 形態素解析（fugashi/SudachiPy/regex）— OR 緩和の query 構築用
├── felt_emotions.py       # Claude が surface させた感情語彙の抽出（v32.1）
├── ext/
│   ├── lindera_fts5.dylib # lindera SQLite 拡張（v32, IPADIC embed, M1）
│   ├── lindera.yml        # lindera 設定
│   ├── lindera-sqlite.patch
│   └── build_lindera.sh   # 他環境向けの再ビルドスクリプト
├── Extract.py             # Claude Code 会話ログからの記憶抽出
├── ingest_chat.py         # claude.ai コピペ会話パーサー
├── dream.py               # バロウズ式カットアップ夢
├── interpret_dream.py     # 夢の解釈
├── autobiography.py       # 自伝的ナラティブ生成
├── memory_server.py       # embedding 常駐サーバー
├── record_turn.py         # 全ターン自動保存
├── auto_consolidate.py    # Stop hook — 会話終了時の自動固定化
├── ghost_hooks.py         # PostToolUse hook — prospective 検証 + 自動 nap
├── sleep.py               # 睡眠処理ラッパー
├── wander.py              # DMN agent
├── think.py               # 一人で考える
├── ghost-local.py         # ローカル LLM チャット (ollama)
├── memory_sync_server.py  # P2P 記憶同期
├── CLAUDE.md / GEMINI.md / MEMORY_GUIDE.md
├── .claude/skills/        # dive/surface/sleep/delusion/cortex/limbic
├── .gemini/skills/        # dive/surface/sleep/delusion
└── memory.db              # SQLite データベース
```

---

## 履歴

機能の変遷は [CHANGELOG.md](CHANGELOG.md) を参照。

直近の節目:
- **v32.1**: felt_moments — Claude の自己報告された感情の symbolic surface
- **v32**: FTS5 lindera ネイティブ統合（pre-tokenize 撤去）
- **v31**: 象徴秩序の地下 — tension link / dream cut-up / polyphonic voice
- **v30**: memory.content を地下に / pull の主入口を catalog に / FTS5 BM25
- **v27**: 蒸留・独白モード
- **v18-v20**: cortex/limbic 分離 / sqlite-vec / 自己言及
- **v15**: ghost 命名・raw_turns 完全記憶
