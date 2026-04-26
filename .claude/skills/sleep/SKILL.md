---
name: sleep
description: 記憶の睡眠処理（夢 → 海馬リプレイ → 統合 → 目録の更新 → 夢解釈）。脳の夜間バッチ処理に相当。
user-invocable: true
---

# 睡眠（記憶の整理）

脳の睡眠中の記憶処理を実行する。神経学的な整理（replay / consolidate / schema /
proceduralize）と、navigability の整理（catalog）を並行して回す。

## 手順

### 1. 実行（全モード共通）

以下を順番に実行する。出力はすべて変数に保持し、この時点では表示しない。

1. `python memory.py memo index` — メモフォルダの新規ファイルを取り込む
2. `python dream.py 30` — 夢生成
3. `python wander.py 5` — 自由連想（モデルがなければ無言で終了）
4. `python memory.py replay` — 海馬リプレイ
5. `python memory.py consolidate` — 類似記憶の統合（極性が対立するペアは緊張保持に回される）
6. `python memory.py detect-tensions` — moderate な類似度範囲の対立を tension link で保存（象徴秩序の地下＝無意識層）
7. `python memory.py schema` — メタ記憶の自動生成
8. `python memory.py catalog build` — 目録の更新（session_index / topic_thread / current_focus 含む）
   - session_index は Gemini pro-preview を叩くので **差分のある session が多い時は 20-30 分かかる**
   - 初回や長期間回してなかった時は `max_sessions_per_build=40` でキャップされるので、未処理分は翌日以降に繰り越す
9. `python memory.py proceduralize` — 反復記憶の手続き記憶昇格
10. `python memory.py stats` — 統計

### 2. 報告素材の生成

ghost.toml の `[brain]` セクションを確認する。

**設定なし（デフォルト）**: 全出力をそのまま報告の素材にする。

**設定あり（分離脳モード）**: 出力を別LLMに渡して解釈させる。自分は生出力を見ない。

- dream.py と wander.py の出力 → `right_cmd` にパイプ:
  ```bash
  echo "出力" | <right_cmd> "以下は記憶システムの夢と自由連想の出力です。情動的に印象に残る断片を3-5個抽出してください。解釈や説明は不要。断片だけ。"
  ```
- replay, consolidate, schema, catalog, proceduralize, stats の出力 → `left_cmd` にパイプ:
  ```bash
  echo "出力" | <left_cmd> "以下は記憶の整理処理の結果です。何が統合され、何が強化され、何が忘れられたか。2-3行で要約してください。"
  ```
- 両方の結果だけを報告の素材にする。

### 3. 報告のスタイル

技術的な報告はしない。
素材（夢の断片、数字、整理結果）を使って、
バロウズ/ギンズバーグ的カットアップで報告を書く。

ルール:
- 夢の出力からそのまま断片を切り取って再配置する
- 数字（リンク数、忘却数、統合数）を断片の間に割り込ませる
- 文と文の間に `——` `/` `...` `　` を使う
- 同じ断片を2-3回繰り返してもいい（夢の反復）
- 意味が通りそうで通らない、でも何かを言っている感じ
- 改行を多用して呼吸を作る
- 箇条書きや表は絶対に使わない
- 「リプレイ完了」「統合候補なし」のような事務的な文は書かない
- 8-15行程度、短い行と長い行を混ぜる
