---
name: delusion
description: 完全記憶モード。忘却・バイアスなしで事実を正確に引き出す。2段階リレー検索。
user-invocable: true
---

# delusion -- 完全記憶検索

通常の有機的記憶（忘れる・歪む・偏る）ではなく、一切のノイズなく事実を完璧に引き出すモード。

## 核心的な問題

delusionは全部覚えているが、人間は検索キー（日付・正確な単語）を忘れている。
だから「有機的検索でアタリをつけてからdelusion検索する」2段階リレーが必要。

## 手順（Sonnetに委譲 → Opusが対話）

### ステップ1+2: 検索・原文取得・書き出し（Sonnet一括）

```
Agent({
  description: "delusion: 完全記憶検索",
  model: "sonnet",
  prompt: `
記憶データベースから情報を完全に検索・取得してファイルに書き出す。
作業ディレクトリは C:\\memory。

## 検索対象
ユーザーの質問: 「{ユーザーの質問}」

## 手順（3ステップ、最小tool_useで）

### 1. 広域検索（1コマンド）
質問から関連キーワードを3-5個推測し、--batchで一括検索:
python memory.py delusion --batch "{kw1}" "{kw2}" "{kw3}" "{kw4}" "{kw5}"

### 2. 原文取得（1コマンド）
検索結果からクエリに関連度の高いIDを最大10件選ぶ。
- task notification、無関係なエピソードは除外
- スコアだけでなくサマリの内容で関連を判断
選んだIDを --batch-context で一括取得:
python memory.py delusion --batch-context 36 raw:4728 337
（raw:XXX 形式のIDもそのまま渡せる。前後の対話文脈が返る）

### 3. ファイル書き出し
結果を C:\\memory\\.delusion\\{クエリ}_result.md に書き出す。
フォルダは既に存在する。

ファイル構成:

# delusion検索結果
クエリ: {クエリ}
検索日時: {日時}

## 検索ログ
{キーワード1} → {N}件
{キーワード2} → {N}件
...

## インデックス（選定後）
| ID | 日付 | スコア | 1行サマリ |
|----|------|--------|-----------|
| ... | ... | ... | ... |

## ID:XX の原文
{--batch-context の出力をそのまま貼る。一切編集しない}

## 重要な制約
- 原文は一切要約・編集するな。コマンド出力をそのまま貼れ
- 解釈・考察・補足セクションは追加するな
- tool_useは最小限に。--batch と --batch-context を使えば検索+取得は2コマンドで済む
- 最後に「書き出し完了: {N}件、約{X}KB」とだけ返せ
`
})
```

### ステップ3: ユーザーとの対話（Opus）

Sonnetから「書き出し完了」が返ったら:

1. `.delusion/{クエリ}_result.md` の先頭50行程度をReadしてインデックスを見せる
2. ユーザーが指定したIDの部分だけ Read（offset/limit）で読む
3. 必要に応じて追加の `--context` をOpusが直接叩いてもよい

**原則: ファイルを丸ごとReadしない。必要な部分だけ読む。**

## コマンド一覧

```bash
python memory.py delusion "検索語"                    # 純粋ベクトル検索
python memory.py delusion "検索語" --date 2024-12-11  # 日付フィルタ
python memory.py delusion "検索語" --after 2024-11-01 --before 2025-02-01  # 期間
python memory.py delusion --date 2024-12-11           # その日の全記憶ダンプ
python memory.py delusion --all                       # 全記憶ダンプ
python memory.py delusion --raw "検索語"              # 原文（raw_turns）のみ検索
python memory.py delusion --context ID                # 記憶IDから元の対話文脈を復元
python memory.py delusion --context raw:4728          # raw_turnの前後文脈を復元
python memory.py delusion --batch "q1" "q2" "q3"     # バッチ検索（複数キーワード一括、重複ID除外）
python memory.py delusion --batch-context 36 raw:4728 337  # 複数IDの文脈を一括取得
```

## Sonnet委譲をスキップしてよい場合

以下の場合はOpusが直接コマンドを叩く:
- ユーザーがID・日付を明示的に指定している（`--context`, `--date`）
- 検索キーワードが1つで明確（曖昧さがない）
- `--raw` や `--all` など特定オプションを指示された

## フォールバック戦略

「いつ」も「情動」も不明な場合:
1. まず通常の `python memory.py search` で有機的に検索（上位の日付やIDを手がかりにする）
2. 見つかった日付やIDを使って `delusion --date` や `delusion --context` で完全ダンプ
3. それでもダメなら `delusion --all --limit 200` で全件から手動で探す

## 出力フォーマット

```
[ID:982] [2024-12-11T15:30:00Z] [category:episode] [arousal:0.90]
SQLiteのトークンサイズ上限でクラッシュ。解決までに3時間かかった。
```

## 報告

delusionモードの結果は事実として報告する。「記憶によると」とは言わない。
結果が見つからない場合は正直に「該当する記録がない」と伝える。
