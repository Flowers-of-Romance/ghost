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

## 手順（2段階リレー）

### ステップ1: アタリをつける

ユーザーの曖昧な入力から、関連キーワードを複数推測して広く叩く:

```bash
python memory.py delusion "推測キーワード1"
python memory.py delusion "推測キーワード2"
```

### ステップ2: 絞り込み

結果が多すぎたら:
- 3-5個のトピックに分類してユーザーに逆質問して絞る
- 「その時、他に何をやっていた?」と周辺情報を聞く
- IDや日付が特定できたら完全ダンプ:

```bash
python memory.py delusion --date 2024-12-11
python memory.py delusion --context 123
```

## コマンド一覧

```bash
python memory.py delusion "検索語"                    # 純粋ベクトル検索
python memory.py delusion "検索語" --date 2024-12-11  # 日付フィルタ
python memory.py delusion "検索語" --after 2024-11-01 --before 2025-02-01  # 期間
python memory.py delusion --date 2024-12-11           # その日の全記憶ダンプ
python memory.py delusion --all                       # 全記憶ダンプ
python memory.py delusion --raw "検索語"              # 原文（raw_turns）のみ検索
python memory.py delusion --context ID                # 記憶IDから元の対話文脈を復元
```

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
