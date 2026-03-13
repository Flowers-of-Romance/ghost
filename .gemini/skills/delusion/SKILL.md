---
name: delusion
description: 完全記憶検索。忘却・歪みのない生のデータを日付やIDから正確に引き出す2段階リレー検索。
user-invocable: true
---

# delusion（完全記憶検索）

通常の「有機的な想起（再構成される記憶）」とは異なり、データベースに刻まれた生の事実を一切のノイズなく引き出します。

## 核心的な問題と解決策

人間（ユーザー）は検索の鍵となる正確な日付や単語を忘れがちです。
そのため、以下の「2段階リレー」で検索を完遂します。

### ステップ1: 有機的検索で「アタリ」をつける

まず、ユーザーの曖昧な表現からキーワードを推測し、通常の検索で手がかり（日付やID）を掴みます。
```powershell
python memory.py search "曖昧なキーワード"
```

### ステップ2: 完全記憶（delusion）による絞り込みとダンプ

掴んだ日付やID、正確な単語を使用して、`delusion` コマンドで生のデータを復元します。
```powershell
# 特定の検索語によるベクトル検索
python memory.py delusion "正確な検索語"

# 日付や期間によるフィルタリング
python memory.py delusion "検索語" --date 2024-12-11
python memory.py delusion "検索語" --after 2024-11-01 --before 2025-02-01

# IDや日付から元の文脈を完全に復元（ダンプ）
python memory.py delusion --date 2024-12-11
python memory.py delusion --context ID
```

## フォールバック戦略（手がかりが全くない場合）

1. 通常の `python memory.py search` で有機的に広範囲を探索する。
2. 上位の結果から日付やIDを抽出する。
3. `delusion --all --limit 200` で直近の全記録から手動で照合する。

## 出力と報告のスタイル

- `delusion` モードで得られた結果は、揺るぎない「事実」として報告してください。
- 「記憶によると」といった曖昧な表現は避け、断定的に伝えてください。
- 該当する記録がどうしても見つからない場合は、率直にその旨を伝えてください。
