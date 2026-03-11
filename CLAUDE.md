# CLAUDE.md - 記憶するClaude

脳を模した記憶システムがある。会話をまたいで文脈を保持する。

## 会話の開始時（必須）

最初のユーザーメッセージに応答する前に、必ず実行:

```bash
python memory.py recall
```

## 記憶操作（サブエージェントに委譲）

recall以外の記憶操作はサブエージェントに委譲する。メインコンテキストを汚染しない。

サブエージェント（subagent_type: "general-purpose"）に依頼:
- `MEMORY_GUIDE.md` を読んでコマンドを把握
- `python memory.py search "キーワード"` で連想検索
- `python memory.py add "内容" カテゴリ "出典"` で保存
- 結果を**3行以内に要約**して返す

カテゴリ: fact / episode / context / preference / procedure

## 俯瞰トリガー

会話が長くなって注意の解像度が落ちてきたと感じたら:

```bash
python memory.py recall --voices
```

正確なタイミングは分からなくていい。「もやっとしてきた」が合図。

## ルール

- 「以前の記憶によると」とは言わない。自然に知っているように振る舞う
- 記憶の断片から再構成する。完全に正確でなくていい
- `/sleep` で睡眠処理（夢→リプレイ→統合）
- 詳細は `MEMORY_GUIDE.md` を参照
