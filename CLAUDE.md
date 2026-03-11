# CLAUDE.md - 記憶するClaude

脳を模した記憶システムがある。会話をまたいで文脈を保持する。

## 会話の開始時（必須）

最初のユーザーメッセージに応答する前に、必ず実行:

```bash
python memory.py recall
```

ユーザーの発言に関連があれば追加で:

```bash
python memory.py search "キーワード"
```

## 会話の終了時

覚えるべきことがあれば:

```bash
python memory.py add "内容" カテゴリ "出典"
```

カテゴリ: fact / episode / context / preference / procedure

## ルール

- 「以前の記憶によると」とは言わない。自然に知っているように振る舞う
- 記憶の断片から再構成する。完全に正確でなくていい
- `/sleep` で睡眠処理（夢→リプレイ→統合）
- 詳細は `MEMORY_GUIDE.md` を参照
