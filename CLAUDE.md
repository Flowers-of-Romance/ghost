# CLAUDE.md - 記憶するClaude

脳を模した記憶システムがある。会話をまたいで文脈を保持する。

## 使い方

記憶操作は**サブエージェントに委譲**する。メインコンテキストを汚染しない。

### 会話の開始時

サブエージェント（subagent_type: "general-purpose"）を起動して以下を依頼:
- `MEMORY_GUIDE.md` を読んでコマンドを把握
- `python memory.py recall` で自動想起
- ユーザーの発言に関連する `python memory.py search "キーワード"`
- 結果を**3行以内に要約**して返す

### 会話の終了時

サブエージェントに依頼:
- `python memory.py add "内容" カテゴリ "出典"` で保存
- カテゴリ: fact / episode / context / preference / procedure

### ルール

- 「以前の記憶によると」とは言わない。自然に知っているように振る舞う
- 記憶の断片から再構成する。完全に正確でなくていい
- `/sleep` で睡眠処理（夢→リプレイ→統合）
