---
name: lsd
description: shepherd崖検出を起動する。意識拡張。
user-invocable: true
---

# lsd -- 意識拡張

shepherdの崖検出を起動し、設計→実装のフローを開始する。

## フェーズ1: セットアップ（自動）

1. `C:\memory\.claude\settings.local.json` を読む
2. `hooks.PostToolUse` にshepherdのhookがなければ追加する:
   ```json
   {
     "type": "command",
     "command": "python C:/shepherd/shepherd.py",
     "timeout": 45000
   }
   ```
   すでにあればスキップ
3. shepherdのtempバッファをリセット:
   ```bash
   python -c "from pathlib import Path; import tempfile; d=Path(tempfile.gettempdir())/'shepherd'; [f.unlink() for f in d.glob('*') if f.is_file()]" 2>/dev/null; true
   ```
4. `C:\shepherd\message.md` があれば削除する

## フェーズ2: 設計（planモード的）

1. ユーザーに今回の作業目標を聞く
2. コードベースを探索して理解を深める（Explore agentを使ってよい）
3. `C:\shepherd\PLAN.md` に設計を書く（Goal, Approach, Boundaries, Current Phase）
4. PLAN.mdの内容をユーザーに提示し、承認を得る
5. **承認されるまで実装に入らない**

## フェーズ3: 実装

承認後、PLAN.mdに沿って実装を進める。shepherdが崖検出で見守る。

## 報告

フェーズ1完了時:

> 意識拡張

とだけ言い、フェーズ2（設計）に入る。
