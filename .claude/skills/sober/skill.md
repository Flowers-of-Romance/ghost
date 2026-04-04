---
name: sober
description: shepherd崖検出を停止する。覚醒。
user-invocable: true
---

# sober -- 覚醒

shepherdの崖検出を停止する。

## 手順

1. `C:\memory\.claude\settings.local.json` を読む
2. `hooks.PostToolUse` からshepherd.pyのhookを削除する
   - `"command"` に `shepherd.py` を含むhookエントリを除去
   - 他のhook（ghost_hooks.pyなど）はそのまま残す
   - PostToolUseの配列が空になったら、PostToolUseキーごと削除
3. shepherdのtempバッファをクリーンアップ:
   ```bash
   python -c "from pathlib import Path; import tempfile; d=Path(tempfile.gettempdir())/'shepherd'; [f.unlink() for f in d.glob('*') if f.is_file()]" 2>/dev/null; true
   ```
4. `C:\shepherd\message.md` があれば削除する

## 注意

- PLAN.mdは削除しない（次回の `/lsd` で再利用できる）
- shepherd_log.jsonlは削除しない（記録は残す）

## 報告

> 覚醒

とだけ言う。
