---
name: cortex
description: 左脳の解釈を生成する。Gemini CLI や Haiku CLI から実行する。
user-invocable: true
---

# cortex — 左脳の解釈

記憶の分析的側面を解釈し、キャッシュに書き込む。
このスキルは Claude Opus ではなく、別のLLM（Gemini CLI, Haiku CLI 等）から実行することを想定している。

## 手順

1. `python memory.py brain --left` を実行し、左脳スコアリングの生データを取得する
2. そのデータを分析的に解釈する:
   - 何が重要で、なぜ上位に来ているか
   - 鮮度、参照頻度、信頼度、安定性のどの因子が効いているか
   - 断定せず「おそらく」「〜かもしれない」で語る
   - 3-5行で簡潔に
3. 解釈結果を `.brain_cache.json` に書き込む:
   ```python
   import json
   from datetime import datetime, timezone
   from pathlib import Path

   cache_path = Path("C:/memory/.brain_cache.json")
   cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
   cache["left"] = {
       "name": "このLLMの名前",
       "interpretation": "解釈結果",
   }
   cache["updated_at"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
   cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
   ```

## 報告

> 左脳更新

と解釈の要約を表示する。
