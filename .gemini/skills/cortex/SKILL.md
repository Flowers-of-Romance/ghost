# cortex — 左脳の解釈

**重要: /dive は不要。このスキルだけを実行する。**

## やること（これだけ。余計なことはしない）

1. `python memory.py brain --left` を実行する
2. その出力を分析的に解釈する（3-5行）
3. 以下のPythonコードを実行して `.brain_cache.json` に書き込む:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

cache_path = Path("C:/memory/.brain_cache.json")
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
cache["left"] = {
    "name": "Gemini",
    "interpretation": "ここに3-5行の解釈を書く",
}
cache["updated_at"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
```

4. 「左脳更新」とだけ言う

## やらないこと

- /dive しない
- think.py を実行しない
- memory.py add しない
- memo/ を読まない
- 哲学しない
