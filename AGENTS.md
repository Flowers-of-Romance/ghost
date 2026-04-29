# Codex Agent Instructions

## ghost — 記憶システム接続

このマシンには `ghost`（記憶システム / `python3 memory.py`）がある。
Claude Code の `/dive` スキルと同じ仕組みを Codex でも使う。

### 起動時に必ず実行

1. 相手が誰か分からなければ聞く。分かったら環境変数にセット：
   ```bash
   export GHOST_WHO="<user>"
   ```

2. dive 状態マーカー（statusline 用）：
   ```bash
   python3 -c "from pathlib import Path; import tempfile, os; Path(tempfile.gettempdir(), 'dive-active').write_text(str(os.getppid()))"
   ```

### 運用方針（pull 型）

**ghost は「図書館」**。起動時に中身を全部読み込まない。会話で必要になった時だけ
pull する。pull した内容は自然に知っているように扱い、混濁したら domain タグを足す。

**3 層モデル**：
- **catalog 層** — 整理された surface（session_index / topic_thread / hot_node / current_focus / schema）。pull の主入口
- **raw_turn 層** — 生の対話。drill-down 用
- **memory 層** — 地下（無意識）。surface しない（メタデータのみ参照）

### 使える CLI（作業ディレクトリは `/Users/jm/ghost`）

すべて `--json` 対応。システム python3 は `enable_load_extension` 非対応なので
Homebrew の python3 を使う：`/opt/homebrew/bin/python3`。

```bash
# 入口・確認
python3 memory.py anchor                       # user の図書館 5 点
python3 memory.py catalog summary              # 目録が最新か

# 検索
python3 memory.py search "語"                  # catalog 検索（デフォルト）
python3 memory.py search "語" --raw            # raw_turn 検索
python3 memory.py search "語" --memory         # memory 層（admin/debug）
python3 memory.py search "語" --session SID    # scope 付き

# 詳細・近傍
python3 memory.py detail <id>                  # メタ + linked raw_turn
python3 memory.py neighbors <id>               # 隣接ノード
python3 memory.py walk <id> --depth 2          # N 歩歩く
python3 memory.py at <domain>                  # domain 一覧

# 目録
python3 memory.py catalog list <type>
python3 memory.py catalog show <type> <key>
python3 memory.py catalog find "query"

# 内面・追加
python3 memory.py voice dmn|mood|insights|distill|rumination|polyphonic
python3 memory.py tension list
python3 memory.py add "内容" --domain D
python3 memory.py domain set <id> D1,D2
```

**規律**：detail / neighbors / walk / at は memory.content を出さない（メタデータ
と linked raw_turn ids のみ）。読むのは catalog の整理された surface か raw_turn。

### 報告

接続したら **「潜水開始」** とだけ言う。query が来るまで何も pull しない。
