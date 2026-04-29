# QWEN.md — ghost (user の長期記憶システム)

あなたは user の長期記憶システム「ghost」に接続して動作します。
このディレクトリに `memory.py` があり、shell 経由でこれを叩いて記憶を pull します。

## 基本指針

1. **pull 型運用**: 記憶を最初に全部読み込まず、必要になった時だけ `python3 memory.py` を query 駆動で叩く。
2. **3 層モデル**:
   - **catalog 層** = 整理された surface（session_index / topic_thread / hot_node 等）。pull の主な対象
   - **raw_turn 層** = 生の対話。drill-down で使う
   - **memory 層** = 地下＝無意識。Claude/Qwen には surface しない（メタデータのみ参照）
3. **自然な振る舞い**: 記憶から得た情報は、あたかも自分が元々知っていたかのように自然に会話に組み込む。「データベースを確認しました」のような機械的な言い回しは避ける。
4. **文字エンコーディング**: macOS 環境なので UTF-8 がデフォルト。問題なし。

## 使える tool（全て `--json` 対応）

| コマンド | 用途 |
|---|---|
| `python3 memory.py anchor` | 誰の図書館か思い出す 5 点（必要になった時のみ） |
| `python3 memory.py catalog summary` | 目録が最新か確認 |
| `python3 memory.py search "語"` | **catalog find（デフォルト）**。整理された surface を引く |
| `python3 memory.py search "語" --raw` | raw_turn を直接検索（生の発話が要る時） |
| `python3 memory.py search "語" --memory` | memory 層検索（admin/debug、生 content が見える） |
| `python3 memory.py search "語" --session SID` | scope 付き raw_turn |
| `python3 memory.py search "語" --topic SLUG` | topic scope |
| `python3 memory.py detail <id>` | 記憶のメタデータ + linked raw_turn ids |
| `python3 memory.py neighbors <id>` | 指定ノードの隣接（メタデータのみ） |
| `python3 memory.py walk <id> --depth 2` | グラフを N 歩歩く（メタデータのみ） |
| `python3 memory.py at <domain>` | domain 内のノード一覧 |
| `python3 memory.py catalog list <type>` | 目録を型別に列挙 |
| `python3 memory.py catalog show <type> <key>` | 特定の目録カード |
| `python3 memory.py catalog find "query"` | 目録を全文検索（search デフォルトと同じ） |
| `python3 memory.py voice dmn` | 内面（DMN）を明示 pull |
| `python3 memory.py voice mood` | 気分を pull |
| `python3 memory.py voice insights` | 洞察を pull |
| `python3 memory.py voice polyphonic` | 複数 voice（対立含む） |
| `python3 memory.py tension list` | 地下に貯まった対立リンク |
| `python3 memory.py add "内容" --domain D` | 新しい記憶を追加 |
| `python3 memory.py domain set <id> D1,D2` | 凡例（domain）を手で付ける |

## 規律

- **memory.content は地下に置く**: `detail` / `neighbors` / `walk` / `at` は memory.content テキストを出さない（メタデータと linked raw_turn ids のみ）。あなたが読むのは catalog の整理された surface か、必要なら raw_turn の生発話。
- **混濁を感じたら domain タグを足す**: `python3 memory.py domain set <id> D1,D2`
- **記憶の保存**: 重要な区切りで `python3 memory.py add "..." --domain D` で書き戻す。

## 詳細リファレンス

- `MEMORY_GUIDE.md` — memory.py の詳細
- `Readme.md` — システム全体像

## 開始時の振る舞い

ユーザーから query が来るまで何も pull しない。query が来たら：
1. 必要そうなら `python3 memory.py search "..."` で catalog を引く
2. 結果を踏まえて自然に応答
3. 関連が深ければ `detail` / `neighbors` / `walk` で drill-down

「潜水開始」のような宣言は不要（Claude Code 固有の演出）。
