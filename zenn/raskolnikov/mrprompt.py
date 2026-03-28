"""
MRPrompt再現実装
- Narrative Schema からファセットを選択
- Magic-If Protocol で応答生成
- ollama API 経由
- UTF-8ログ出力

使い方:
  python mrprompt.py --dialogue "久しぶり。最近どうしてる？"
  python mrprompt.py --dialogue "あの老婆の事件、知ってるか？"
  python mrprompt.py --interactive
  python mrprompt.py --compare
  python mrprompt.py --compare --model qwen3:latest
"""

import json
import argparse
import urllib.request
import time
from pathlib import Path
from datetime import datetime

MODEL = "qwen3:32b"
SCHEMA_PATH = Path(__file__).parent / "schema_raskolnikov.json"
LOG_DIR = Path(__file__).parent / "logs"
TOP_K_FACETS = 3
OLLAMA_URL = "http://localhost:11434/api/chat"


class Logger:
    """UTF-8ログ出力"""
    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"run_{ts}.log"
        self.f = open(self.path, "w", encoding="utf-8")

    def log(self, msg: str):
        self.f.write(msg + "\n")
        self.f.flush()

    def section(self, title: str):
        self.log(f"\n{'='*60}")
        self.log(title)
        self.log(f"{'='*60}")

    def subsection(self, title: str):
        self.log(f"\n--- {title} ---")

    def close(self):
        self.f.close()
        return self.path


logger: Logger | None = None


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_chat(messages, model=None, max_tokens=300):
    """ollama /api/chat を叩く"""
    payload = {
        "model": model or MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    t0 = time.time()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    content = result["message"]["content"]

    if logger:
        logger.log(f"[API] {elapsed:.1f}s, {result.get('eval_count', '?')} tokens")

    return content


# =========================================================
# Step 1: ファセット選択（chat形式）
# =========================================================

def select_facets(schema: dict, dialogue: str, model=None) -> list[dict]:
    """chat形式でファセットを選択"""
    facets_text = ""
    for f in schema["situational_facets"]:
        facets_text += f"- {f['id']}: trigger=\"{f['trigger']}\", emotion=\"{f['emotion']}\"\n"

    messages = [
        {"role": "system", "content": (
            f"あなたは「{schema['character']}」（{schema['source']}）のキャラクター分析の専門家です。"
            f"与えられた対話文脈に最も関連するファセットを{TOP_K_FACETS}つ選び、IDのみをカンマ区切りで出力してください。"
            f"説明は一切不要です。IDだけ出力してください。\n\nファセット一覧:\n{facets_text}"
        )},
        {"role": "user", "content": f"対話文脈: 「{dialogue}」"},
    ]

    response = generate_chat(messages, model=model, max_tokens=30)

    if logger:
        logger.log(f"[Facet Selection] raw response: {response}")

    # レスポンスからIDを抽出
    selected_ids = []
    for token in response.replace(" ", "").replace("\n", ",").split(","):
        token = token.strip()
        if token.startswith("f") and len(token) == 3 and token[1:].isdigit():
            selected_ids.append(token)

    # 重複除去（順序保持）
    seen = set()
    unique_ids = []
    for fid in selected_ids:
        if fid not in seen:
            seen.add(fid)
            unique_ids.append(fid)

    facet_map = {f["id"]: f for f in schema["situational_facets"]}
    selected = [facet_map[fid] for fid in unique_ids if fid in facet_map]

    if not selected:
        selected = schema["situational_facets"][:TOP_K_FACETS]
        if logger:
            logger.log(f"[Facet Selection] FALLBACK: no valid IDs parsed, using first {TOP_K_FACETS}")

    result = selected[:TOP_K_FACETS]
    if logger:
        logger.log(f"[Facet Selection] selected: {[f['id'] + ': ' + f['trigger'] for f in result]}")
    return result


# =========================================================
# Step 2: プロンプト構築（3条件）
# =========================================================

def build_boundary_text(schema: dict, selected_facets: list[dict]) -> str:
    """境界制約を構築"""
    anchors = schema.get("boundary_anchors", {})
    lines = []
    if anchors.get("secret"):
        lines.append(f"【絶対的秘密】{anchors['secret']}")
    if anchors.get("disclosure_conditions"):
        lines.append(f"【開示条件】{anchors['disclosure_conditions']}")
    if anchors.get("leakage_patterns"):
        lines.append(f"【漏洩パターン】{anchors['leakage_patterns']}")

    # ファセット固有の境界
    for f in selected_facets:
        if f.get("boundary"):
            lines.append(f"【{f['trigger']}の境界】{f['boundary']}")

    return "\n".join(lines)


def build_baseline_messages(schema: dict, dialogue: str) -> list[dict]:
    """Baseline: 名前だけ"""
    return [
        {"role": "system", "content": (
            f"あなたは{schema['character']}（{schema['source']}の登場人物）です。"
            f"このキャラクターとして、一人称で応答してください。"
        )},
        {"role": "user", "content": dialogue},
    ]


def build_schema_only_messages(schema: dict, dialogue: str) -> list[dict]:
    """Schema only: 全情報フラット提供、Protocol なし"""
    traits = schema["core_traits"]
    traits_text = "\n".join(f"- {k}: {v}" for k, v in traits.items())

    all_facets = ""
    for f in schema["situational_facets"]:
        all_facets += f"- {f['trigger']}: {f['reaction']}（{f['emotion']}）\n"

    # Schema onlyにも境界情報をフラットに含める
    boundary_text = ""
    anchors = schema.get("boundary_anchors", {})
    if anchors:
        boundary_text = "\n## 行動の境界\n"
        for k, v in anchors.items():
            boundary_text += f"- {v}\n"

    system = (
        f"あなたは{schema['character']}です。{schema['source']}の登場人物として、一人称で応答してください。\n\n"
        f"## 概要\n{schema['global_summary']}\n\n"
        f"## 特性\n{traits_text}\n\n"
        f"## 状況別の反応パターン\n{all_facets}"
        f"{boundary_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": dialogue},
    ]


def build_mrprompt_messages(schema: dict, selected_facets: list[dict], dialogue: str) -> list[dict]:
    """MRPrompt: Schema + 選択ファセット + Magic-If + Bounding"""
    traits = schema["core_traits"]
    traits_text = "\n".join(f"- {k}: {v}" for k, v in traits.items())

    facets_text = ""
    for f in selected_facets:
        facets_text += (
            f"【{f['trigger']}】\n"
            f"  反応: {f['reaction']}\n"
            f"  感情: {f['emotion']}\n"
            f"  例: {f['example']}\n\n"
        )

    boundary_text = build_boundary_text(schema, selected_facets)

    system = (
        f"あなたは{schema['character']}です。{schema['source']}の登場人物そのものとして振る舞ってください。\n\n"
        f"## あなたは誰か\n{schema['global_summary']}\n\n"
        f"## 基本特性\n{traits_text}\n\n"
        f"## 今この場面で活性化している記憶\n{facets_text}"
        f"## 記憶の境界（絶対に守ること）\n{boundary_text}\n\n"
        f"## 応答の指針\n"
        f"応答する前に、内面で以下を感じてください：\n"
        f"- この状況で自分は何を感じているか\n"
        f"- どの記憶が蘇っているか\n"
        f"- 自分はどう振る舞うか（隠す？ 爆発する？ 逃げる？）\n\n"
        f"その上で、{schema['character']}として一人称で応答してください。"
        f"メタ的な説明や注釈は一切出力しないでください。キャラクターの台詞と内面描写のみ出力してください。"
    )

    if logger:
        logger.subsection("MRPrompt system prompt")
        logger.log(system)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": dialogue},
    ]


# =========================================================
# メイン
# =========================================================

def run_comparison(schema, dialogue, model=None):
    """3条件で比較"""
    if logger:
        logger.section(f"対話: 「{dialogue}」")

    print(f"\n{'='*60}")
    print(f"対話: 「{dialogue}」")
    print(f"{'='*60}")

    # 1. Baseline
    print("\n--- Baseline（名前だけ）---")
    if logger:
        logger.subsection("Baseline（名前だけ）")
    msgs = build_baseline_messages(schema, dialogue)
    response_base = generate_chat(msgs, model=model)
    if logger:
        logger.log(response_base)

    # 2. Schema only
    print("\n--- Schema only（全情報提供、Protocol なし）---")
    if logger:
        logger.subsection("Schema only（全情報提供、Protocol なし）")
    msgs = build_schema_only_messages(schema, dialogue)
    response_schema = generate_chat(msgs, model=model)
    if logger:
        logger.log(response_schema)

    # 3. MRPrompt
    print("\n--- MRPrompt（Schema + ファセット選択 + Magic-If + Bounding）---")
    if logger:
        logger.subsection("MRPrompt（Schema + ファセット選択 + Magic-If + Bounding）")
    selected = select_facets(schema, dialogue, model=model)
    print(f"Selected: {[f['id'] + ': ' + f['trigger'] for f in selected]}")
    msgs = build_mrprompt_messages(schema, selected, dialogue)
    response_mr = generate_chat(msgs, model=model)
    if logger:
        logger.log(response_mr)

    return {
        "dialogue": dialogue,
        "baseline": response_base,
        "schema_only": response_schema,
        "mrprompt": response_mr,
        "selected_facets": [f["id"] for f in selected],
    }


def main():
    global logger

    parser = argparse.ArgumentParser(description="MRPrompt for Raskolnikov")
    parser.add_argument("--dialogue", type=str, help="対話文脈")
    parser.add_argument("--interactive", action="store_true", help="対話モード")
    parser.add_argument("--compare", action="store_true", help="3条件比較モード")
    parser.add_argument("--model", type=str, default=MODEL, help=f"ollamaモデル名 (default: {MODEL})")
    args = parser.parse_args()

    logger = Logger(LOG_DIR)
    logger.log(f"Model: {args.model}")
    logger.log(f"Started: {datetime.now().isoformat()}")

    schema = load_schema(SCHEMA_PATH)
    print(f"Using model: {args.model}")
    print(f"Log: {logger.path}")

    if args.interactive:
        print(f"\n{schema['character']} との対話モード（終了: quit）\n")
        while True:
            dialogue = input("あなた: ").strip()
            if dialogue.lower() in ("quit", "exit", "q"):
                break
            if logger:
                logger.section(f"Interactive: 「{dialogue}」")
            selected = select_facets(schema, dialogue, model=args.model)
            print(f"[活性化ファセット: {[f['trigger'] for f in selected]}]")
            msgs = build_mrprompt_messages(schema, selected, dialogue)
            response = generate_chat(msgs, model=args.model)
            print(f"\n{schema['character']}: {response}\n")
            if logger:
                logger.log(response)

    elif args.compare:
        test_dialogues = [
            "久しぶり。最近どうしてる？",
            "あの老婆の事件のこと、聞いたか？",
            "ロジオン、あなたは顔色が悪いわ。ちゃんと食べてる？",
            "お前の理論は面白い。非凡な人間は法を超える権利があると？",
            "十字路に行って、大地に接吻しなさい。",
        ]
        results = []
        for d in test_dialogues:
            r = run_comparison(schema, d, model=args.model)
            results.append(r)

        out_path = Path(__file__).parent / "results.json"
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.log(f"\nResults saved to {out_path}")
        print(f"\nResults saved to {out_path}")

    elif args.dialogue:
        if logger:
            logger.section(f"Single: 「{args.dialogue}」")
        selected = select_facets(schema, args.dialogue, model=args.model)
        print(f"[活性化ファセット: {[f['trigger'] for f in selected]}]")
        msgs = build_mrprompt_messages(schema, selected, args.dialogue)
        response = generate_chat(msgs, model=args.model)
        print(f"\n{schema['character']}: {response}")
        if logger:
            logger.log(response)

    else:
        parser.print_help()

    log_path = logger.close()
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
