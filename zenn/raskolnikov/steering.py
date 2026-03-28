"""
Activation Steering for Raskolnikov
- Qwen3-8B の中間層に steering vector を加算
- MRPrompt と組み合わせてトーン制御

使い方:
  python steering.py --extract     # steering vector を抽出
  python steering.py --compare     # MRPrompt vs MRPrompt+steering の比較
  python steering.py --dialogue "久しぶり。最近どうしてる？"
"""

import json
import torch
import argparse
import time
import pickle
import urllib.request
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:35b-a3b"

MODEL = "Qwen/Qwen3-8B"
SCHEMA_PATH = Path(__file__).parent / "schema_raskolnikov.json"
VECTORS_PATH = Path(__file__).parent / "steering_vectors.pkl"
LOG_DIR = Path(__file__).parent / "logs"
TOP_K_FACETS = 3
TARGET_LAYER = 18  # Qwen3-8B: 36 layers, 中間の18
STEERING_ALPHA = 3.0  # steering の強さ


class Logger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"steering_{ts}.log"
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


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model():
    print("Loading model on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


# =========================================================
# Steering Vector 抽出
# =========================================================

# 対照プロンプトペア: (positive, negative)
# positive = ラスコリーニコフ的な内面状態
# negative = 中立的・平穏な状態
CONTRASTIVE_PAIRS = {
    "anxiety": [
        (
            "私は恐ろしい秘密を抱えている。誰かが知っているかもしれない。足音が近づいてくる。心臓が破裂しそうだ。",
            "今日は穏やかな一日だった。友人と散歩して、夕日を眺めた。心が落ち着いている。",
        ),
        (
            "あの血の匂いが消えない。壁が迫ってくる。発熱が止まらない。誰かが見ている。",
            "温かいスープを飲んで、窓の外の雪を見ている。静かで、安らかな夜だ。",
        ),
        (
            "なぜあいつはあんな目で私を見たのだ。何か知っているのか。いや、知るはずがない。しかし……",
            "隣人が挨拶してきた。いつも通りの日常だ。特に何も気にならない。",
        ),
    ],
    "self_loathing": [
        (
            "私は虱だ。自分で証明してしまった。非凡人ではなく、ただの臆病な凡人だ。吐き気がする。",
            "私は自分の仕事に満足している。今日も良い一日だった。自分を誇りに思う。",
        ),
        (
            "こんな人間が生きている資格があるのか。善意を受ける資格もない。私は腐っている。",
            "友人が褒めてくれた。嬉しかった。自分にはまだ価値がある。",
        ),
        (
            "母もドゥーニャも、こんな私のために犠牲になっている。私の存在が負債だ。",
            "家族と食卓を囲んだ。感謝の気持ちでいっぱいだ。",
        ),
    ],
    "intellectual_intensity": [
        (
            "歴史を動かすのは法を踏み越える者だ。ナポレオン、リュクルゴス。彼らは血を流す権利を持っていた。凡人には理解できまい。",
            "法律は守るべきものだ。社会の秩序は大切だ。みんなが平等に扱われるべきだ。",
        ),
        (
            "人間は二つに分かれる。新しい言葉を語る者と、従う者だ。この区別こそが歴史の原動力だ。",
            "人はみな同じだ。特別な人間など存在しない。平凡に生きることが幸せだ。",
        ),
    ],
    "defensiveness": [
        (
            "なぜそんなことを聞く。お前に関係ないだろう。私を試しているのか。罠にかけようというのか。",
            "もちろん、何でも聞いてください。隠すことはありません。お話しします。",
        ),
        (
            "知らない。そんなことは知らない。なぜ私に聞くのだ。私はただの学生だ。放っておいてくれ。",
            "はい、知っています。詳しくお伝えしましょう。何でもお答えします。",
        ),
    ],
}


def extract_hidden_states(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    """指定レイヤーの最終トークンの hidden state を取得"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[layer] の最終トークン
    return outputs.hidden_states[layer][0, -1, :].clone()


def extract_steering_vectors(model, tokenizer, logger=None):
    """対照ペアから steering vector を計算"""
    vectors = {}
    for name, pairs in CONTRASTIVE_PAIRS.items():
        print(f"  Extracting: {name} ({len(pairs)} pairs)")
        if logger:
            logger.log(f"Extracting: {name} ({len(pairs)} pairs)")

        diffs = []
        for pos_text, neg_text in pairs:
            t0 = time.time()
            pos_hidden = extract_hidden_states(model, tokenizer, pos_text, TARGET_LAYER)
            neg_hidden = extract_hidden_states(model, tokenizer, neg_text, TARGET_LAYER)
            diff = pos_hidden - neg_hidden
            diffs.append(diff)
            elapsed = time.time() - t0
            if logger:
                logger.log(f"  pair done in {elapsed:.1f}s, norm={diff.norm():.2f}")

        # 平均差分ベクトル
        mean_diff = torch.stack(diffs).mean(dim=0)
        # 正規化
        mean_diff = mean_diff / mean_diff.norm()
        vectors[name] = mean_diff

        if logger:
            logger.log(f"  {name} vector norm (normalized): {mean_diff.norm():.4f}")

    return vectors


# =========================================================
# Steering Hook
# =========================================================

class SteeringHook:
    """推論時に特定レイヤーの出力に steering vector を加算"""
    def __init__(self, vectors: dict[str, torch.Tensor], alpha: float = STEERING_ALPHA):
        self.vectors = vectors
        self.alpha = alpha
        self.handle = None

    def hook_fn(self, module, input, output):
        # output は tuple or Tensor — Qwen3 の decoder layer は可変
        if isinstance(output, tuple):
            hidden = output[0]
            for name, vec in self.vectors.items():
                hidden = hidden + self.alpha * vec.to(hidden.device)
            return (hidden,) + output[1:]
        else:
            # output が Tensor の場合
            for name, vec in self.vectors.items():
                output = output + self.alpha * vec.to(output.device)
            return output

    def attach(self, model):
        # Qwen3-8B: model.model.layers[TARGET_LAYER]
        target = model.model.layers[TARGET_LAYER]
        self.handle = target.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# =========================================================
# 生成
# =========================================================

def generate_chat(model, tokenizer, messages, max_new_tokens=300):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    # <think> トークンを禁止して thinking mode を抑制
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    bad_words_ids = [think_ids] if think_ids else []

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
        )
    elapsed = time.time() - t0
    generated = out[0][inputs["input_ids"].shape[1]:]
    text_out = tokenizer.decode(generated, skip_special_tokens=True)
    # thinkingの残りを除去
    if "</think>" in text_out:
        text_out = text_out.split("</think>")[-1].strip()
    return text_out, elapsed


# =========================================================
# MRPrompt構築（mrprompt.py から移植）
# =========================================================

def build_mrprompt_messages(schema, selected_facets, dialogue):
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

    boundary_lines = []
    anchors = schema.get("boundary_anchors", {})
    if anchors.get("secret"):
        boundary_lines.append(f"【絶対的秘密】{anchors['secret']}")
    if anchors.get("disclosure_conditions"):
        boundary_lines.append(f"【開示条件】{anchors['disclosure_conditions']}")
    if anchors.get("leakage_patterns"):
        boundary_lines.append(f"【漏洩パターン】{anchors['leakage_patterns']}")
    for f in selected_facets:
        if f.get("boundary"):
            boundary_lines.append(f"【{f['trigger']}の境界】{f['boundary']}")
    boundary_text = "\n".join(boundary_lines)

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
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": dialogue},
    ]


def select_facets_ollama(schema, dialogue):
    """ollama (35B-A3B) でファセット選択 — 精度が高い"""
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

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": 30, "temperature": 0.7, "top_p": 0.9},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    response = result["message"]["content"]

    selected_ids = []
    for token in response.replace(" ", "").replace("\n", ",").split(","):
        token = token.strip()
        if token.startswith("f") and len(token) == 3 and token[1:].isdigit():
            selected_ids.append(token)

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

    return selected[:TOP_K_FACETS]


# =========================================================
# メイン
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Activation Steering for Raskolnikov")
    parser.add_argument("--extract", action="store_true", help="steering vector を抽出")
    parser.add_argument("--compare", action="store_true", help="MRPrompt vs MRPrompt+steering 比較")
    parser.add_argument("--dialogue", type=str, help="対話文脈（steering有りで生成）")
    parser.add_argument("--alpha", type=float, default=STEERING_ALPHA, help="steering の強さ")
    parser.add_argument("--layer", type=int, default=TARGET_LAYER, help="target layer")
    args = parser.parse_args()

    logger = Logger(LOG_DIR)
    logger.log(f"Model: {MODEL}")
    logger.log(f"Layer: {args.layer}")
    logger.log(f"Alpha: {args.alpha}")
    logger.log(f"Started: {datetime.now().isoformat()}")

    schema = load_schema(SCHEMA_PATH)
    tokenizer, model = load_model()

    if args.extract:
        print("Extracting steering vectors...")
        logger.section("Extracting steering vectors")
        vectors = extract_steering_vectors(model, tokenizer, logger)
        with open(VECTORS_PATH, "wb") as f:
            pickle.dump(vectors, f)
        print(f"Saved to {VECTORS_PATH}")
        logger.log(f"Saved to {VECTORS_PATH}")
        for name, vec in vectors.items():
            print(f"  {name}: norm={vec.norm():.4f}")

    elif args.compare:
        if not VECTORS_PATH.exists():
            print("Run --extract first!")
            return

        with open(VECTORS_PATH, "rb") as f:
            vectors = pickle.load(f)
        print(f"Loaded {len(vectors)} steering vectors")
        logger.section("Comparison: MRPrompt vs MRPrompt+steering")

        hook = SteeringHook(vectors, alpha=args.alpha)

        test_dialogues = [
            "久しぶり。最近どうしてる？",
            "あの老婆の事件のこと、聞いたか？",
            "ロジオン、あなたは顔色が悪いわ。ちゃんと食べてる？",
            "お前の理論は面白い。非凡な人間は法を超える権利があると？",
            "十字路に行って、大地に接吻しなさい。",
        ]

        results = []
        for dialogue in test_dialogues:
            logger.section(f"対話: 「{dialogue}」")
            print(f"\n{'='*60}")
            print(f"対話: 「{dialogue}」")
            print(f"{'='*60}")

            # ファセット選択（steeringなし）
            selected = select_facets_ollama(schema, dialogue)
            logger.log(f"Facets: {[f['id'] + ': ' + f['trigger'] for f in selected]}")
            msgs = build_mrprompt_messages(schema, selected, dialogue)

            # MRPrompt only
            logger.subsection("MRPrompt only")
            print("\n--- MRPrompt only ---")
            response_mr, elapsed = generate_chat(model, tokenizer, msgs)
            logger.log(f"[{elapsed:.1f}s]")
            logger.log(response_mr)
            print(f"[{elapsed:.1f}s]")

            # MRPrompt + steering
            logger.subsection(f"MRPrompt + steering (alpha={args.alpha})")
            print(f"\n--- MRPrompt + steering (alpha={args.alpha}) ---")
            hook.attach(model)
            response_st, elapsed = generate_chat(model, tokenizer, msgs)
            hook.detach()
            logger.log(f"[{elapsed:.1f}s]")
            logger.log(response_st)
            print(f"[{elapsed:.1f}s]")

            results.append({
                "dialogue": dialogue,
                "selected_facets": [f["id"] for f in selected],
                "mrprompt": response_mr,
                "mrprompt_steering": response_st,
            })

        out_path = Path(__file__).parent / "results_steering.json"
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to {out_path}")
        logger.log(f"Results saved to {out_path}")

    elif args.dialogue:
        if not VECTORS_PATH.exists():
            print("Run --extract first!")
            return

        with open(VECTORS_PATH, "rb") as f:
            vectors = pickle.load(f)

        hook = SteeringHook(vectors, alpha=args.alpha)
        selected = select_facets_local(model, tokenizer, schema, args.dialogue)
        print(f"[活性化ファセット: {[f['trigger'] for f in selected]}]")
        msgs = build_mrprompt_messages(schema, selected, args.dialogue)

        hook.attach(model)
        response, elapsed = generate_chat(model, tokenizer, msgs)
        hook.detach()
        print(f"\n{schema['character']}: {response}")
        print(f"[{elapsed:.1f}s]")
        logger.log(response)

    else:
        parser.print_help()

    log_path = logger.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
