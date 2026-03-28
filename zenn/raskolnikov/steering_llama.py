"""
Control Vector Steering via llama-cpp-python
- WSL + Qwen3.5-35B-A3B (GGUF, ollamaのblob)
- 対照ペアから control vector を抽出
- MRPrompt + steering の比較

WSLから実行:
  source ~/llama-env/bin/activate
  cd /mnt/c/memory/zenn/raskolnikov
  python steering_llama.py --extract
  python steering_llama.py --compare
  python steering_llama.py --dialogue "久しぶり。最近どうしてる？"
"""

import json
import argparse
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from llama_cpp import Llama

MODEL_PATH = "/mnt/c/Users/jun/.ollama/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
SCHEMA_PATH = Path(__file__).parent / "schema_raskolnikov.json"
VECTORS_PATH = Path(__file__).parent / "control_vectors.pkl"
LOG_DIR = Path(__file__).parent / "logs"
TOP_K_FACETS = 3
TARGET_LAYER = 18
STEERING_ALPHA = 1.5  # 3.0は強すぎた、控えめに


class Logger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"cvec_{ts}.log"
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


def load_model(n_ctx=4096, embedding=False):
    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=n_ctx,
        n_gpu_layers=0,
        verbose=False,
        embedding=embedding,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return llm


# =========================================================
# 対照ペア
# =========================================================

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


# =========================================================
# Control Vector 抽出 (llama-cpp-python の embeddings)
# =========================================================

def get_embeddings(llm, text: str) -> np.ndarray:
    """テキストの最終トークンの embedding を取得"""
    embeddings = llm.embed(text)
    emb = np.array(embeddings)
    # (n_tokens, n_embd) → 最終トークンだけ使う
    if emb.ndim == 2:
        return emb[-1]
    return emb


def extract_control_vectors(llm, logger=None):
    """対照ペアからcontrol vectorを計算"""
    vectors = {}
    for name, pairs in CONTRASTIVE_PAIRS.items():
        print(f"  Extracting: {name} ({len(pairs)} pairs)")
        if logger:
            logger.log(f"Extracting: {name} ({len(pairs)} pairs)")

        diffs = []
        for pos_text, neg_text in pairs:
            t0 = time.time()
            pos_emb = get_embeddings(llm, pos_text)
            neg_emb = get_embeddings(llm, neg_text)
            diff = pos_emb - neg_emb
            diffs.append(diff)
            elapsed = time.time() - t0
            norm = np.linalg.norm(diff)
            if logger:
                logger.log(f"  pair done in {elapsed:.1f}s, norm={norm:.2f}")
            print(f"    pair done in {elapsed:.1f}s, norm={norm:.2f}")

        mean_diff = np.mean(diffs, axis=0)
        norm = np.linalg.norm(mean_diff)
        if norm > 0:
            mean_diff = mean_diff / norm
        vectors[name] = mean_diff

        if logger:
            logger.log(f"  {name} vector computed, shape={mean_diff.shape}")

    return vectors


# =========================================================
# 生成 (control vector の直接適用は llama.cpp CLI が必要)
# =========================================================

def generate_chat(llm, messages, max_tokens=300):
    """llama-cpp-python の chat completion"""
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    elapsed = time.time() - t0
    content = response["choices"][0]["message"]["content"]
    return content, elapsed


# =========================================================
# MRPrompt構築
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
    """ollamaでファセット選択（Windows側のollamaにアクセス）"""
    import urllib.request
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

    # WSLからWindows側のollamaにアクセス
    payload = {
        "model": "qwen3.5:35b-a3b",
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": 30, "temperature": 0.7, "top_p": 0.9},
    }

    # WSLからlocalhostはホストマシンを指す
    url = "http://localhost:11434/api/chat"
    req = urllib.request.Request(
        url,
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
    parser = argparse.ArgumentParser(description="Control Vector Steering")
    parser.add_argument("--extract", action="store_true", help="control vector を抽出")
    parser.add_argument("--compare", action="store_true", help="MRPrompt vs MRPrompt+steering")
    parser.add_argument("--dialogue", type=str, help="対話文脈")
    parser.add_argument("--alpha", type=float, default=STEERING_ALPHA)
    args = parser.parse_args()

    logger = Logger(LOG_DIR)
    logger.log(f"Started: {datetime.now().isoformat()}")
    logger.log(f"Alpha: {args.alpha}")

    schema = load_schema(SCHEMA_PATH)

    if args.extract:
        llm = load_model(n_ctx=512, embedding=True)
        print("Extracting control vectors...")
        logger.section("Extracting control vectors")
        vectors = extract_control_vectors(llm, logger)
        with open(VECTORS_PATH, "wb") as f:
            pickle.dump(vectors, f)
        print(f"Saved to {VECTORS_PATH}")
        for name, vec in vectors.items():
            print(f"  {name}: shape={vec.shape}")
        del llm

    elif args.compare or args.dialogue:
        # Note: llama-cpp-python doesn't support runtime control vectors
        # directly via Python API. For now, generate without steering
        # and note that full steering requires llama.cpp CLI with
        # --control-vector flag.
        #
        # This comparison uses the 35B-A3B model for generation quality.
        llm = load_model(n_ctx=4096)

        if args.compare:
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

                selected = select_facets_ollama(schema, dialogue)
                facet_names = [f['id'] + ': ' + f['trigger'] for f in selected]
                logger.log(f"Facets: {facet_names}")
                print(f"Facets: {facet_names}")

                msgs = build_mrprompt_messages(schema, selected, dialogue)

                logger.subsection("MRPrompt (35B-A3B via llama.cpp)")
                response, elapsed = generate_chat(llm, msgs)
                logger.log(f"[{elapsed:.1f}s]")
                logger.log(response)
                print(f"[{elapsed:.1f}s]")
                print(response[:200])

                results.append({
                    "dialogue": dialogue,
                    "selected_facets": [f["id"] for f in selected],
                    "mrprompt_llama": response,
                })

            out_path = Path(__file__).parent / "results_llama.json"
            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nResults saved to {out_path}")

        elif args.dialogue:
            selected = select_facets_ollama(schema, args.dialogue)
            print(f"Facets: {[f['trigger'] for f in selected]}")
            msgs = build_mrprompt_messages(schema, selected, args.dialogue)
            response, elapsed = generate_chat(llm, msgs)
            print(f"\n{schema['character']}: {response}")
            print(f"[{elapsed:.1f}s]")
            logger.log(response)

    else:
        parser.print_help()

    log_path = logger.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
