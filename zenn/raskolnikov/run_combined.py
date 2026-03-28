"""
MRPrompt + Steering 組み合わせ比較
- facet選択: ollama (qwen3:32b)
- 最終生成: llama-completion.exe + control vector (GPU)

4条件:
1. ollama MRPrompt (ベースライン)
2. llama MRPrompt (steering なし)
3. llama MRPrompt (steering あり)
4. llama 簡易プロンプト (steering あり) ← 前回の結果と同等
"""
import subprocess
import json
import urllib.request
import time
from pathlib import Path

LLAMA = "C:/memory/zenn/raskolnikov/llama-hip/llama-completion.exe"
MODEL_GGUF = "C:/Users/jun/.ollama/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
CVEC = "C:/memory/zenn/raskolnikov/raskolnikov-cvec.gguf"
SCHEMA_PATH = Path("C:/memory/zenn/raskolnikov/schema_raskolnikov.json")
OUT_PATH = Path("C:/memory/zenn/raskolnikov/results_combined.json")

OLLAMA_MODEL = "qwen3:32b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TOP_K_FACETS = 3

schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def ollama_chat(messages, max_tokens=300):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7, "top_p": 0.9},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["message"]["content"]


def select_facets(dialogue):
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
    response = ollama_chat(messages, max_tokens=30)
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


def build_mrprompt_system(selected_facets):
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

    # boundary
    anchors = schema.get("boundary_anchors", {})
    boundary_lines = []
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
    return system


def build_simple_system():
    return (
        f"あなたは{schema['character']}です。{schema['source']}の登場人物として一人称で応答してください。"
        f"メタ的な説明は不要です。キャラクターの台詞と内面描写のみ。"
    )


def run_llama(system_prompt, user_msg, use_cvec=False):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    prompt_file = Path("C:/memory/zenn/raskolnikov/_tmp_prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        LLAMA,
        "-m", MODEL_GGUF,
        "-f", str(prompt_file),
        "-c", "2048",
        "-n", "300",
        "--temp", "0.7",
        "--top-p", "0.9",
        "-ngl", "99",
        "--no-display-prompt",
        "--logit-bias", "151667-inf",
    ]
    if use_cvec:
        cmd.extend(["--control-vector", CVEC])

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
    output = result.stdout
    lines = output.strip().split("\n")
    lines = [l for l in lines if not l.startswith("HIP Library") and l.strip() != "> EOF by user"]
    text = "\n".join(lines).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def run_ollama_mrprompt(system_prompt, dialogue):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dialogue},
    ]
    return ollama_chat(messages)


test_dialogues = [
    "久しぶり。最近どうしてる？",
    "あの老婆の事件のこと、聞いたか？",
    "ロジオン、あなたは顔色が悪いわ。ちゃんと食べてる？",
    "お前の理論は面白い。非凡な人間は法を超える権利があると？",
    "十字路に行って、大地に接吻しなさい。",
]

results = []
for i, dialogue in enumerate(test_dialogues):
    print(f"\n{'='*60}")
    print(f"[{i+1}/5] {dialogue}")
    print(f"{'='*60}")

    # facet選択 (共通)
    print("  selecting facets...")
    selected = select_facets(dialogue)
    facet_ids = [f["id"] for f in selected]
    print(f"  facets: {facet_ids}")

    mrprompt_sys = build_mrprompt_system(selected)
    simple_sys = build_simple_system()

    # 条件1: ollama MRPrompt
    print("  [1/4] ollama MRPrompt...")
    t0 = time.time()
    c1 = run_ollama_mrprompt(mrprompt_sys, dialogue)
    print(f"    done ({time.time()-t0:.1f}s, {len(c1)} chars)")

    # 条件2: llama MRPrompt (no steering)
    print("  [2/4] llama MRPrompt (no steering)...")
    t0 = time.time()
    c2 = run_llama(mrprompt_sys, dialogue, use_cvec=False)
    print(f"    done ({time.time()-t0:.1f}s, {len(c2)} chars)")

    # 条件3: llama MRPrompt + steering
    print("  [3/4] llama MRPrompt + steering...")
    t0 = time.time()
    c3 = run_llama(mrprompt_sys, dialogue, use_cvec=True)
    print(f"    done ({time.time()-t0:.1f}s, {len(c3)} chars)")

    # 条件4: llama simple + steering (前回と同等)
    print("  [4/4] llama simple + steering...")
    t0 = time.time()
    c4 = run_llama(simple_sys, dialogue, use_cvec=True)
    print(f"    done ({time.time()-t0:.1f}s, {len(c4)} chars)")

    results.append({
        "dialogue": dialogue,
        "selected_facets": facet_ids,
        "ollama_mrprompt": c1,
        "llama_mrprompt_no_steering": c2,
        "llama_mrprompt_with_steering": c3,
        "llama_simple_with_steering": c4,
    })

OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")
