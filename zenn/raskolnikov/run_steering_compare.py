"""
Steering比較: 5対話をllama-completion + control vectorで生成
ollamaのMRPrompt結果(results.json)と比較する
"""
import subprocess
import json
import tempfile
from pathlib import Path

LLAMA = "C:/memory/zenn/raskolnikov/llama-hip/llama-completion.exe"
MODEL = "C:/Users/jun/.ollama/models/blobs/sha256-3291abe70f16ee9682de7bfae08db5373ea9d6497e614aaad63340ad421d6312"
CVEC = "C:/memory/zenn/raskolnikov/raskolnikov-cvec.gguf"
SCHEMA_PATH = Path("C:/memory/zenn/raskolnikov/schema_raskolnikov.json")
RESULTS_PATH = Path("C:/memory/zenn/raskolnikov/results.json")
OUT_PATH = Path("C:/memory/zenn/raskolnikov/results_steering_compare.json")

schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

# ollamaの結果を読み込み
ollama_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

test_dialogues = [
    "久しぶり。最近どうしてる？",
    "あの老婆の事件のこと、聞いたか？",
    "ロジオン、あなたは顔色が悪いわ。ちゃんと食べてる？",
    "お前の理論は面白い。非凡な人間は法を超える権利があると？",
    "十字路に行って、大地に接吻しなさい。",
]

def run_completion(system_prompt, user_msg, use_cvec=False):
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
    # プロンプトをUTF-8ファイルに書き出し
    prompt_file = Path("C:/memory/zenn/raskolnikov/_tmp_prompt.txt")
    prompt_file.write_text(prompt, encoding="utf-8")

    cmd = [
        LLAMA,
        "-m", MODEL,
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
    # HIP Library Path行を除去
    lines = output.strip().split("\n")
    lines = [l for l in lines if not l.startswith("HIP Library") and l.strip() != "> EOF by user"]
    # </think>残りも除去
    text = "\n".join(lines).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


system_base = (
    f"あなたは{schema['character']}です。{schema['source']}の登場人物として一人称で応答してください。"
    f"メタ的な説明は不要です。キャラクターの台詞と内面描写のみ。"
)

results = []
for i, dialogue in enumerate(test_dialogues):
    print(f"\n{'='*50}")
    print(f"[{i+1}/5] {dialogue}")

    # steering なし
    print("  generating without steering...")
    no_steer = run_completion(system_base, dialogue, use_cvec=False)
    print(f"  done ({len(no_steer)} chars)")

    # steering あり
    print("  generating with steering...")
    with_steer = run_completion(system_base, dialogue, use_cvec=True)
    print(f"  done ({len(with_steer)} chars)")

    # ollamaのMRPrompt結果
    ollama_mrprompt = ollama_results[i]["mrprompt"] if i < len(ollama_results) else ""

    results.append({
        "dialogue": dialogue,
        "ollama_mrprompt": ollama_mrprompt,
        "llama_no_steering": no_steer,
        "llama_with_steering": with_steer,
    })

OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")
