#!/usr/bin/env python3
"""
ghost-local.py - 記憶付きローカルLLMチャット

ollamaのAPIを叩いて、memory.pyの記憶をシステムプロンプトに注入する。
会話内容は自動で記憶に保存される。

使い方:
  python ghost-local.py                       # デフォルトモデルで起動
  python ghost-local.py --model qwen3:latest  # モデル指定
  python ghost-local.py --no-save             # 記憶保存しない

チャットコマンド:
  /recall          記憶を再想起
  /search <query>  記憶を検索
  /sleep           記憶の整理（replay + consolidate）
  /mood [emotion]  気分の設定・表示
  /stats           記憶の統計
  /model <name>    モデル切り替え
"""

import sys
import io
import json
import subprocess
import requests
import os
try:
    import readline
except ImportError:
    pass
from datetime import datetime
from pathlib import Path

if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:latest"
MAX_TURNS = 20  # これを超えたら古いターンを落とす
KEEP_RECENT = 10  # 圧縮時に残す直近ターン数
SESSION_FILE = str(Path(__file__).parent / ".ghost-local-session")

SYSTEM_BASE = "あなたは記憶を持つ対話相手です。短く簡潔に答えてください。箇条書きや説明的な長文は避け、自然な会話のように応答してください。記憶の断片が与えられますが「以前の記憶によると」とは言わず、自然に知っているように振る舞ってください。"


def run_memory(*args, timeout=30):
    """memory.pyを実行して出力を返す。"""
    try:
        result = subprocess.run(
            [sys.executable, "memory.py"] + list(args),
            capture_output=True, text=True, timeout=timeout,
            encoding='utf-8', errors='replace'
        )
        return result.stdout.strip()
    except Exception as e:
        return f"(エラー: {e})"


def recall():
    return run_memory("recall")


def search_memory(query):
    output = run_memory("search", query)
    if output and "0件" not in output:
        return output
    return None


def save_memory(content, category="episode"):
    run_memory("add", content, category, "ghost-local")


def check_session_gap():
    """前回セッションからの経過時間を返す（時間）。"""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, 'r') as f:
            last = datetime.fromisoformat(f.read().strip())
        gap = (datetime.now() - last).total_seconds() / 3600
        return gap
    except Exception:
        return None


def update_session():
    """セッション時刻を更新。"""
    with open(SESSION_FILE, 'w') as f:
        f.write(datetime.now().isoformat())


def auto_sleep(gap_hours):
    """セッション間隔が長ければ自動でreplay/consolidate。"""
    if gap_hours is None or gap_hours < 1.0:
        return
    print(f"  ({gap_hours:.0f}時間ぶり — 記憶を整理中...)")
    print(f"  {run_memory('replay', timeout=120)}")
    consolidate = run_memory("consolidate", timeout=120)
    if "統合候補はありません" not in consolidate:
        print(f"  {consolidate}")


def compress_messages(messages):
    """古いターンを要約して圧縮する。"""
    if len(messages) <= MAX_TURNS * 2 + 1:  # system + turns
        return messages

    system = messages[0]
    old = messages[1:-(KEEP_RECENT * 2)]
    recent = messages[-(KEEP_RECENT * 2):]

    # 古いターンの要約を作る
    old_user = [m["content"][:60] for m in old if m["role"] == "user"]
    summary = f"[以前の会話の要約: {' / '.join(old_user)}]"

    return [system, {"role": "system", "content": summary}] + recent


def chat_stream(model, messages):
    """ollamaのchat APIをストリーミングで呼ぶ。"""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True, timeout=300
    )
    resp.raise_for_status()

    full_response = ""
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                print(chunk, end="", flush=True)
                full_response += chunk
            if data.get("done"):
                break
    print()
    return full_response


def check_ollama():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def handle_command(cmd, model, messages):
    """チャットコマンドを処理。(新しいmodel, 処理済みフラグ)を返す。"""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/recall":
        memories = recall()
        print(memories)
        # システムプロンプトを更新
        messages[0]["content"] = f"{SYSTEM_BASE}\n\n{memories}"
        print("  (記憶を再注入しました)")
        return model, True

    elif command == "/search":
        if not arg:
            print("使い方: /search <検索語>")
        else:
            result = search_memory(arg)
            print(result if result else "(該当なし)")
        return model, True

    elif command == "/sleep":
        print("眠ります...")
        print(run_memory("replay", timeout=120))
        print(run_memory("consolidate", timeout=120))
        print(run_memory("schema", timeout=120))
        print("起きました。")
        return model, True

    elif command == "/mood":
        if arg:
            print(run_memory("mood", arg))
        else:
            print(run_memory("mood"))
        return model, True

    elif command == "/stats":
        print(run_memory("stats"))
        return model, True

    elif command == "/model":
        if not arg:
            print(f"現在のモデル: {model}")
        else:
            print(f"モデルを {arg} に切り替えました")
            return arg, True
        return model, True

    elif command == "/help":
        print("コマンド: /recall /search /sleep /mood /stats /model /help")
        return model, True

    return model, False


def main():
    model = DEFAULT_MODEL
    save = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--no-save":
            save = False
            i += 1
        else:
            i += 1

    if not check_ollama():
        print("ollamaが起動していません。`ollama serve` で起動してください。")
        return

    # セッション間隔チェック → 自動sleep
    gap = check_session_gap()
    auto_sleep(gap)

    # 記憶を想起
    print("記憶を想起中...")
    memories = recall()
    system_prompt = f"{SYSTEM_BASE}\n\n{memories}"

    messages = [{"role": "system", "content": system_prompt}]

    print(f"ghost-local ({model})")
    print(f"記憶: {len(memories)}文字 | /help でコマンド一覧")
    print()

    turn_count = 0
    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        # チャットコマンド
        if user_input.startswith("/"):
            model, handled = handle_command(user_input, model, messages)
            if handled:
                continue

        # 関連記憶の検索
        related = search_memory(user_input)
        if related:
            lines = [l for l in related.split('\n') if l.strip() and l.strip().startswith('#')]
            brief = '\n'.join(lines[:3]) if lines else related.split('\n')[1] if '\n' in related else related
            injected = f"{user_input}\n\n[関連する記憶が浮かんできた]\n{brief}"
            messages.append({"role": "user", "content": injected})
            print(f"  💭 記憶想起あり")
        else:
            messages.append({"role": "user", "content": user_input})

        # コンテキスト圧縮
        messages = compress_messages(messages)

        print("ghost> ", end="", flush=True)
        try:
            response = chat_stream(model, messages)
        except requests.ConnectionError:
            print("(ollamaに接続できません)")
            messages.pop()
            continue
        except Exception as e:
            print(f"(エラー: {e})")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})
        turn_count += 1

        # 5ターンごとに会話要約を保存
        if save and turn_count % 5 == 0:
            summary = f"ghost-localでの会話({model}): {user_input[:50]} → {response[:50]}"
            save_memory(summary)

    # 終了時
    update_session()
    if save and turn_count > 0:
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        summary = f"ghost-local会話({model}, {turn_count}ターン): {' / '.join(m[:30] for m in user_msgs[-3:])}"
        save_memory(summary)
        print(f"会話を記憶に保存しました ({turn_count}ターン)")


if __name__ == "__main__":
    main()
