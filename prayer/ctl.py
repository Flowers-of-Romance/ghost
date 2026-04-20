"""prayer daemon 制御クライアント。

使用:
    python prayer/ctl.py status
    python prayer/ctl.py observe
    python prayer/ctl.py set-mode active
    python prayer/ctl.py shutdown
"""

import argparse
import json
import socket
import sys

SOCKET_PATH = "/tmp/prayer.sock"


def call(cmd: dict, timeout: float = 12.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        return {"ok": False, "error": f"daemon未起動: {e}"}
    try:
        s.sendall((json.dumps(cmd) + "\n").encode())
        data = b""
        while b"\n" not in data:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        line = data.split(b"\n", 1)[0]
        if not line:
            return {"ok": False, "error": "空レスポンス"}
        return json.loads(line)
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="subcmd", required=True)
    sub.add_parser("status")
    sub.add_parser("observe")
    m = sub.add_parser("set-mode")
    m.add_argument("mode", choices=["auto", "idle", "active"])
    sp = sub.add_parser("speak")
    sp.add_argument("text")
    sp.add_argument("--voice", default=None, help="macOS say voice (fallback)")
    sp.add_argument("--speaker", type=int, default=None, help="VOICEVOX speaker id")
    sub.add_parser("shutdown")
    args = p.parse_args()

    if args.subcmd == "status":
        resp = call({"cmd": "status"})
    elif args.subcmd == "observe":
        resp = call({"cmd": "observe"})
    elif args.subcmd == "set-mode":
        resp = call({"cmd": "set_mode", "mode": args.mode})
    elif args.subcmd == "speak":
        cmd = {"cmd": "speak", "text": args.text}
        if args.voice:
            cmd["voice"] = args.voice
        if args.speaker is not None:
            cmd["speaker"] = args.speaker
        resp = call(cmd, timeout=60.0)
    elif args.subcmd == "shutdown":
        resp = call({"cmd": "shutdown"})
    else:
        p.error(f"unknown: {args.subcmd}")

    print(json.dumps(resp, ensure_ascii=False, indent=2))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
