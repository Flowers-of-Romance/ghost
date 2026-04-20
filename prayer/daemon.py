"""prayer_daemon — ghost の視聴覚末梢神経。

Batesonian 原則 (三次サイバネティックス):
- 差異重視: 前フレームからの変化で駆動
- double description: cortex (構造的) + limbic (情調的) 両パスで記述
- no central state: perceptions テーブルの流れが状態そのもの
- unit of survival: circuit (J↔daemon↔Claude) として機能

TODO(後続):
- audio capture + VAD
- Qwen3-Omni 実ロード (現在 MockPerceiver)
- sleep consolidation 連携
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memory import add_perception, get_last_perception  # noqa: E402


def _lazy_cv():
    import cv2  # noqa
    import numpy as np  # noqa
    return cv2, np


FRAMES_DIR = Path(__file__).resolve().parent / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
SOCKET_PATH = "/tmp/prayer.sock"


class MockPerceiver:
    """Qwen3-Omni ロード前の仮実装。画像を見ずにラベルを返す。"""

    def describe(self, frame, audio=None, context=None):
        h, w = frame.shape[:2]
        brightness = float(frame.mean())
        cortex = f"mock: frame {w}x{h}, brightness={brightness:.1f}"
        limbic = "mock: 情調不明"
        return cortex, limbic


class QwenOmniPerceiver:
    """本実装（モデルダウンロード完了後に有効化）。"""

    WHISPER_MODEL = "mlx-community/whisper-small-mlx"

    # Whisper が静寂/ノイズに対して返す定番ハルシネーション（YouTube キャプション由来）。
    # 完全一致でブロック。
    WHISPER_BOILERPLATE = {
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました。",
        "ありがとうございました",
        "ありがとうございました。",
        "チャンネル登録お願いします",
        "チャンネル登録お願いします。",
        "Thank you.",
        "Thanks for watching!",
        "you",
        "You",
    }

    def __init__(self, model_path):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        self.model_path = model_path
        # whisper をプリロード＆ウォームアップ — 後で audio describe 中に Qwen inference と
        # 同時にロードが走ると MLX 側で bus error/segfault になる。
        # 無音の短い配列で1回 transcribe しておいて MLX allocator を安定化させる。
        from mlx_whisper.transcribe import ModelHolder
        import mlx.core as mx
        import numpy as np
        print(f"[daemon] preloading whisper: {self.WHISPER_MODEL}")
        ModelHolder.get_model(self.WHISPER_MODEL, dtype=mx.float16)
        import mlx_whisper
        _ = mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),  # 1秒の無音
            path_or_hf_repo=self.WHISPER_MODEL,
            language="ja",
        )
        print(f"[daemon] whisper preloaded & warmed")

    def _describe_one(self, frame, prompt, max_tokens):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        out = generate(self.model, self.processor, formatted, [frame], max_tokens=max_tokens)
        # mlx_vlm.generate returns a GenerationResult or str; normalize
        text = getattr(out, 'text', out) if not isinstance(out, str) else out
        return str(text).strip()

    def describe(self, frame, audio=None, context=None):
        cortex_prompt = (
            "この画像を構造的に記述して。何が見える、配置、色、物理的事実のみ。"
            "解釈や情調は含めない。"
        )
        limbic_prompt = (
            "この画像から感じる情調を短く記述して。"
            "雰囲気、緊張、穏やかさ、疲労など、felt sense のみ。"
        )
        cortex = self._describe_one(frame, cortex_prompt, max_tokens=120)
        limbic = self._describe_one(frame, limbic_prompt, max_tokens=60)
        return cortex, limbic

    def describe_audio(self, audio_path):
        """音声 describe は mlx_vlm の qwen3-omni-moe 音声パスに既知バグがあるため
        ASR は mlx-whisper を使う。limbic は v0 では空（TODO: prosody ベースの簡易推定）。
        wav を自前で np.ndarray にロードして ffmpeg 依存を回避。
        """
        import mlx_whisper
        import numpy as np
        import wave
        with wave.open(audio_path, 'rb') as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != 16000:
            # whisper は 16kHz 想定。VADで16kHz保存してるので基本ここは通らない
            import scipy.signal
            pcm = scipy.signal.resample_poly(pcm, 16000, sr).astype(np.float32)
        result = mlx_whisper.transcribe(
            pcm,
            path_or_hf_repo=self.WHISPER_MODEL,
            language="ja",
            no_speech_threshold=0.8,
            condition_on_previous_text=False,
        )
        cortex = (result.get("text") or "").strip()
        if cortex in self.WHISPER_BOILERPLATE:
            cortex = ""
        limbic = None
        return cortex, limbic  # cortex が空なら呼び出し側で perception 作成skip


def frame_hash(frame, size=32):
    """軽量な知覚ハッシュ — 大きな変化の検知用。"""
    cv2, _ = _lazy_cv()
    small = cv2.resize(frame, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def frame_diff_score(f1, f2, size=64):
    """フレーム間の平均差（0-1）。"""
    cv2, np = _lazy_cv()
    s1 = cv2.resize(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float32)
    s2 = cv2.resize(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), (size, size)).astype(np.float32)
    return float(np.mean(np.abs(s1 - s2)) / 255.0)


def save_frame(frame, perception_uuid):
    """perception 用に frame を保存。"""
    cv2, _ = _lazy_cv()
    path = FRAMES_DIR / f"{perception_uuid}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


class DaemonState:
    """socketスレッドと mainループが共有する状態。lockで保護。"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.lock = threading.Lock()
        self.shutdown = False
        self.mode = 'auto'  # 'auto' | 'idle' | 'active'
        self.observe_request = None  # dict or None — 1回分の強制観察要求
        self.observe_result = {}  # request_id -> perception_id
        self.observe_cv = threading.Condition(self.lock)
        self.last_diff = 0.0
        self.last_perception_id = None
        self.frames_seen = 0
        self.latest_frame_path = None  # audio perception が紐づけるための直近 video frame
        # MLX推論の直列化 — video describe と audio ASR が並行で走ると SIGSEGV
        self.perceiver_lock = threading.Lock()


def capture_once(perceiver, cap, session_id, invoked_by='invoke', forced_arousal=None,
                 lock=None):
    """1フレーム取って記述、perception保存。成功時 perception_id を返す。"""
    ok, frame = cap.read()
    if not ok:
        return None, None
    if lock is not None:
        with lock:
            cortex_text, limbic_text = perceiver.describe(frame)
    else:
        cortex_text, limbic_text = perceiver.describe(frame)
    arousal = forced_arousal if forced_arousal is not None else 0.3
    import uuid as _uuid
    frame_uuid = str(_uuid.uuid4())
    frame_path = save_frame(frame, frame_uuid)
    pid = add_perception(
        modality='vision',
        content_cortex=cortex_text,
        content_limbic=limbic_text,
        session_id=session_id,
        arousal=arousal,
        raw_frame_path=frame_path,
        invoked_by=invoked_by,
        importance=2,  # invoke は auto より重い
    )
    if not pid:
        os.remove(frame_path)
    return pid, frame


TTS_VOICE = "Kyoko"  # macOS say fallback
VOICEVOX_HOST = "http://127.0.0.1:50021"
VOICEVOX_DEFAULT_SPEAKER = 3  # ずんだもん ノーマル


def _speak_voicevox(text, speaker=VOICEVOX_DEFAULT_SPEAKER, timeout=10.0):
    """VOICEVOX engine が動いていれば使う。wav を一時ファイルに落として afplay。"""
    import json
    import subprocess
    import tempfile
    import urllib.parse
    import urllib.request
    try:
        # 1. audio_query
        q_url = f"{VOICEVOX_HOST}/audio_query?text={urllib.parse.quote(text)}&speaker={speaker}"
        req = urllib.request.Request(q_url, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            query = r.read()
        # 2. synthesis
        s_url = f"{VOICEVOX_HOST}/synthesis?speaker={speaker}"
        req = urllib.request.Request(
            s_url, data=query, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout * 3) as r:
            wav = r.read()
    except Exception as e:
        return False, f"voicevox: {e}"
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            wav_path = f.name
        subprocess.run(["afplay", wav_path], check=True)
        import os as _os
        _os.unlink(wav_path)
        return True, None
    except Exception as e:
        return False, f"afplay: {e}"


def _voicevox_alive():
    import urllib.request
    try:
        with urllib.request.urlopen(f"{VOICEVOX_HOST}/version", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def _speak(text, voice=TTS_VOICE, speaker=None):
    """発話。VOICEVOX engine が起動していれば VOICEVOX、ダメなら macOS say にフォールバック。"""
    import subprocess
    if not text:
        return False, "empty text"
    if _voicevox_alive():
        sp = speaker if speaker is not None else VOICEVOX_DEFAULT_SPEAKER
        ok, err = _speak_voicevox(text, speaker=sp)
        if ok:
            return True, None
        print(f"[tts] voicevox failed ({err}), falling back to say")
    try:
        subprocess.run(["say", "-v", voice, text], check=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"say exit {e.returncode}"
    except FileNotFoundError:
        return False, "say not found (macOS only)"


def handle_command(state, cmd):
    """JSONコマンドをdispatch、JSON応答辞書を返す。"""
    name = cmd.get("cmd")
    if name == "status":
        with state.lock:
            return {
                "ok": True,
                "session_id": state.session_id,
                "mode": state.mode,
                "last_diff": state.last_diff,
                "last_perception_id": state.last_perception_id,
                "frames_seen": state.frames_seen,
            }
    if name == "set_mode":
        mode = cmd.get("mode", "auto")
        if mode not in ("auto", "idle", "active"):
            return {"ok": False, "error": f"invalid mode: {mode}"}
        with state.lock:
            state.mode = mode
        return {"ok": True, "mode": mode}
    if name == "observe":
        req_id = f"req-{time.time_ns()}"
        with state.observe_cv:
            state.observe_request = {"id": req_id}
            state.observe_cv.notify_all()
            # メインループが拾うまで短時間待つ
            state.observe_cv.wait_for(lambda: req_id in state.observe_result, timeout=10.0)
            pid = state.observe_result.pop(req_id, None)
        if pid is None:
            return {"ok": False, "error": "observe timeout or failed"}
        return {"ok": True, "perception_id": pid}
    if name == "speak":
        text = cmd.get("text", "")
        voice = cmd.get("voice", TTS_VOICE)
        speaker = cmd.get("speaker")  # VOICEVOX speaker id（指定なければ default）
        ok, err = _speak(text, voice=voice, speaker=speaker)
        return {"ok": ok, **({"error": err} if err else {"spoke": text})}
    if name == "shutdown":
        with state.lock:
            state.shutdown = True
        return {"ok": True, "shutting_down": True}
    return {"ok": False, "error": f"unknown cmd: {name}"}


def socket_server(state):
    """Unix socket サーバ — 1接続1コマンドのJSONL。"""
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    srv.listen(4)
    srv.settimeout(0.5)
    print(f"[daemon] socket listening on {SOCKET_PATH}")
    while not state.shutdown:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        try:
            conn.settimeout(2.0)
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > 65536:
                    break
            line = data.split(b"\n", 1)[0]
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {"ok": False, "error": f"json parse: {e}"}
            else:
                resp = handle_command(state, cmd)
            conn.sendall((json.dumps(resp) + "\n").encode())
        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
            except Exception:
                pass
        finally:
            conn.close()
    srv.close()
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    print("[daemon] socket closed")


AUDIO_DIR = Path(__file__).resolve().parent / "audio"


def _on_speech_segment_factory(state, perceiver, session_id):
    """audio segment 完了時のコールバックを返す。"""
    def on_segment(wav_path, duration_sec):
        try:
            with state.perceiver_lock:
                cortex_text, limbic_text = perceiver.describe_audio(wav_path)
        except Exception as e:
            print(f"[audio] describe failed: {e}")
            return
        if not cortex_text:
            # no_speech判定・書き起こし空 → perception作らず wav も捨てる
            try:
                os.remove(wav_path)
            except OSError:
                pass
            return
        with state.lock:
            linked_frame = state.latest_frame_path
        arousal = min(0.8, 0.3 + duration_sec / 10.0)
        pid = add_perception(
            modality='audio',
            content_cortex=cortex_text,
            content_limbic=limbic_text,
            session_id=session_id,
            arousal=arousal,
            raw_audio_path=wav_path,
            raw_frame_path=linked_frame,
            invoked_by='auto',
            importance=2,
        )
        if pid:
            with state.lock:
                state.last_perception_id = pid
            print(f"[audio] #{pid} dur={duration_sec:.1f}s → {cortex_text[:60]}")
    return on_segment


def run_loop(session_id, perceiver, interval_active=1.0, interval_idle=15.0,
             diff_threshold=0.05, camera_index=0, enable_socket=True,
             enable_audio=False, audio_device_id=None):
    """メインループ — 適応的サンプリング込み。

    変化大: interval_active (1秒)
    変化小: interval_idle (15秒)
    mode=active: 常に active interval
    mode=idle: 常に idle interval、差分があっても緩める
    """
    cv2, _ = _lazy_cv()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {camera_index} を開けない")

    state = DaemonState(session_id)
    srv_thread = None
    if enable_socket:
        srv_thread = threading.Thread(target=socket_server, args=(state,), daemon=True)
        srv_thread.start()

    audio_stop_event = threading.Event()
    audio_thread = None
    if enable_audio and hasattr(perceiver, 'describe_audio'):
        from audio_capture import VadCapturer
        resolved_device_id = None
        if audio_device_id is not None:
            import miniaudio
            caps = miniaudio.Devices().get_captures()
            if 0 <= audio_device_id < len(caps):
                resolved_device_id = caps[audio_device_id]['id']
                print(f"[daemon] audio device: {caps[audio_device_id]['name']}")
            else:
                print(f"[daemon] audio device index {audio_device_id} 範囲外 — default を使う")
        on_segment = _on_speech_segment_factory(state, perceiver, session_id)
        capturer = VadCapturer(
            out_dir=AUDIO_DIR,
            on_segment=on_segment,
            device_id=resolved_device_id,
        )
        audio_thread = threading.Thread(
            target=capturer.run, args=(audio_stop_event,), daemon=True,
        )
        audio_thread.start()
        print(f"[daemon] audio capture thread started")
    elif enable_audio:
        print("[daemon] audio 要求されたが perceiver が describe_audio 未対応 — skip")

    prev_frame = None
    interval = interval_idle
    print(f"[daemon] session={session_id} perceiver={type(perceiver).__name__}")

    try:
        while not state.shutdown:
            # 1. 強制観察要求を先に処理
            with state.observe_cv:
                req = state.observe_request
                state.observe_request = None
            if req is not None:
                pid, frame = capture_once(perceiver, cap, session_id, invoked_by='invoke',
                                          forced_arousal=0.5, lock=state.perceiver_lock)
                with state.observe_cv:
                    state.observe_result[req["id"]] = pid
                    if pid is not None:
                        state.last_perception_id = pid
                    state.observe_cv.notify_all()
                if frame is not None:
                    prev_frame = frame
                continue

            ok, frame = cap.read()
            if not ok:
                print("[daemon] frame読み取り失敗, リトライ")
                time.sleep(1.0)
                continue

            diff = frame_diff_score(prev_frame, frame) if prev_frame is not None else 1.0
            with state.lock:
                state.last_diff = diff
                state.frames_seen += 1
                mode = state.mode

            if mode == 'active':
                is_significant = True
            elif mode == 'idle':
                is_significant = diff > diff_threshold * 3.0
            else:
                is_significant = diff > diff_threshold
            interval = interval_active if is_significant else interval_idle

            if is_significant or prev_frame is None:
                with state.perceiver_lock:
                    cortex_text, limbic_text = perceiver.describe(frame)
                arousal = min(0.8, 0.2 + diff * 2.0)
                import uuid as _uuid
                frame_uuid = str(_uuid.uuid4())
                frame_path = save_frame(frame, frame_uuid)

                pid = add_perception(
                    modality='vision',
                    content_cortex=cortex_text,
                    content_limbic=limbic_text,
                    session_id=session_id,
                    arousal=arousal,
                    raw_frame_path=frame_path,
                    invoked_by='auto',
                    importance=1 + int(diff > 0.15),
                )
                if pid:
                    with state.lock:
                        state.last_perception_id = pid
                        state.latest_frame_path = frame_path
                    print(f"[daemon] #{pid} diff={diff:.3f} → {cortex_text[:60]}")
                else:
                    os.remove(frame_path)

            prev_frame = frame
            # shutdown を早く拾うために細切れで待つ
            slept = 0.0
            while slept < interval and not state.shutdown and state.observe_request is None:
                time.sleep(min(0.2, interval - slept))
                slept += 0.2
    except KeyboardInterrupt:
        print("\n[daemon] 停止")
    finally:
        with state.lock:
            state.shutdown = True
        audio_stop_event.set()
        cap.release()
        if srv_thread is not None:
            srv_thread.join(timeout=2.0)
        if audio_thread is not None:
            audio_thread.join(timeout=3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", default=None, help="Qwen3-Omni MLX model path (未指定ならmock)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval-active", type=float, default=1.0)
    parser.add_argument("--interval-idle", type=float, default=15.0)
    parser.add_argument("--diff-threshold", type=float, default=0.05)
    parser.add_argument("--no-socket", action="store_true", help="socketサーバ無効化")
    parser.add_argument("--audio", action="store_true", help="マイク＋VAD で音声知覚も有効化")
    parser.add_argument("--audio-device", type=int, default=None, help="入力デバイス index (miniaudio)")
    args = parser.parse_args()

    if args.model and Path(args.model).exists():
        perceiver = QwenOmniPerceiver(args.model)
    else:
        print("[daemon] MockPerceiver (モデル未指定/未DL)")
        perceiver = MockPerceiver()

    run_loop(
        session_id=args.session_id,
        perceiver=perceiver,
        interval_active=args.interval_active,
        interval_idle=args.interval_idle,
        diff_threshold=args.diff_threshold,
        camera_index=args.camera,
        enable_socket=not args.no_socket,
        enable_audio=args.audio,
        audio_device_id=args.audio_device,
    )


if __name__ == "__main__":
    main()
