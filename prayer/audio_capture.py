"""prayer の聴覚末梢 — VAD で発話区間を切り出して wav に保存。

miniaudio (PortAudio wrapper) で16kHz 16-bit mono キャプチャし、webrtcvad で
発話区間を検出。区間終了時に callback(wav_path, duration_sec) を呼ぶ。
"""

import collections
import struct
import time
import uuid
import wave
from pathlib import Path


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit = 2 bytes/sample


def write_wav(path, pcm_bytes, sample_rate=SAMPLE_RATE):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)


class VadCapturer:
    """連続キャプチャ + VAD で発話区間を検出。

    min_speech_ms: これ未満の発話は捨てる（クリック音など）
    end_silence_ms: この長さの無音で発話終了とみなす
    max_speech_ms: ハードリミット（長すぎる発話は途中で切る）
    pre_buffer_ms: 発話検出時、直前のこの長さ分も含める（立ち上がり切り捨て防止）
    """

    def __init__(self, out_dir, on_segment, device_id=None,
                 vad_aggressiveness=2,
                 min_speech_ms=400,
                 end_silence_ms=500,
                 max_speech_ms=15000,
                 pre_buffer_ms=300):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.on_segment = on_segment
        self.device_id = device_id
        self.vad_aggressiveness = vad_aggressiveness
        self.min_speech_samples = min_speech_ms * SAMPLE_RATE // 1000
        self.end_silence_frames = end_silence_ms // FRAME_MS
        self.max_speech_samples = max_speech_ms * SAMPLE_RATE // 1000
        pre_frames = pre_buffer_ms // FRAME_MS
        self._pre_buffer = collections.deque(maxlen=pre_frames)

    def _emit(self, pcm_bytes):
        if len(pcm_bytes) < self.min_speech_samples * 2:
            return
        uid = uuid.uuid4().hex[:12]
        path = self.out_dir / f"speech_{int(time.time())}_{uid}.wav"
        write_wav(path, pcm_bytes)
        duration = len(pcm_bytes) / 2 / SAMPLE_RATE
        try:
            self.on_segment(str(path), duration)
        except Exception as e:
            print(f"[audio] on_segment callback error: {e}")

    def run(self, stop_event):
        import miniaudio
        import webrtcvad

        vad = webrtcvad.Vad(self.vad_aggressiveness)

        # ジェネレータ: 30ms ブロックずつ送る
        def consumer():
            speech_pcm = bytearray()
            in_speech = False
            silence_run = 0
            while not stop_event.is_set():
                chunk = yield
                if chunk is None:
                    continue
                # miniaudio は buffer を返す。bytes にする
                pcm = bytes(chunk)
                # webrtcvad は 10/20/30ms のピッタリのフレームを要求
                if len(pcm) != FRAME_BYTES:
                    continue
                is_speech = vad.is_speech(pcm, SAMPLE_RATE)

                if not in_speech:
                    self._pre_buffer.append(pcm)
                    if is_speech:
                        in_speech = True
                        speech_pcm = bytearray()
                        for pre in self._pre_buffer:
                            speech_pcm.extend(pre)
                        speech_pcm.extend(pcm)
                        silence_run = 0
                else:
                    speech_pcm.extend(pcm)
                    if is_speech:
                        silence_run = 0
                    else:
                        silence_run += 1
                    if (silence_run >= self.end_silence_frames
                            or len(speech_pcm) >= self.max_speech_samples * 2):
                        self._emit(bytes(speech_pcm))
                        in_speech = False
                        speech_pcm = bytearray()
                        silence_run = 0
                        self._pre_buffer.clear()

        gen = consumer()
        next(gen)  # prime

        device = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
            buffersize_msec=FRAME_MS,
            device_id=self.device_id,
        )

        def feeder():
            while not stop_event.is_set():
                frames = yield
                if frames is None:
                    continue
                # frames は bytes か array.array。どちらも bytes に
                pcm = frames if isinstance(frames, (bytes, bytearray)) else bytes(frames)
                for i in range(0, len(pcm), FRAME_BYTES):
                    block = pcm[i:i+FRAME_BYTES]
                    if len(block) == FRAME_BYTES:
                        try:
                            gen.send(block)
                        except StopIteration:
                            return

        f = feeder()
        next(f)
        device.start(f)
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        finally:
            device.stop()
            device.close()
