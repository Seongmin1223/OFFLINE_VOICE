# backend/domains/tts/typecast_engine.py
"""
시연용 Typecast (온라인) TTS 엔진.

PiperEngine과 동일한 인터페이스를 노출하므로
`from domains.tts.typecast_engine import TypecastEngine`로 교체만 하면
파이프라인(필러/문장 단위 enqueue/시각화 브로드캐스트)이 그대로 동작한다.

오프라인 MeloTTS+OpenVoice 경로는 piper_engine.py에 그대로 보존.
"""
from __future__ import annotations
import asyncio
import io
import os
import queue
import threading

import requests
import sounddevice as sd
import soundfile as sf

from domains.tts.models import TTSRequest


# ── Typecast 설정 ────────────────────────────────────────────────
TYPECAST_API_URL = "https://api.typecast.ai/v1/text-to-speech"
TYPECAST_API_KEY = os.getenv(
    "TYPECAST_API_KEY",
    "api_key",
)
TYPECAST_VOICE_ID = os.getenv(
    "TYPECAST_VOICE_ID",
    "tc_60915b5616d74069af8e8cab",
)
TYPECAST_MODEL          = os.getenv("TYPECAST_MODEL", "ssfm-v30")
TYPECAST_LANGUAGE       = os.getenv("TYPECAST_LANGUAGE", "kor")
TYPECAST_EMOTION_PRESET = os.getenv("TYPECAST_EMOTION_PRESET", "normal")
TYPECAST_TIMEOUT        = float(os.getenv("TYPECAST_TIMEOUT", "20"))


# ── 엔진 코어 ───────────────────────────────────────────────────
_engine_instance: "_TypecastCore | None" = None


def get_engine() -> "_TypecastCore":
    global _engine_instance
    if _engine_instance is None:
        print("[Typecast] 엔진 초기화")
        _engine_instance = _TypecastCore()
    return _engine_instance


class _TypecastCore:
    """HTTP 호출로 wav 바이트를 받아 numpy로 디코드만 담당."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-KEY":    TYPECAST_API_KEY,
            "Content-Type": "application/json",
        })

    def synthesize(self, text: str):
        payload = {
            "text":     text,
            "model":    TYPECAST_MODEL,
            "voice_id": TYPECAST_VOICE_ID,
            "language": TYPECAST_LANGUAGE,
            "prompt":   {"emotion_preset": TYPECAST_EMOTION_PRESET},
            "output":   {"audio_format": "wav"},
        }
        resp = self._session.post(
            TYPECAST_API_URL, json=payload, timeout=TYPECAST_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"[Typecast] {resp.status_code}: {resp.text[:200]}"
            )
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr


# ── PiperEngine과 동일 인터페이스 ────────────────────────────────
class TypecastEngine:
    """드롭인 교체용. 합성 워커가 HTTP를 때리고 재생 워커가 sounddevice로 출력."""

    _instance: "TypecastEngine | None" = None

    @classmethod
    def get_instance(cls) -> "TypecastEngine":
        """프로세스 전역에서 동일한 TypecastEngine을 공유."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._synth_queue: queue.Queue = queue.Queue()
        self._play_queue:  queue.Queue = queue.Queue()
        self._synth_thread: threading.Thread | None = None
        self._play_thread:  threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        get_engine()

    def set_loop(self, loop):
        self._loop = loop

    def _broadcast_frames(self, samples, sr):
        if self._loop is None:
            return
        try:
            from domains.tts.visualizer import compute_bar_frames, VIS_FPS
            from api.avatar_ws import broadcast
            frames = compute_bar_frames(samples, sr)
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "audio_frames", "frames": frames, "fps": VIS_FPS}),
                self._loop,
            )
        except Exception as e:
            print(f"[Typecast] frames broadcast 오류: {e}")

    def _synth_worker(self):
        while True:
            item = self._synth_queue.get()
            if item is None:
                self._play_queue.put(None)
                break
            try:
                text = item
                engine = get_engine()
                samples, sr = engine.synthesize(text)
                self._play_queue.put((samples, sr))
            except Exception as e:
                print(f"[Typecast] 합성 오류: {e}")
            finally:
                self._synth_queue.task_done()

    def _play_worker(self):
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            try:
                samples, sr = item
                self._broadcast_frames(samples, sr)
                sd.play(samples, sr)
                sd.wait()
            except Exception as e:
                print(f"[Typecast] 재생 오류: {e}")
            finally:
                self._play_queue.task_done()

    def start_workers(self):
        if self._synth_thread is None or not self._synth_thread.is_alive():
            self._synth_thread = threading.Thread(
                target=self._synth_worker, daemon=True)
            self._synth_thread.start()
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(
                target=self._play_worker, daemon=True)
            self._play_thread.start()

    def enqueue(self, text: str):
        self.start_workers()
        self._synth_queue.put(text)

    def wait_done(self):
        self._synth_queue.join()
        self._play_queue.join()

    def speak_sync(self, request: TTSRequest) -> None:
        print(f"[Typecast] 합성: {request.text[:60]}...")
        samples, sr = get_engine().synthesize(request.text)
        print("[Typecast] 재생")
        sd.play(samples, sr)
        sd.wait()

    async def speak(self, text: str) -> None:
        loop    = asyncio.get_event_loop()
        request = TTSRequest(text=text)
        await loop.run_in_executor(None, self.speak_sync, request)
