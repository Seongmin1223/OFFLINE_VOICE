import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro
from domains.tts.models import TTSRequest
from config import config


_kokoro_instance = None

def get_kokoro() -> Kokoro:
    global _kokoro_instance
    if _kokoro_instance is None:
        print("[TTS] kokoro 모델 로딩 중... (최초 1회)")
        _kokoro_instance = Kokoro(
            config.TTS_MODEL,
            config.TTS_CONFIG
        )
        print("[TTS] kokoro 모델 로딩 완료")
    return _kokoro_instance


class PiperEngine:

    def __init__(self):
        self.voice = "af_kore"
        self._synth_queue: queue.Queue = queue.Queue()  # 합성 대기
        self._play_queue:  queue.Queue = queue.Queue()  # 재생 대기
        self._synth_thread = None
        self._play_thread  = None
        get_kokoro()

    # ── 합성 워커: 텍스트 → 오디오 샘플 ──────────────────
    def _synth_worker(self):
        while True:
            item = self._synth_queue.get()
            if item is None:
                self._play_queue.put(None)
                break
            try:
                kokoro  = get_kokoro()
                samples, sr = kokoro.create(
                    item, voice=self.voice, speed=1.0, lang="ko"
                )
                self._play_queue.put((samples, sr))
            except Exception as e:
                print(f"[TTS] 합성 오류: {e}")
            finally:
                self._synth_queue.task_done()

    # ── 재생 워커: 오디오 샘플 → 스피커 ─────────────────
    def _play_worker(self):
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            try:
                samples, sr = item
                sd.play(samples, sr)
                sd.wait()
            except Exception as e:
                print(f"[TTS] 재생 오류: {e}")
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
        """문장을 합성 큐에 추가."""
        self.start_workers()
        self._synth_queue.put(text)

    def wait_done(self):
        """합성+재생 모두 완료될 때까지 대기."""
        self._synth_queue.join()
        self._play_queue.join()

    def speak_sync(self, request: TTSRequest) -> None:
        print(f"[TTS] 음성 합성 중: {request.text[:60]}...")
        kokoro = get_kokoro()
        samples, sr = kokoro.create(
            request.text, voice=self.voice, speed=1.0, lang="ko"
        )
        print("[TTS] 합성 완료. 재생 중...")
        sd.play(samples, sr)
        sd.wait()

    async def speak(self, text: str) -> None:
        loop    = asyncio.get_event_loop()
        request = TTSRequest(text=text)
        await loop.run_in_executor(None, self.speak_sync, request)