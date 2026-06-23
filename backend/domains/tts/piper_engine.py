from __future__ import annotations

import asyncio
import os
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from config import config
from domains.tts.models import TTSRequest


_engine_instance: "_MeloTTSCore | None" = None

_TTS_TERM_READINGS = {
    "Firewatcher": "파이어워처",
    "Secuwatcher": "시큐워처",
    "Safewatcher": "세이프워처",
    "Crowdwatcher": "크라우드워처",
    "Viscoper": "비스코퍼",
    "Map2D": "맵 투디",
    "EventHistory": "이벤트 히스토리",
    "Tomcat": "톰캣",
    "Java": "자바",
    "JDK": "제이디케이",
    "JRE": "제이알이",
    "XML": "엑스엠엘",
    "JSON": "제이슨",
    "CSV": "씨에스브이",
    "WAV": "웨이브",
    "STT": "에스티티",
    "LLM": "엘엘엠",
    "TTS": "티티에스",
    "ONNX": "오닉스",
    "API": "에이피아이",
    "CPU": "씨피유",
    "GPU": "지피유",
}

_LETTER_READINGS = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이",
    "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐",
    "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이",
    "W": "더블유", "X": "엑스", "Y": "와이", "Z": "지",
}


def _path_from_env(value: str, default: Path) -> Path:
    if value and value.strip() and value.lower() != "unused":
        return Path(value)
    return default


def _normalize_for_korean_tts(text: str) -> str:
    text = re.sub(r"\[case:[^\]]*\]", "", text)
    text = re.sub(r"#\d{4}", "", text)

    for term, reading in sorted(_TTS_TERM_READINGS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", reading, text, flags=re.IGNORECASE)

    def _spell_acronym(match: re.Match) -> str:
        token = match.group(0)
        if len(token) > 6:
            return token
        return " ".join(_LETTER_READINGS.get(ch, ch) for ch in token.upper())

    text = re.sub(r"\b[A-Z]{2,6}\b", _spell_acronym, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prepare_mecab() -> None:
    mecab_bin = Path(os.getenv("MECAB_BIN_PATH", r"C:\Program Files\MeCab\bin"))
    if hasattr(os, "add_dll_directory") and mecab_bin.is_dir():
        os.add_dll_directory(str(mecab_bin))
    try:
        import unidic
        import unidic_lite

        unidic.DICDIR = unidic_lite.DICDIR
        os.environ.setdefault("MECABRC", str(Path(unidic_lite.DICDIR) / "mecabrc"))
    except ImportError:
        pass


class _MeloTTSCore:
    sample_rate = 44100

    def __init__(self) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        default_model = backend_dir / "models" / "tts" / "melotts" / "melotts_base_kr_v2.onnx"
        self.model_path = _path_from_env(config.TTS_MODEL, default_model)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MeloTTS ONNX model not found: {self.model_path}")

        repo_path = Path(os.getenv("MELOTTS_REPO_PATH", backend_dir / "vendor" / "MeloTTS-Windows"))
        if repo_path.is_dir() and str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        _prepare_mecab()

        from melo.api import TTS
        from melo import utils

        print("[TTS] Loading MeloTTS KR frontend")
        self.tts = TTS(language="KR", device="cpu")
        self.hps = self.tts.hps
        self.utils = utils
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        print("[TTS] MeloTTS ONNX ready")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = _normalize_for_korean_tts(text)
        if not text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        _bert, ja_bert, phones, tones, _lang_ids = self.utils.get_text_for_tts_infer(
            text,
            "KR",
            self.hps,
            "cpu",
            self.tts.symbol_to_id,
        )
        inputs = {
            "x": phones.detach().cpu().numpy()[np.newaxis, :],
            "x_lengths": np.array([phones.shape[0]], dtype=np.int64),
            "tones": tones.detach().cpu().numpy()[np.newaxis, :],
            "ja_bert": ja_bert.detach().cpu().numpy()[np.newaxis, :, :],
            "sid": np.array([0], dtype=np.int64),
            "noise_scale": np.array([0.667], dtype=np.float32),
            "length_scale": np.array([1.0], dtype=np.float32),
            "noise_scale_w": np.array([0.8], dtype=np.float32),
        }
        return self.session.run(None, inputs)[0].squeeze().astype(np.float32), self.sample_rate


def get_engine() -> _MeloTTSCore:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = _MeloTTSCore()
    return _engine_instance


class PiperEngine:
    _instance: "PiperEngine | None" = None

    @classmethod
    def get_instance(cls) -> "PiperEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._synth_queue: queue.Queue[str | None] = queue.Queue()
        self._play_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        get_engine()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _broadcast_frames(self, samples: np.ndarray, sr: int) -> None:
        if self._loop is None or samples.size == 0:
            return
        try:
            from api.avatar_ws import broadcast
            from domains.tts.visualizer import VIS_FPS, compute_bar_frames

            frames = compute_bar_frames(samples, sr)
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "audio_frames", "frames": frames, "fps": VIS_FPS}),
                self._loop,
            )
        except Exception as exc:
            print(f"[TTS] visualizer frame broadcast failed: {exc}")

    def _synth_worker(self) -> None:
        while True:
            item = self._synth_queue.get()
            if item is None:
                self._play_queue.put(None)
                self._synth_queue.task_done()
                break
            try:
                self._play_queue.put(get_engine().synthesize(item))
            except Exception as exc:
                print(f"[TTS] synthesis error: {exc}")
            finally:
                self._synth_queue.task_done()

    def _play_worker(self) -> None:
        while True:
            item = self._play_queue.get()
            if item is None:
                self._play_queue.task_done()
                break
            try:
                samples, sr = item
                self._broadcast_frames(samples, sr)
                if samples.size:
                    sd.play(samples, sr)
                    sd.wait()
            except Exception as exc:
                print(f"[TTS] playback error: {exc}")
            finally:
                self._play_queue.task_done()

    def start_workers(self) -> None:
        if self._synth_thread is None or not self._synth_thread.is_alive():
            self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
            self._synth_thread.start()
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
            self._play_thread.start()

    def enqueue(self, text: str) -> None:
        self.start_workers()
        self._synth_queue.put(text)

    def wait_done(self) -> None:
        self._synth_queue.join()
        self._play_queue.join()

    def speak_sync(self, request: TTSRequest) -> None:
        samples, sr = get_engine().synthesize(request.text)
        sd.play(samples, sr)
        sd.wait()

    async def speak(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak_sync, TTSRequest(text=text))
