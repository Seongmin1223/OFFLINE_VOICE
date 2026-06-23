import asyncio
import os
import re
import struct
import tempfile
import wave

import requests

from config import config
from domains.stt.models import STTResult


_NOISE_PATTERN = re.compile(r'^[\s\-\.\,\!\/\(\)\[\]]+$')


class WhisperEngine:
    """파인튜닝 Whisper-small(아동음성)을 whisper.cpp 서버로 추론.

    transformers in-process 대비 encode가 빠르고(q8_0 양자화), 모델이 서버에
    상주하므로 요청당 모델 로드 비용이 없다. config.WHISPER_SERVER_URL(기본
    127.0.0.1:8081)의 whisper-server.exe 에 WAV를 업로드해 전사한다.
    """

    def __init__(self):
        self.server_url = config.WHISPER_SERVER_URL
        self.language   = config.WHISPER_LANGUAGE

    def _check_server(self) -> None:
        try:
            requests.get(f"{self.server_url}/", timeout=10)
        except Exception as e:
            raise RuntimeError(
                f"whisper-server 연결 실패 [{type(e).__name__}] {e}\n"
                f"  target: {self.server_url}/\n"
                "  먼저 whisper-server를 실행하세요:\n"
                f"  {config.WHISPER_BIN} -m {config.WHISPER_MODEL} "
                f"-l {config.WHISPER_LANGUAGE} --port {self.server_url.split(':')[-1]}"
            )

    def _is_noise(self, text: str) -> bool:
        if not text or len(text) < 2:
            return True
        if _NOISE_PATTERN.match(text):
            return True
        if not re.search(r'[가-힣a-zA-Z]', text):
            return True
        return False

    def transcribe_sync(self, audio_path: str) -> STTResult:
        self._check_server()

        print(f"[STT] 음성 인식 중... ({audio_path})")
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.server_url}/inference",
                files={"file": f},
                data={"language": self.language, "temperature": "0.0"},
                timeout=30,
            )

        if response.status_code != 200:
            raise RuntimeError(f"whisper-server 오류: {response.text}")

        text = response.json().get("text", "").strip()

        if self._is_noise(text):
            print(f"[STT] 노이즈 필터링: {text!r}")
            text = ""

        print(f"[STT] 인식 결과: {text!r}")
        return STTResult(text=text)

    async def transcribe(self, audio_path: str) -> str:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.transcribe_sync, audio_path)
        return result.text

    # ── 워밍업 ────────────────────────────────────────────
    def _silent_wav(self, seconds: float = 0.5) -> str:
        """encode 그래프를 데우기 위한 짧은 무음 16kHz mono WAV 생성(임시 파일)."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="stt_warm_")
        os.close(fd)
        n = int(16000 * seconds)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
        return path

    def warmup_sync(self) -> None:
        """서버 기동 직후 1회: 무음 클립으로 추론 경로(encode)를 미리 데운다."""
        path = None
        try:
            self._check_server()
            path = self._silent_wav()
            with open(path, "rb") as f:
                requests.post(
                    f"{self.server_url}/inference",
                    files={"file": f},
                    data={"language": self.language, "temperature": "0.0"},
                    timeout=60,
                )
            print("[Warmup] STT encode 워밍업 완료")
        except Exception as e:
            print(f"[Warmup] STT 워밍업 건너뜀: {e}")
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    async def warmup(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.warmup_sync)
