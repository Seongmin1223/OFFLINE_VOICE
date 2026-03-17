import asyncio
import re
import requests
from config import config
from domains.stt.models import STTResult


_NOISE_PATTERN = re.compile(r'^[\s\-\.\,\!\/\(\)\[\]]+$')


class WhisperEngine:

    def __init__(self):
        self.server_url = config.WHISPER_SERVER_URL
        self.language   = config.WHISPER_LANGUAGE

    def _check_server(self) -> None:
        try:
            requests.get(f"{self.server_url}/", timeout=3)
        except Exception:
            raise RuntimeError(
                "whisper-server가 실행되지 않았습니다!\n"
                "새 터미널에서 먼저 실행:\n"
                f"{config.WHISPER_BIN} "
                f"-m {config.WHISPER_MODEL} -l {config.WHISPER_LANGUAGE} --port {config.WHISPER_SERVER_URL.split(':')[-1]}"
            )

    def _is_noise(self, text: str) -> bool:
        """노이즈성 텍스트 여부 확인."""
        if not text or len(text) < 2:
            return True
        # 특수문자/하이픈만 있는 경우
        if _NOISE_PATTERN.match(text):
            return True
        # 한글/영문 한 글자도 없는 경우
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
                data={"language": self.language},
                timeout=30,
            )

        if response.status_code != 200:
            raise RuntimeError(f"whisper-server 오류: {response.text}")

        text = response.json().get("text", "").strip()

        # 노이즈 필터링
        if self._is_noise(text):
            print(f"[STT] 노이즈 필터링: {text!r}")
            text = ""

        print(f"[STT] 인식 결과: {text!r}")
        return STTResult(text=text)

    async def transcribe(self, audio_path: str) -> str:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.transcribe_sync, audio_path)
        return result.text