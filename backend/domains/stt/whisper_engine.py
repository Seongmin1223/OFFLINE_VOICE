import asyncio
import re
from config import config
from domains.stt.models import STTResult


_NOISE_PATTERN = re.compile(r'^[\s\-\.\,\!\/\(\)\[\]]+$')


class WhisperEngine:
    """파인튜닝 Whisper-small(아동음성 LoRA 병합본)을 transformers로 in-process 추론.
    whisper.cpp 서버(8081) 불필요 — 모델을 서버 기동 시 RAM에 적재한다.
    config.WHISPER_MODEL 은 merged HF 모델 디렉토리를 가리킨다.
    """

    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch

        model_path    = config.WHISPER_MODEL
        self.language = config.WHISPER_LANGUAGE

        print(f"[STT] 파인튜닝 Whisper 로딩... ({model_path})")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.processor = WhisperProcessor.from_pretrained(model_path)
        # 한국어·전사 강제 (transformers 4.27 호환: forced_decoder_ids)
        self._forced = self.processor.get_decoder_prompt_ids(
            language="korean", task="transcribe"
        )
        try:
            torch.set_num_threads(config.WHISPER_THREADS)
        except Exception:
            pass
        self._torch = torch
        print("[STT] Whisper 준비 완료")

    def _is_noise(self, text: str) -> bool:
        if not text or len(text) < 2:
            return True
        if _NOISE_PATTERN.match(text):
            return True
        if not re.search(r'[가-힣a-zA-Z]', text):
            return True
        return False

    def transcribe_sync(self, audio_path: str) -> STTResult:
        import soundfile as sf

        print(f"[STT] 음성 인식 중... ({audio_path})")
        audio, sr = sf.read(audio_path)
        # 스테레오면 모노로 평균 (녹음 파일은 보통 16kHz mono)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        feat = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with self._torch.no_grad():
            ids = self.model.generate(feat, forced_decoder_ids=self._forced)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        if self._is_noise(text):
            print(f"[STT] 노이즈 필터링: {text!r}")
            text = ""

        print(f"[STT] 인식 결과: {text!r}")
        return STTResult(text=text)

    async def transcribe(self, audio_path: str) -> str:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.transcribe_sync, audio_path)
        return result.text
