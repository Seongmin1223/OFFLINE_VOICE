import wave
import struct
import math
import asyncio
import subprocess
import tempfile
import os
from config import config


class AudioRecorder:
    """
    저사양 환경을 고려한 마이크 녹음기.
    pyaudio 또는 arecord(Linux) 둘 다 지원합니다.
    """

    def __init__(self):
        self.sample_rate  = config.AUDIO_SAMPLE_RATE
        self.channels     = config.AUDIO_CHANNELS
        self.chunk_size   = config.AUDIO_CHUNK_SIZE
        self.silence_thresh = config.AUDIO_SILENCE_THRESH
        self.silence_sec  = config.AUDIO_SILENCE_SEC
        self.max_sec      = config.AUDIO_MAX_SEC
        self.output_path  = config.AUDIO_RECORD_FILE
    @staticmethod
    def _rms(data: bytes) -> float:
        """16-bit PCM 청크의 RMS(음량) 계산."""
        count = len(data) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack(f"{count}h", data)
        mean_sq = sum(s * s for s in shorts) / count
        return math.sqrt(mean_sq) / 32768.0

    def _record_pyaudio(self) -> str:
        import pyaudio  # optional dependency
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("[Recorder] 말씀해 주세요... (침묵 감지 시 자동 종료)")
        frames: list[bytes] = []
        silent_chunks = 0
        max_chunks    = int(self.max_sec * self.sample_rate / self.chunk_size)
        silence_limit = int(self.silence_sec * self.sample_rate / self.chunk_size)

        for _ in range(max_chunks):
            chunk = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(chunk)
            if self._rms(chunk) < self.silence_thresh:
                silent_chunks += 1
                if silent_chunks >= silence_limit and len(frames) > silence_limit:
                    break
            else:
                silent_chunks = 0

        stream.stop_stream()
        stream.close()
        pa.terminate()

        self._save_wav(frames)
        print(f"[Recorder] 녹음 완료 → {self.output_path}")
        return self.output_path

    def _record_arecord(self) -> str:
        duration = int(self.max_sec)
        cmd = [
            "arecord",
            "-f", "S16_LE",
            "-r", str(self.sample_rate),
            "-c", str(self.channels),
            "-d", str(duration),
            self.output_path,
        ]
        print(f"[Recorder] arecord 녹음 시작 (최대 {duration}초)...")
        subprocess.run(cmd, check=True)
        print(f"[Recorder] 녹음 완료 → {self.output_path}")
        return self.output_path
    def _save_wav(self, frames: list[bytes]) -> None:
        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))
    def record(self) -> str:
        """동기 녹음. 사용 가능한 백엔드를 자동 선택합니다."""
        try:
            return self._record_pyaudio()
        except ImportError:
            pass
        try:
            return self._record_arecord()
        except FileNotFoundError:
            raise RuntimeError(
                "오디오 녹음 백엔드를 찾을 수 없습니다.\n"
                "  pip install pyaudio  또는  sudo apt install alsa-utils"
            )

    async def record_async(self) -> str:
        """비동기 래퍼 (이벤트 루프 블로킹 방지)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.record)