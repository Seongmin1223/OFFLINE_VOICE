import wave
import asyncio
import subprocess
import numpy as np
import torch
from config import config

from silero_vad import load_silero_vad, VADIterator

class AudioRecorder:
    def __init__(self):
        self.sample_rate = config.AUDIO_SAMPLE_RATE
        self.channels = config.AUDIO_CHANNELS
        self.chunk_size = 512
        self.max_sec = 60
        self.output_path = config.AUDIO_RECORD_FILE
        
        print("Silero VAD 모델 초기 로딩")
        self.vad_model = load_silero_vad(onnx=True)
        
        self.vad_iterator = VADIterator(
            self.vad_model,
            sampling_rate = self.sample_rate,
            threshold=0.8, #사람 목소리 간주의 임계점(수정 가능)
            min_silence_duration_ms=2000 #2초간 침묵 시 문장이 끝난 것으로 간주(수정 가능)
        )
        print("VAD 사용 준비 완료")
        
    def _record_pyaudio(self) -> str:
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        print('recorder 정상 동작 중')
        
        frames: list[bytes] = []
        has_started = False
        self.vad_iterator.reset_states() #이전 대화 초기화
        
        max_chunks = int(self.max_sec * self.sample_rate / self.chunk_size)
        
        for _ in range(max_chunks):
            chunk = stream.read(self.chunk_size, exception_on_overflow=False)
            
            #pyaudio의 bytes data -> float32 tensor로 변환
            #16bit PCM data -> -1.0 ~ 1.0 으로 정규화
            audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            audio_tensor = torch.from_numpy(audio_np)
            
            #vad 추론 (예상 : < 1ms)
            speech_dict = self.vad_iterator(audio_tensor,return_seconds=False)
            
            if speech_dict:
                if 'start' in speech_dict:
                    has_started = True
                    print("발화 감지됨")
                    
                if 'end' in speech_dict and has_started:
                    print("발화 종료")
                    frames.append(chunk)
                    break
            #audio data buffering
            if not has_started:
                # 말을 하지 않고 대기 중일 때 메모리 터지는 것 방지
                # 직전 1초의 오디오만 유지
                frames.append(chunk)
                keep_chunks = int(self.sample_rate / self.chunk_size) #1s
                if len(frames) > keep_chunks:
                    frames.pop(0)
            else:
                frames.append(chunk)
        
        stream.stop_stream()
        stream.close()
        pa.terminate()

        total_chunks = len(frames)
        duration_sec = (total_chunks * self.chunk_size) / self.sample_rate

        if duration_sec < 0.5: # 0.5초 미만은 잡음으로 간주 (수정 가능)
            print(f"녹음된 음성이 너무 짧습니다. {duration_sec:.2f}초 잡음으로 간주.")
            return
        
        self._save_wav(frames)
        return self.output_path
    
    def _save_wav(self, frames: list[bytes]) -> None:
        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2) #16bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))
            
    def record(self) -> str:
        return self._record_pyaudio()
    
    
    async def record_async(self) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.record)
            