# backend/domains/tts/piper_engine.py
import asyncio
import queue
import threading
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf

os.environ["PATH"] = r"C:\ffmpeg\bin" + os.pathsep + os.environ["PATH"]
sys.path.insert(0, r'C:\MeloTTS-Windows')

import onnxruntime as ort
from melo.api import TTS as MeloTTS
from melo import utils
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

from domains.tts.models import TTSRequest
from config import config

ONNX_PATH    = r'C:\voice\melotts_base_kr_v2.onnx'
CKPT_PATH    = r'C:\OpenVoice\checkpoints_v2\converter'
NEUTRAL_REF  = r'C:\dev\voices\neutral.wav'
TEMP_TTS     = r'C:\dev\temp_tts.wav'
TEMP_OUT     = r'C:\dev\temp_out.wav'


_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        print("[TTS] 모델 로딩 중... (최초 1회)")
        _engine_instance = _TTSCore()
        print("[TTS] 모델 로딩 완료")
    return _engine_instance


class _TTSCore:
    def __init__(self):
        self.tts  = MeloTTS(language='KR', device='cpu')
        _sess_opts = ort.SessionOptions()
        _sess_opts.intra_op_num_threads = os.cpu_count() or 4
        _sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self.sess = ort.InferenceSession(ONNX_PATH, sess_options=_sess_opts)
        device = 'cpu'
        self.converter = ToneColorConverter(f'{CKPT_PATH}/config.json', device=device)
        self.converter.load_ckpt(f'{CKPT_PATH}/checkpoint.pth')
        # 톤 변환용 단일 reference (neutral)
        print("[TTS] 톤 레퍼런스 추출 중...")
        if not os.path.exists(NEUTRAL_REF):
            raise FileNotFoundError(f"neutral reference 필수: {NEUTRAL_REF}")
        self.tgt_se, _ = se_extractor.get_se(NEUTRAL_REF, self.converter, vad=True)
        print("[TTS] 톤 레퍼런스 추출 완료")
        # MeloTTS 기본 화자의 src_se 캐시 (매 합성마다 whisper 분할 반복 방지)
        self._cached_src_se = None
        # 워밍: 첫 합성은 BERT 임베딩/그래프 초기화로 느려서 미리 한 번 돌려둠
        print("[TTS] 워밍 중...")
        try:
            self.synthesize("준비 완료")
            print("[TTS] 워밍 완료")
        except Exception as e:
            print(f"[TTS] 워밍 스킵: {e}")

    def synthesize(self, text: str):
        bert, ja_bert, phones, tones, _ = utils.get_text_for_tts_infer(
            text, 'KR', self.tts.hps, 'cpu', self.tts.symbol_to_id)
        inputs = {
            'x':             phones.numpy()[np.newaxis, :],
            'x_lengths':     np.array([phones.shape[0]], dtype=np.int64),
            'tones':         tones.numpy()[np.newaxis, :],
            'ja_bert':       ja_bert.numpy()[np.newaxis, :, :],
            'sid':           np.array([0], dtype=np.int64),
            'noise_scale':   np.array([0.667], dtype=np.float32),
            'length_scale':  np.array([1.0],   dtype=np.float32),
            'noise_scale_w': np.array([0.8],   dtype=np.float32),
        }
        out = self.sess.run(None, inputs)
        audio_arr = out[0].squeeze()
        # 짧은 wav는 whisper 분할에 실패하므로 2초 미만이면 무음 패딩
        if len(audio_arr) < 44100 * 2:
            audio_arr = np.pad(audio_arr, (0, 44100 * 2 - len(audio_arr)))
        sf.write(TEMP_TTS, audio_arr, 44100)

        try:
            if self._cached_src_se is None:
                self._cached_src_se, _ = se_extractor.get_se(TEMP_TTS, self.converter, vad=False)
            self.converter.convert(
                audio_src_path=TEMP_TTS,
                src_se=self._cached_src_se,
                tgt_se=self.tgt_se,
                output_path=TEMP_OUT,
            )
            audio, sr = sf.read(TEMP_OUT)
        except Exception as e:
            print(f"[TTS] OpenVoice 변환 실패 ({type(e).__name__}: {e}) → MeloTTS 원본 사용")
            audio, sr = sf.read(TEMP_TTS)
        return audio, sr


class PiperEngine:

    _instance = None

    @classmethod
    def get_instance(cls) -> "PiperEngine":
        """프로세스 전역에서 동일한 PiperEngine을 공유 (큐/스레드/_TTSCore 1개씩)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._synth_queue: queue.Queue = queue.Queue()
        self._play_queue:  queue.Queue = queue.Queue()
        self._synth_thread = None
        self._play_thread  = None
        self._loop = None
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
            print(f"[TTS] frames broadcast 오류: {e}")

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
                print(f"[TTS] 합성 오류: {e}")
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
        self.start_workers()
        self._synth_queue.put(text)

    def wait_done(self):
        self._synth_queue.join()
        self._play_queue.join()

    def speak_sync(self, request: TTSRequest) -> None:
        print(f"[TTS] 음성 합성 중: {request.text[:60]}...")
        engine = get_engine()
        samples, sr = engine.synthesize(request.text)
        print("[TTS] 합성 완료. 재생 중...")
        sd.play(samples, sr)
        sd.wait()

    async def speak(self, text: str) -> None:
        loop    = asyncio.get_event_loop()
        request = TTSRequest(text=text)
        await loop.run_in_executor(None, self.speak_sync, request)
