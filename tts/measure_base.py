# measure_base.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import time
import psutil
import numpy as np
import soundfile as sf
import onnxruntime as ort
from melo.api import TTS

text = "안녕하세요, 오늘 날씨가 정말 좋네요. 같이 산책하러 가실래요?"

# Full FT PyTorch (CPU)
print("=== Full FT PyTorch (CPU) ===")
model = TTS(language='KR', device='cpu')
spk_id = model.hps.data.spk2id['KR']

process = psutil.Process(os.getpid())
start = time.time()
model.tts_to_file(text, spk_id, 'base_pytorch.wav', speed=1.0)
elapsed = time.time() - start
mem = process.memory_info().rss / 1024 / 1024
data, sr = sf.read('base_pytorch.wav')
rtf = elapsed / (len(data) / sr)

import glob
pth_files = glob.glob(r'C:\Users\pc\.cache\cached_path\*.133b77b9d9162e348486a0a0778fa47d726930e3ec12ea5e2684c0c919743a65')
model_size = os.path.getsize(pth_files[0]) / 1024 / 1024 if pth_files else 0

print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {model_size:.1f} MB")

# ONNX FP32
def measure_onnx(path):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    x = np.random.randint(0, 10, size=(1, 30), dtype=np.int64)
    x_lengths = np.array([30], dtype=np.int64)
    tones = np.zeros((1, 30), dtype=np.int64)
    sid = np.array([0], dtype=np.int64)
    noise_scale = np.array([1.0], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)
    noise_scale_w = np.array([1.0], dtype=np.float32)
    process = psutil.Process(os.getpid())
    start = time.time()
    output = sess.run(None, {"x": x, "x_lengths": x_lengths, "tones": tones,
        "sid": sid, "noise_scale": noise_scale, "length_scale": length_scale, "noise_scale_w": noise_scale_w})
    elapsed = time.time() - start
    mem = process.memory_info().rss / 1024 / 1024
    audio = output[0].squeeze()
    rtf = elapsed / (len(audio) / 48000)
    size = os.path.getsize(path) / 1024 / 1024
    return rtf, mem, size

print("\n=== Full FT ONNX FP32 ===")
rtf, mem, size = measure_onnx("melotts_base_fp32.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")

print("\n=== Full FT ONNX INT8 ===")
rtf, mem, size = measure_onnx("melotts_base_int8.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")