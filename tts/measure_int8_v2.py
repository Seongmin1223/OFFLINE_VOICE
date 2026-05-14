# measure_int8_v2.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import time
import psutil
import numpy as np
import soundfile as sf
import onnxruntime as ort

def measure_onnx(onnx_path, output_file):
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    x = np.random.randint(0, 10, size=(1, 30), dtype=np.int64)
    x_lengths = np.array([30], dtype=np.int64)
    tones = np.zeros((1, 30), dtype=np.int64)
    sid = np.array([0], dtype=np.int64)
    noise_scale = np.array([1.0], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)
    noise_scale_w = np.array([1.0], dtype=np.float32)

    process = psutil.Process(os.getpid())
    start = time.time()
    output = sess.run(None, {
        "x": x, "x_lengths": x_lengths, "tones": tones,
        "sid": sid, "noise_scale": noise_scale,
        "length_scale": length_scale, "noise_scale_w": noise_scale_w
    })
    elapsed = time.time() - start
    mem = process.memory_info().rss / 1024 / 1024

    audio = output[0].squeeze()
    sf.write(output_file, audio, 48000)
    audio_len = len(audio) / 48000
    rtf = elapsed / audio_len
    size = os.path.getsize(onnx_path) / 1024 / 1024
    return rtf, mem, size

print("=== ONNX INT8 v2 ===")
rtf, mem, size = measure_onnx("melotts_kr_finetuned_int8_v2.onnx", "test_int8_v2.wav")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")