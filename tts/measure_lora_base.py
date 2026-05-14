# measure_lora_base.py
import os
import time
import psutil
import numpy as np
import onnxruntime as ort

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

print("=== LoRA ONNX FP32 ===")
rtf, mem, size = measure_onnx("melotts_base_lora_fp32.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")

print("\n=== LoRA ONNX INT8 ===")
rtf, mem, size = measure_onnx("melotts_base_lora_int8.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")