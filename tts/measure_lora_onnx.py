# measure_lora_onnx.py
import os
import time
import psutil
import numpy as np
import soundfile as sf
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# INT8 양자화
print("INT8 양자화 중...")
quantize_dynamic(
    "melotts_lora_fp32.onnx",
    "melotts_lora_int8.onnx",
    weight_type=QuantType.QUInt8
)

def measure_onnx(onnx_path):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
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
    audio_len = len(audio) / 48000
    rtf = elapsed / audio_len
    size = os.path.getsize(onnx_path) / 1024 / 1024
    return rtf, mem, size

print("\n=== LoRA ONNX FP32 ===")
rtf, mem, size = measure_onnx("melotts_lora_fp32.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")

print("\n=== LoRA ONNX INT8 ===")
rtf, mem, size = measure_onnx("melotts_lora_int8.onnx")
print(f"RTF: {rtf:.3f}")
print(f"메모리: {mem:.1f} MB")
print(f"모델크기: {size:.1f} MB")