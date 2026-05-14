# quantize_int8.py
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "melotts_kr_finetuned.onnx",
    "melotts_kr_finetuned_int8.onnx",
    weight_type=QuantType.QInt8
)

import os
fp32_size = os.path.getsize("melotts_kr_finetuned.onnx") / 1024 / 1024
int8_size = os.path.getsize("melotts_kr_finetuned_int8.onnx") / 1024 / 1024
print(f"FP32: {fp32_size:.1f} MB")
print(f"INT8: {int8_size:.1f} MB")
print(f"압축률: {(1-int8_size/fp32_size)*100:.1f}%")