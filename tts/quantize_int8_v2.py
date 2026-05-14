# quantize_int8_v2.py
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "melotts_kr_finetuned.onnx",
    "melotts_kr_finetuned_int8_v2.onnx",
    weight_type=QuantType.QUInt8  # QInt8 대신 QUInt8
)

import os
size = os.path.getsize("melotts_kr_finetuned_int8_v2.onnx") / 1024 / 1024
print(f"INT8 v2 크기: {size:.1f} MB")