# export_lora_base.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import torch
from melo.api import TTS
from peft import LoraConfig, get_peft_model
from melo.text import language_id_map
from onnxruntime.quantization import quantize_dynamic, QuantType

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lang_id = language_id_map["KR"]

    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        lang_id = torch.zeros_like(x)
        lang_id[:, 1::2] = self.lang_id
        return self.model.infer(
            x=x, x_lengths=x_lengths, sid=sid, tone=tones,
            language=lang_id, bert=bert, ja_bert=ja_bert,
            noise_scale=noise_scale, noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0]

print("기존 MeloTTS KR 로드 중...")
model = TTS(language='KR', device='cpu')

total = sum(p.numel() for p in model.model.parameters())
print(f"전체 파라미터: {total:,}")

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["spk_emb_linear"],
    lora_dropout=0.1, bias="none",
)
lora_model = get_peft_model(model.model, lora_config)
lora_model.print_trainable_parameters()

merged = lora_model.merge_and_unload()
merged.eval()

wrapper = ModelWrapper(merged)
wrapper.eval()

x = torch.randint(0, 10, size=(1, 60), dtype=torch.int64)
x_lengths = torch.tensor([60], dtype=torch.int64)
sid = torch.tensor([0], dtype=torch.int64)
tones = torch.zeros_like(x)
noise_scale = torch.tensor([1.0], dtype=torch.float32)
length_scale = torch.tensor([1.0], dtype=torch.float32)
noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

print("LoRA ONNX FP32 변환 중...")
torch.onnx.export(
    wrapper,
    (x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w),
    "melotts_base_lora_fp32.onnx",
    opset_version=17,
    input_names=["x","x_lengths","tones","sid","noise_scale","length_scale","noise_scale_w"],
    output_names=["y"],
    dynamic_axes={"x":{0:"N",1:"L"},"x_lengths":{0:"N"},"tones":{0:"N",1:"L"},"y":{0:"N",1:"S",2:"T"}},
)
fp32_size = os.path.getsize("melotts_base_lora_fp32.onnx") / 1024 / 1024
print(f"FP32 완료! 크기: {fp32_size:.1f} MB")

print("INT8 양자화 중...")
quantize_dynamic("melotts_base_lora_fp32.onnx", "melotts_base_lora_int8.onnx", weight_type=QuantType.QUInt8)
int8_size = os.path.getsize("melotts_base_lora_int8.onnx") / 1024 / 1024
print(f"INT8 완료! 크기: {int8_size:.1f} MB")