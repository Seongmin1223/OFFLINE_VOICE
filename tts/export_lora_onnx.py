# export_lora_onnx.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import torch
import onnx
import numpy as np
from melo.api import TTS
from peft import LoraConfig, get_peft_model
from melo.text import language_id_map

config_path = r'C:\MeloTTS-Windows\melo\logs\data\example\config\config.json'
ckpt_path   = r'C:\MeloTTS-Windows\melo\logs\data\example\config\G_64000.pth'

model = TTS(language='KR', device='cpu', config_path=config_path, ckpt_path=ckpt_path)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["spk_emb_linear"],
    lora_dropout=0.1, bias="none",
)
lora_model = get_peft_model(model.model, lora_config)

# 어댑터 병합
merged_model = lora_model.merge_and_unload()
merged_model.eval()

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

wrapper = ModelWrapper(merged_model)
wrapper.eval()

x = torch.randint(0, 10, size=(1, 60), dtype=torch.int64)
x_lengths = torch.tensor([60], dtype=torch.int64)
sid = torch.tensor([0], dtype=torch.int64)
tones = torch.zeros_like(x)
noise_scale = torch.tensor([1.0], dtype=torch.float32)
length_scale = torch.tensor([1.0], dtype=torch.float32)
noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

filename = "melotts_lora_fp32.onnx"
print("LoRA ONNX FP32 변환 중...")
torch.onnx.export(
    wrapper,
    (x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w),
    filename,
    opset_version=17,
    input_names=["x", "x_lengths", "tones", "sid", "noise_scale", "length_scale", "noise_scale_w"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "N", 1: "L"},
        "x_lengths": {0: "N"},
        "tones": {0: "N", 1: "L"},
        "y": {0: "N", 1: "S", 2: "T"},
    },
)
size = os.path.getsize(filename) / 1024 / 1024
print(f"완료! 크기: {size:.1f} MB")