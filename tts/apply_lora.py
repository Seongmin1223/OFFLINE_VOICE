# apply_lora.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

import torch
from melo.api import TTS
from peft import LoraConfig, get_peft_model

config_path = r'C:\MeloTTS-Windows\melo\logs\data\example\config\config.json'
ckpt_path   = r'C:\MeloTTS-Windows\melo\logs\data\example\config\G_64000.pth'

model = TTS(language='KR', device='cpu', config_path=config_path, ckpt_path=ckpt_path)

total_params = sum(p.numel() for p in model.model.parameters())
print(f"전체 파라미터: {total_params:,}")

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["spk_emb_linear"],
    lora_dropout=0.1,
    bias="none",
)

lora_model = get_peft_model(model.model, lora_config)
lora_model.print_trainable_parameters()

trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"LoRA 학습 파라미터: {trainable:,} ({trainable/total_params*100:.2f}%)")

# 어댑터 저장
lora_model.save_pretrained("lora_adapter")
adapter_size = sum(
    os.path.getsize(os.path.join("lora_adapter", f))
    for f in os.listdir("lora_adapter")
) / 1024 / 1024
print(f"어댑터 크기: {adapter_size:.2f} MB")