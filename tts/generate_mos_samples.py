# generate_mos_samples.py
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

from melo.api import TTS
from peft import LoraConfig, get_peft_model
import torch

os.makedirs("mos_samples", exist_ok=True)

sentences = [
    "안녕하세요, 오늘 날씨가 정말 좋네요.",
    "어머, 정말요? 너무 잘했어요!",
    "같이 공부해볼까요? 재미있을 것 같아요.",
    "걱정하지 마세요, 제가 도와드릴게요.",
    "와, 대단해요! 정말 훌륭하네요."
]

# 1. Full FT PyTorch (베이스라인)
print("=== Full FT PyTorch 생성 중 ===")
model = TTS(language='KR', device='cuda')
spk_id = model.hps.data.spk2id['KR']
for i, text in enumerate(sentences):
    model.tts_to_file(text, spk_id, f"mos_samples/A_fulft_{i+1}.wav")
    print(f"A_{i+1} 완료")

# 2. LoRA 병합 PyTorch
print("\n=== LoRA PyTorch 생성 중 ===")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["spk_emb_linear"], lora_dropout=0.1, bias="none")
lora_model = get_peft_model(model.model, lora_config)
merged = lora_model.merge_and_unload()
model.model = merged
for i, text in enumerate(sentences):
    model.tts_to_file(text, spk_id, f"mos_samples/B_lora_{i+1}.wav")
    print(f"B_{i+1} 완료")

print("\n완료! mos_samples 폴더 확인해주세요.")
print("A_fulft_*.wav = Full FT")
print("B_lora_*.wav  = LoRA")