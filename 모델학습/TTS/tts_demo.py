import torch
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
import uroman as ur

print("한국어 TTS 모델을 불러오는 중...")

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

text = "안녕 윤호야? 오늘 학업은 좀 어때? 내가 도와줄 일이 있으면 언제든 말해!"

# uroman으로 한국어 → 로마자 변환
print("텍스트 로마자 변환 중...")
romanizer = ur.Uroman()
romanized_text = romanizer.romanize_string(text)
print(f"변환 결과: {romanized_text}")

print("음성 합성 중...")
inputs = tokenizer(romanized_text, return_tensors="pt")

# input_ids를 Long 타입으로 명시적 변환
inputs["input_ids"] = inputs["input_ids"].long()

with torch.no_grad():
    output = model(**inputs)

audio = output.waveform.squeeze().numpy()
audio_int16 = (audio * 32767).astype(np.int16)

output_path = "output_tts.wav"
wavfile.write(output_path, model.config.sampling_rate, audio_int16)
print(f"완료! '{output_path}' 파일을 재생해봐!")