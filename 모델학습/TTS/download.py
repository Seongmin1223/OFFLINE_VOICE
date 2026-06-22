from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

# 로컬에 저장
model.save_pretrained("./models/mms-tts-kor")
tokenizer.save_pretrained("./models/mms-tts-kor")

print("모델 저장 완료!")