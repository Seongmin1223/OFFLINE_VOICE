import whisper
import numpy as np
import librosa
from datasets import load_from_disk
import re

print("모델 로딩 중...")
model = whisper.load_model("small", download_root="./models/whisper")

print("데이터셋 로딩 중...")
ds = load_from_disk("./data/kss")
print(f"총 샘플 수: {len(ds)}")

def normalize(text):
    # 문장부호 제거, 소문자, 공백 정리
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip().lower()
    return text

def cer(reference, hypothesis):
    # CER 계산 (편집거리 기반)
    r = list(reference)
    h = list(hypothesis)
    d = np.zeros((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)
    return d[len(r)][len(h)] / max(len(r), 1)

total = len(ds)
cer_scores = []
exact_match = 0

for i in range(total):
    sample = ds[i]
    audio_data = np.array(sample['audio']['array'], dtype=np.float32)
    sample_rate = sample['audio']['sampling_rate']
    reference = normalize(sample['original_script'])

    # 리샘플링
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    result = model.transcribe(audio_data, language="ko")
    hypothesis = normalize(result['text'])

    # CER 계산
    score = cer(reference, hypothesis)
    cer_scores.append(score)

    # 완전 일치
    if reference == hypothesis:
        exact_match += 1

    # 진행상황 출력
    if (i+1) % 10 == 0:
        print(f"진행: {i+1}/{total} | 현재 평균 CER: {np.mean(cer_scores):.4f}")

# 최종 결과
print("\n========== 결과 ==========")
print(f"총 샘플:       {total}개")
print(f"완전 일치:     {exact_match}개 ({exact_match/total*100:.1f}%)")
print(f"평균 CER:      {np.mean(cer_scores):.4f} ({np.mean(cer_scores)*100:.1f}%)")
print(f"CER 0.2 이하:  {sum(1 for s in cer_scores if s <= 0.2)}개 ({sum(1 for s in cer_scores if s <= 0.2)/total*100:.1f}%)")
print("==========================")