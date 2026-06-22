from datasets import load_dataset
import os

print("KSS 100개만 다운로드 중...")
ds = load_dataset("Bingsu/KSS_Dataset", split="train[:100]")

print(f"샘플 수: {len(ds)}")

save_path = "./data/kss"
os.makedirs(save_path, exist_ok=True)
ds.save_to_disk(save_path)

print(f"저장 완료!")