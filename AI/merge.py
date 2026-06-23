import os
import sys

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
OUTPUT_BASE = "D:/shin/종프2"

for rank in [8, 16, 32]:
    print(f"\n🔀 Rank {rank} 병합 중...")

    # FP16으로 베이스 모델 로드 (양자화 없이)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,          # torch_dtype → dtype 으로 수정
        trust_remote_code=True,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 🚨 EXAONE 전용 패치 (merge.py에서도 반드시 필요!)
    def get_input_embeddings_for_exaone(self):
        return self.transformer.wte
    base_model.get_input_embeddings = get_input_embeddings_for_exaone.__get__(base_model)

    # 어댑터 장착 후 베이스 모델에 흡수
    adapter_path = f"{OUTPUT_BASE}/poby_adapters_r{rank}"
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    # 완성된 FP16 모델 저장
    save_path = f"{OUTPUT_BASE}/poby_merged_r{rank}"
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Rank {rank} 저장 완료: {save_path}")

    del base_model, model, merged_model
    print(f"🧹 메모리 정리 완료")

print("\n🎉 모든 Rank 병합 완료!")
print("다음 단계: llama.cpp로 GGUF 변환")