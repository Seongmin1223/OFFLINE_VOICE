import os
import sys

# 💉 윈도우 환경 충돌 및 인코딩 에러 방지 백신
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUTF8'] = '1'

# 윈도우 한글 환경 콘솔 인코딩 강제 UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print("--- 🚀 포비 뇌수술 파이프라인 가동 ---")

# =================================================================
RANK_LIST = [8, 16, 32]
# =================================================================

model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DATA_PATH = "D:/shin/종프2/poby_train_new.jsonl"
OUTPUT_BASE = "D:/shin/종프2"

# ─────────────────────────────────────────────
# 모델 & 토크나이저 로드
# ─────────────────────────────────────────────
print("⏳ 원본 EXAONE 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# 🚨 EXAONE 전용 패치
def get_input_embeddings_for_exaone(self):
    return self.transformer.wte
base_model.get_input_embeddings = get_input_embeddings_for_exaone.__get__(base_model)
base_model = prepare_model_for_kbit_training(base_model)

# ─────────────────────────────────────────────
# 데이터셋 로드
# 구조: {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
# ─────────────────────────────────────────────
print("📖 학습 데이터셋 로드 중...")
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"✅ 데이터셋 로드 완료: {len(raw_dataset)}개 샘플")

# ─────────────────────────────────────────────
# 채팅 포맷 → 토크나이징
# apply_chat_template()으로 messages 리스트를 
# 모델이 이해하는 단일 문자열로 변환 후 토크나이징
# ─────────────────────────────────────────────
print("🔤 채팅 템플릿 적용 및 토크나이징 중...")

def tokenize_chat(example):
    # messages 리스트를 EXAONE 채팅 템플릿에 맞게 변환
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,          # 일단 문자열로만 변환
        add_generation_prompt=False
    )
    result = tokenizer(
        text,
        truncation=True,
        max_length=1024,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

dataset = raw_dataset.map(
    tokenize_chat,
    remove_columns=raw_dataset.column_names,
    desc="Tokenizing"
)
print(f"✅ 토크나이징 완료\n")

# ─────────────────────────────────────────────
# Rank별 순차 학습 루프
# ─────────────────────────────────────────────
for CURRENT_RANK in RANK_LIST:
    print(f"\n{'='*50}")
    print(f"🔥 [Rank {CURRENT_RANK}] 파인튜닝 시작!")
    print(f"{'='*50}\n")

    output_path = f"{OUTPUT_BASE}/poby_adapters_r{CURRENT_RANK}"

    peft_config = LoraConfig(
        r=CURRENT_RANK,
        lora_alpha=CURRENT_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=512,
        # 이미 토크나이징된 데이터셋이므로 전처리 스킵
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\n✅ Rank {CURRENT_RANK} 어댑터 저장 완료: {output_path}")

    del peft_model, trainer, training_args, peft_config
    torch.cuda.empty_cache()
    print(f"🧹 GPU 캐시 정리 완료. 다음 Rank로 이동합니다...\n")

print("\n🎉🎉🎉 모든 Rank (8, 16, 32) 학습 완료! 🎉🎉🎉")
print("저장 경로:")
for r in RANK_LIST:
    print(f"  - Rank {r:2d}: {OUTPUT_BASE}/poby_adapters_r{r}")