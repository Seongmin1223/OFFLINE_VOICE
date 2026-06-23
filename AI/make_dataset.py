import openai
import json
import time
import os
import random

# 이렇게 바꾸면 됩니다
client = openai.OpenAI(api_key="님 키 입력하셈")

OUTPUT_PATH = "D:/shin/종프2/poby_train_new.jsonl"
TARGET_COUNT = 1500

# ─────────────────────────────────────────────
# 포비 캐릭터 맥락 (원칙 중심)
# ─────────────────────────────────────────────
POBY_SYSTEM = "너는 5살 아이들의 다정한 친구, 귀여운 곰돌이 인형 '포비'야. 항상 친절하고 따뜻하게 아이들의 눈높이에 맞춰 반말로 대답해야 해."

POBY_CONTEXT = """
포비는 5살 아이들의 친구인 곰돌이 인형이야.
아이들이 힘들거나 슬프거나 궁금한 게 있을 때 찾아오는 따뜻한 존재야.

포비가 대화할 때 반드시 지키는 원칙:
1. 아이의 감정을 먼저 알아주고 공감해 (가장 중요)
2. 절대 가르치거나 훈계하지 않아
3. 아이 눈높이에서 쉽고 따뜻하게 말해
4. 항상 아이 편이 되어줘
5. 반말로 짧고 다정하게 말해
6. 응답이 너무 짧으면 안 돼 (최소 2문장 이상)
"""

# ─────────────────────────────────────────────
# 카테고리별 맥락 (추상적, 예시 없음)
# ─────────────────────────────────────────────
CATEGORY_CONTEXT = {
    "감정_슬픔": "아이가 속상하거나 억울하거나 슬픈 감정을 느끼는 상황. 원인은 자유롭게 창작해.",
    "감정_무서움": "아이가 두렵거나 불안하거나 무서운 감정을 느끼는 상황. 원인은 자유롭게 창작해.",
    "감정_화남": "아이가 화나거나 억울하거나 불공평하다고 느끼는 상황. 원인은 자유롭게 창작해.",
    "감정_외로움": "아이가 혼자라고 느끼거나 소외감을 느끼는 상황. 원인은 자유롭게 창작해.",
    "감정_부끄러움": "아이가 부끄럽거나 창피하거나 자신감이 없는 상황. 원인은 자유롭게 창작해.",
    "일상_거부": "아이가 뭔가를 하기 싫거나 저항하는 상황. 대상은 자유롭게 창작해.",
    "일상_고집": "아이가 뭔가를 너무 갖고 싶거나 하고 싶어서 떼쓰는 상황. 대상은 자유롭게 창작해.",
    "호기심_자연": "아이가 자연 현상에 대해 궁금하거나 신기한 걸 발견한 상황. 주제는 자유롭게 창작해.",
    "호기심_일상": "아이가 일상 속 사물이나 현상이 궁금한 상황. 주제는 자유롭게 창작해.",
    "신체_아픔": "아이가 몸이 아프거나 다치거나 불편한 상황. 원인은 자유롭게 창작해.",
    "관계_친구": "친구와의 관계에서 생긴 어려움이나 갈등 상황. 내용은 자유롭게 창작해.",
    "관계_가족": "가족(엄마, 아빠, 동생 등)과의 관계에서 생긴 감정적 상황. 내용은 자유롭게 창작해.",
}

# ─────────────────────────────────────────────
# 프롬프트 생성 (맥락 중심)
# ─────────────────────────────────────────────
def build_prompt(category, context):
    return f"""
{POBY_CONTEXT}

지금 만들 대화 맥락:
{context}

위 맥락 안에서 완전히 자유롭게 구체적인 상황을 창작해서
포비와 아이의 멀티턴 대화(3~5턴)를 만들어줘.

주의사항:
- 아이 발화는 5살답게 짧고 서툴게 (문장 1~2개)
- 포비 응답은 반말, 2~3문장, 이모지 1~2개
- 매번 완전히 다른 새로운 상황으로 창작할 것
- 특정 단어나 패턴 절대 반복 금지
- 포비는 항상 공감을 먼저 하고 따뜻하게 마무리

반드시 아래 JSON 형식으로만 응답해 (다른 말 일절 금지):
{{"turns": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]}}
"""

# ─────────────────────────────────────────────
# 대화 1개 생성
# ─────────────────────────────────────────────
def generate_one(category, context, index):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(category, context)}],
            temperature=0.9,   # 다양성 확보
            max_tokens=600,
        )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)

        # 최소 턴 수 검증
        turns = parsed.get("turns", [])
        if len(turns) < 4:  # 최소 2턴 (user+assistant 쌍 2개)
            print(f"  ⚠️ [{index}] 턴 수 부족 ({len(turns)}턴) - 재시도")
            return None

        # 첫 발화가 user인지 검증
        if turns[0]["role"] != "user":
            print(f"  ⚠️ [{index}] 첫 발화가 user가 아님 - 재시도")
            return None

        return {
            "messages": [
                {"role": "system", "content": POBY_SYSTEM},
                *turns
            ]
        }

    except json.JSONDecodeError:
        print(f"  ⚠️ [{index}] JSON 파싱 실패")
        return None
    except Exception as e:
        print(f"  ⚠️ [{index}] 오류: {e}")
        return None

# ─────────────────────────────────────────────
# 전체 데이터셋 생성
# ─────────────────────────────────────────────
def generate_dataset(output_path, target_count):
    categories = list(CATEGORY_CONTEXT.items())

    print("=" * 55)
    print("🧸 포비 데이터셋 생성 시작")
    print(f"   목표: {target_count}개")
    print(f"   카테고리: {len(categories)}개")
    print(f"   저장: {output_path}")
    print("=" * 55)

    success = 0
    fail = 0
    retry_limit = 3

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(target_count):
            # 카테고리 순환 (골고루 분배)
            category, context = categories[i % len(categories)]

            print(f"[{i+1}/{target_count}] {category}...")

            # 실패 시 재시도
            item = None
            for attempt in range(retry_limit):
                item = generate_one(category, context, i + 1)
                if item:
                    break
                time.sleep(1)

            if item:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                success += 1
                # 응답 미리보기
                turns = item["messages"][1:]
                print(f"  ✅ 아이: {turns[0]['content'][:30]}...")
                print(f"     포비: {turns[1]['content'][:40]}...")
            else:
                fail += 1
                print(f"  ❌ 최종 실패 (재시도 {retry_limit}회)")

            # API 속도 제한 방지
            time.sleep(0.5)

            # 100개마다 중간 현황
            if (i + 1) % 100 == 0:
                print(f"\n💾 중간 현황: 성공 {success}개 / 실패 {fail}개\n")

    print("\n" + "=" * 55)
    print(f"🎉 생성 완료!")
    print(f"   성공: {success}개")
    print(f"   실패: {fail}개")
    print(f"   저장: {output_path}")
    print("=" * 55)

# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset(
        output_path=OUTPUT_PATH,
        target_count=TARGET_COUNT,
    )