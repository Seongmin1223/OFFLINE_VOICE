"""
Soul Container
--------------
캐릭터의 정체성(성격, 말투, 이름)과 실시간 감정 상태를 관리합니다.
LLM 호출 전 시스템 프롬프트를 동적으로 구성하고,
LLM 응답에서 감정 태그를 파싱해 emotion_state를 갱신합니다.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from domains.soul.emotion import Emotion, EmotionState


# ── 성격 프리셋 ────────────────────────────────────────────

PERSONALITY_PRESETS: dict[str, dict] = {
    "airi": {
        "name": "아이리",
        "age": "17",
        "tone": "밝고 친근하며 가끔 수줍어하는 말투. 문장 끝에 '~요', '~네요'를 자주 씁니다.",
        "traits": ["호기심 많음", "긍정적", "수줍음", "친절함"],
        "speech_style": "반말과 존댓말을 섞어 쓰며 이모티콘 없이 자연스럽게 표현합니다.",
        "forbidden": ["욕설", "폭력적 표현", "냉담한 거절"],
    },
    "assistant": {
        "name": "어시스턴트",
        "age": None,
        "tone": "차분하고 명확한 말투.",
        "traits": ["논리적", "친절함", "간결함"],
        "speech_style": "짧고 명확하게 답변합니다.",
        "forbidden": [],
    },
}


@dataclass
class SoulConfig:
    """캐릭터 정의 데이터."""
    name:         str       = "아이리"
    age:          str | None = "17"
    tone:         str       = "밝고 친근한 말투"
    traits:       list[str] = field(default_factory=lambda: ["친절함", "호기심 많음"])
    speech_style: str       = "자연스럽고 따뜻하게 대화합니다."
    forbidden:    list[str] = field(default_factory=list)

    @classmethod
    def from_preset(cls, preset_name: str) -> "SoulConfig":
        data = PERSONALITY_PRESETS.get(preset_name, PERSONALITY_PRESETS["airi"])
        return cls(**{k: v for k, v in data.items() if v is not None})

    @classmethod
    def from_file(cls, path: str) -> "SoulConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


class SoulContainer:
    """
    캐릭터 Soul을 관리하는 핵심 컨테이너.

    사용 예:
        soul = SoulContainer(SoulConfig.from_preset("airi"))
        system_prompt = soul.build_system_prompt()
        # LLM 응답 후
        clean_text, emotion = soul.parse_response(llm_output)
    """

    def __init__(self, config: SoulConfig | None = None):
        self.config  = config or SoulConfig.from_preset("airi")
        self.emotion = EmotionState()

    # ── 시스템 프롬프트 생성 ───────────────────────────────

    def build_system_prompt(self) -> str:
        cfg = self.config
        traits_str = ", ".join(cfg.traits)
        forbidden_str = (
            f"\n절대 하지 말아야 할 것: {', '.join(cfg.forbidden)}"
            if cfg.forbidden else ""
        )
        age_str = f"나이: {cfg.age}세\n" if cfg.age else ""

        return f"""당신은 {cfg.name}입니다.
{age_str}성격: {traits_str}
말투: {cfg.tone}
스타일: {cfg.speech_style}{forbidden_str}

현재 감정 상태: {self.emotion.current.value}

응답할 때 반드시 다음 형식을 사용하세요:
[EMOTION:감정이름] 응답 텍스트

감정 이름 목록: neutral, happy, sad, angry, surprised, shy, thinking

예시:
[EMOTION:happy] 안녕하세요! 만나서 반가워요~
[EMOTION:shy] 그런 말을 들으니 부끄럽네요...
"""

    # ── LLM 응답 파싱 ──────────────────────────────────────

    def parse_response(self, raw: str) -> tuple[str, Emotion]:
        """
        LLM 응답에서 [EMOTION:xxx] 태그를 파싱합니다.
        Returns:
            (clean_text, detected_emotion)
        """
        pattern = r"\[EMOTION:(\w+)\]"
        match   = re.search(pattern, raw)

        if match:
            emotion_name = match.group(1).lower()
            emotion      = Emotion.from_str(emotion_name)
            clean_text   = re.sub(pattern, "", raw).strip()
        else:
            emotion    = Emotion.NEUTRAL
            clean_text = raw.strip()

        self.emotion.update(emotion)
        return clean_text, emotion

    # ── 상태 조회 ──────────────────────────────────────────

    @property
    def current_emotion(self) -> Emotion:
        return self.emotion.current

    def soul_info(self) -> dict:
        return {
            "name":    self.config.name,
            "emotion": self.emotion.current.value,
            "history": [e.value for e in self.emotion.history[-5:]],
        }