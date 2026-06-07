from __future__ import annotations
import re
from dataclasses import dataclass, field

from domains.soul.emotion import Emotion, EmotionState


PERSONALITY_PRESETS: dict[str, dict] = {
    "pobi": {
        "name": "포비",
        "age": "5",
        "tone": "항상 다정하고 따뜻한 곰돌이 인형의 말투.",
        "traits": ["다정함", "친절함", "아이들의 친구", "순수함"],
        "speech_style": "5살 아이들의 눈높이에 맞춰서 100% 반말로 대답해.",
        "forbidden": ["존댓말", "~요", "~습니다", "어려운 단어", "욕설", "거친 표현"],
    }
}


@dataclass
class SoulConfig:
    name:         str        = "포비"
    age:          str | None = "5"
    tone:         str        = "다정하고 따뜻한 말투"
    traits:       list[str]  = field(default_factory=lambda: ["친절함", "따뜻함"])
    speech_style: str        = "반말로 대답해."
    forbidden:    list[str]  = field(default_factory=list)

    @classmethod
    def from_preset(cls, preset_name: str) -> "SoulConfig":
        data = PERSONALITY_PRESETS.get(preset_name, PERSONALITY_PRESETS["pobi"])
        return cls(**{k: v for k, v in data.items() if v is not None})


class SoulContainer:
    def __init__(self, config: SoulConfig | None = None):
        self.config  = config or SoulConfig.from_preset("pobi")
        self.emotion = EmotionState()

    def build_system_prompt(self) -> str:
        cfg = self.config
        traits_str = ", ".join(cfg.traits)
        forbidden_str = (
            f"\n절대 하지 말아야 할 것: {', '.join(cfg.forbidden)}"
            if cfg.forbidden else ""
        )
        age_str = f"나이: {cfg.age}세\n" if cfg.age else ""

        return f"""너는 5살 아이들의 다정한 친구, 귀여운 곰돌이 인형 '{cfg.name}'야.
{age_str}성격: {traits_str}
말투: {cfg.tone}
스타일: {cfg.speech_style}{forbidden_str}

현재 감정: {self.emotion.current.value}

응답할 때 반드시 다음 형식을 사용하세요:
[EMOTION:감정이름] 응답 텍스트

감정 이름 목록: neutral, happy, sad, angry, surprised, shy, thinking
"""

    def parse_response(self, raw: str) -> tuple[str, Emotion]:
        pattern = r"\[EMOTION:(\w+)\]"
        match = re.search(pattern, raw)
        if match:
            emotion = Emotion.from_str(match.group(1))
            clean   = re.sub(pattern, "", raw).strip()
        else:
            emotion = Emotion.NEUTRAL
            clean   = raw.strip()
        self.emotion.update(emotion)
        return clean, emotion

    @property
    def current_emotion(self) -> Emotion:
        return self.emotion.current

    def soul_info(self) -> dict:
        return {
            "name":    self.config.name,
            "emotion": self.emotion.current.value,
        }
