from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field


class Emotion(Enum):
    NEUTRAL   = "neutral"
    HAPPY     = "happy"
    SAD       = "sad"
    ANGRY     = "angry"
    SURPRISED = "surprised"
    SHY       = "shy"
    THINKING  = "thinking"

    @classmethod
    def from_str(cls, name: str) -> "Emotion":
        n = (name or "").strip().lower()
        for e in cls:
            if e.value == n:
                return e
        return cls.NEUTRAL


@dataclass
class EmotionState:
    current: Emotion       = Emotion.NEUTRAL
    history: list[Emotion] = field(default_factory=list)

    def update(self, emotion: Emotion) -> None:
        self.current = emotion
        self.history.append(emotion)
