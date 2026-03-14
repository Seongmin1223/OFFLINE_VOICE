"""
감정 상태 정의 및 트래킹.
"""

from __future__ import annotations
from enum import Enum
from collections import deque


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
        try:
            return cls(name.lower())
        except ValueError:
            return cls.NEUTRAL

    # Live2D / VRM 표정 파라미터 매핑
    def to_live2d_params(self) -> dict[str, float]:
        mapping = {
            Emotion.NEUTRAL:   {"ParamEyeOpen": 1.0, "ParamMouthOpenY": 0.0, "ParamBrowLY": 0.0},
            Emotion.HAPPY:     {"ParamEyeOpen": 0.6, "ParamMouthOpenY": 0.8, "ParamBrowLY": 0.3},
            Emotion.SAD:       {"ParamEyeOpen": 0.7, "ParamMouthOpenY": 0.1, "ParamBrowLY": -0.5},
            Emotion.ANGRY:     {"ParamEyeOpen": 1.0, "ParamMouthOpenY": 0.3, "ParamBrowLY": -0.8},
            Emotion.SURPRISED: {"ParamEyeOpen": 1.5, "ParamMouthOpenY": 0.6, "ParamBrowLY": 0.8},
            Emotion.SHY:       {"ParamEyeOpen": 0.5, "ParamMouthOpenY": 0.2, "ParamBrowLY": 0.2},
            Emotion.THINKING:  {"ParamEyeOpen": 0.8, "ParamMouthOpenY": 0.0, "ParamBrowLY": 0.1},
        }
        return mapping.get(self, mapping[Emotion.NEUTRAL])

    # VRM BlendShape 매핑
    def to_vrm_blendshape(self) -> str:
        mapping = {
            Emotion.NEUTRAL:   "Neutral",
            Emotion.HAPPY:     "Joy",
            Emotion.SAD:       "Sorrow",
            Emotion.ANGRY:     "Angry",
            Emotion.SURPRISED: "Surprised",
            Emotion.SHY:       "Joy",      # VRM에 Shy 없으면 Joy로 대체
            Emotion.THINKING:  "Neutral",
        }
        return mapping.get(self, "Neutral")


class EmotionState:
    """실시간 감정 상태 + 히스토리 관리."""

    def __init__(self, max_history: int = 20):
        self.current: Emotion          = Emotion.NEUTRAL
        self.history: deque[Emotion]   = deque(maxlen=max_history)
        self.counts:  dict[Emotion, int] = {e: 0 for e in Emotion}

    def update(self, emotion: Emotion) -> None:
        self.history.append(self.current)
        self.current = emotion
        self.counts[emotion] += 1

    def dominant_emotion(self) -> Emotion:
        """대화 전체에서 가장 많이 나온 감정."""
        return max(self.counts, key=lambda e: self.counts[e])

    def to_dict(self) -> dict:
        return {
            "current":  self.current.value,
            "dominant": self.dominant_emotion().value,
            "counts":   {e.value: c for e, c in self.counts.items()},
            "live2d":   self.current.to_live2d_params(),
            "vrm":      self.current.to_vrm_blendshape(),
        }