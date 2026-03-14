"""
Avatar Bridge
-------------
감정 상태를 Live2D / VRM 캐릭터로 전달하는 브릿지입니다.
WebSocket으로 프론트엔드(브라우저)에 감정 파라미터를 실시간 전송합니다.

프론트엔드 연동:
  - Live2D Cubism SDK (웹)
  - three-vrm (Three.js 기반 VRM)
"""

from __future__ import annotations
import json
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

from domains.soul.emotion import Emotion, EmotionState


class AvatarBridge:
    """
    감정 변화를 연결된 WebSocket 클라이언트들에게 브로드캐스트합니다.
    프론트엔드에서 이 메시지를 받아 Live2D / VRM 표정을 업데이트합니다.
    """

    def __init__(self):
        self._clients: list["WebSocket"] = []

    # ── 클라이언트 관리 ────────────────────────────────────

    def connect(self, ws: "WebSocket") -> None:
        self._clients.append(ws)

    def disconnect(self, ws: "WebSocket") -> None:
        self._clients = [c for c in self._clients if c is not ws]

    # ── 감정 전송 ──────────────────────────────────────────

    async def send_emotion(self, emotion: Emotion,
                           text: str = "") -> None:
        """감정 변화를 모든 클라이언트에 전송합니다."""
        payload = json.dumps({
            "type":    "emotion",
            "emotion": emotion.value,
            "live2d":  emotion.to_live2d_params(),
            "vrm":     emotion.to_vrm_blendshape(),
            "text":    text,        # 현재 발화 텍스트 (립싱크용)
        }, ensure_ascii=False)

        disconnected = []
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    async def send_speaking(self, is_speaking: bool,
                            text: str = "") -> None:
        """TTS 재생 시작/종료 알림 (립싱크 트리거)."""
        payload = json.dumps({
            "type":       "speaking",
            "is_speaking": is_speaking,
            "text":       text,
        }, ensure_ascii=False)

        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    async def send_state(self, emotion_state: EmotionState,
                         soul_name: str = "") -> None:
        """전체 상태 스냅샷 전송."""
        payload = json.dumps({
            "type":  "state",
            "name":  soul_name,
            **emotion_state.to_dict(),
        }, ensure_ascii=False)

        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    @property
    def client_count(self) -> int:
        return len(self._clients)


# ── 싱글턴 ────────────────────────────────────────────────
avatar_bridge = AvatarBridge()