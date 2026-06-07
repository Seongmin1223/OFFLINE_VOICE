from __future__ import annotations
from api.avatar_ws import broadcast
import asyncio
import re
import random
from config import config
from core.event_bus import EventBus
from domains.soul.soul_container import SoulContainer, SoulConfig
from domains.soul.memory import MemoryManager
from core.context_builder import build_messages
from core.tokenizer import count_tokens
from domains.soul.avatar_bridge import AvatarBridge


MIN_SENTENCE_LEN = 25


class VoicePipeline:
    _THINKING_FILLERS = [
        "음...",
        "어...",
        "흠...",
        "잠깐만요.",
        "음, 잠깐만요.",
        "어, 생각해볼게요.",
    ]

    def __init__(self, stt, llm, tts,
                 event_bus=None, soul=None, memory=None, avatar_bridge=None):
        self.stt       = stt
        self.llm       = llm
        self.tts       = tts
        self.event_bus = event_bus or EventBus()
        self.soul      = soul   or SoulContainer(SoulConfig.from_preset("airi"))
        self.memory    = memory or MemoryManager()
        self.avatar    = avatar_bridge

    async def run(self, audio_path: str) -> dict:

        # ── 1. STT ──────────────────────────────────────────
        print("\n[Pipeline] ▶ 1/3 음성 인식(STT)")
        try:
            user_text = await self.stt.transcribe(audio_path)
        except Exception as e:
            await self.event_bus.publish("error", {"stage": "stt", "error": str(e)})
            raise

        if not user_text.strip():
            print("[Pipeline] ⚠ 인식된 텍스트가 없습니다.")
            return {"user_text": "", "ai_text": "", "emotion": "neutral"}

        await self.event_bus.publish("stt_complete", {"text": user_text})

        # ── 2+3. LLM 스트리밍 + TTS 오버랩 ──────────────────
        print("[Pipeline] ▶ 2+3/3 LLM 스트리밍 + TTS 오버랩 재생")

        messages = await build_messages(
            system_prompt=self.soul.build_system_prompt(),
            memory=self.memory,
            user_text=user_text,
            token_budget=config.LLM_CONTEXT_SIZE - config.LLM_MAX_TOKENS,
        )

        full_response = []
        pending = ""
        loop = asyncio.get_event_loop()
        last_emotion = None

        self.tts.start_workers()
        # 비쥬얼라이저 broadcast용 이벤트 루프 등록
        if hasattr(self.tts, "set_loop"):
            self.tts.set_loop(loop)
        def _enqueue_filler():
            filler = random.choice(self._THINKING_FILLERS)
            print(f"[Pipeline] → TTS 필러: {filler}")
            self.tts.enqueue(filler)

        _enqueue_filler()
        def _clean_text(text: str) -> str:
            """이모지, 특수문자 제거."""
            # 이모지 제거
            text = re.sub(r'[^\w\s\.,!?~\-가-힣a-zA-Z]', '', text)
            # 연속 공백 정리
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def on_sentence(sentence: str):
            nonlocal pending, last_emotion
            clean, emotion = self.soul.parse_response(sentence)
            clean = _clean_text(clean)
            if not clean.strip():
                return
            last_emotion = emotion
            pending += clean
            if len(pending) >= MIN_SENTENCE_LEN:
                print(f"[Pipeline] → TTS: {pending}")
                full_response.append(pending)
                self.tts.enqueue(pending)
                # 감정 신호 전송
                asyncio.run_coroutine_threadsafe(
                    broadcast({"type": "emotion", "emotion": emotion.value}),
                    loop
                )
                pending = ""

        # 말하기 시작 신호
        await broadcast({"type": "speaking", "speaking": True})

        await loop.run_in_executor(
            None, self.llm.stream_sync, messages, on_sentence
        )

        # 남은 pending 처리
        if pending.strip():
            print(f"[Pipeline] → TTS (잔여): {pending}")
            full_response.append(pending)
            self.tts.enqueue(pending)

        # 재생 완료 대기
        self.tts.wait_done()

        # 말하기 종료 신호
        await broadcast({"type": "speaking", "speaking": False})

        ai_text = " ".join(full_response)
        _, emotion = self.soul.parse_response(ai_text) if ai_text else (None, None)

        from domains.soul.emotion import Emotion
        if emotion is None:
            emotion = last_emotion or Emotion.NEUTRAL

        await self.memory.add_turn("user",      user_text, token_count=count_tokens(user_text))
        await self.memory.add_turn("assistant", ai_text,   token_count=count_tokens(ai_text))

        result = {
            "user_text": user_text,
            "ai_text":   ai_text,
            "emotion":   emotion.value,
            "turn":      self.memory.turn_count,
        }
        await self.event_bus.publish("turn_complete", result)
        return result