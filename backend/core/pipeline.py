from __future__ import annotations
import asyncio
import re
from config import config
from core.event_bus import EventBus
from domains.soul.soul_container import SoulContainer, SoulConfig
from domains.soul.memory import MemoryManager
from core.context_builder import build_messages
from core.tokenizer import count_tokens


MIN_SENTENCE_LEN = 10


class VoicePipeline:

    def __init__(self, stt, llm, tts,
                 event_bus=None, soul=None, memory=None):
        self.stt       = stt
        self.llm       = llm
        self.tts       = tts
        self.event_bus = event_bus or EventBus()
        self.soul      = soul   or SoulContainer(SoulConfig.from_preset("pobi"))
        self.memory    = memory or MemoryManager()

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
            return {"user_text": "", "ai_text": ""}

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

        self.tts.start_workers()

        def _clean_text(text: str) -> str:
            """이모지, 특수문자 제거."""
            text = re.sub(r'[^\w\s\.,!?~\-가-힣a-zA-Z]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def on_sentence(sentence: str):
            nonlocal pending
            clean = self.soul.parse_response(sentence)
            clean = _clean_text(clean)
            if not clean.strip():
                return
            pending += clean
            if len(pending) >= MIN_SENTENCE_LEN:
                print(f"[Pipeline] → TTS: {pending}")
                full_response.append(pending)
                self.tts.enqueue(pending)
                pending = ""

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

        ai_text = " ".join(full_response)

        await self.memory.add_turn("user",      user_text, token_count=count_tokens(user_text))
        await self.memory.add_turn("assistant", ai_text,   token_count=count_tokens(ai_text))

        result = {
            "user_text": user_text,
            "ai_text":   ai_text,
            "turn":      self.memory.turn_count,
        }
        await self.event_bus.publish("turn_complete", result)
        return result
