from __future__ import annotations
import asyncio
import re
import random
import threading
import time
from typing import Optional

from config import config
from core.event_bus import EventBus
from core.context_builder import build_messages
from core.tokenizer import count_tokens
from domains.soul.soul_container import SoulContainer, SoulConfig
from domains.soul.memory import MemoryManager
from domains.soul.emotion import Emotion
from api.avatar_ws import broadcast


MIN_SENTENCE_LEN = 25

_THINKING_FILLERS = [
    "음...",
    "어...",
    "흠...",
    "잠깐만.",
    "음, 잠깐만.",
    "어, 생각해볼게.",
]


class VoicePipeline:

    def __init__(self, stt, llm, tts,
                 event_bus=None, soul=None, memory=None):
        self.stt       = stt
        self.llm       = llm
        self.tts       = tts
        self.event_bus = event_bus or EventBus()
        self.soul      = soul   or SoulContainer(SoulConfig.from_preset("pobi"))
        self.memory    = memory or MemoryManager()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._prefill_in_flight = False

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """main event loop reference. mic_loop 시작 시 1회 호출."""
        self._loop = loop
        if hasattr(self.tts, "set_loop"):
            self.tts.set_loop(loop)

    def trigger_prefill(self) -> None:
        """VAD 'start' 콜백 — 별도 스레드에서 system+memory prefill 던짐."""
        if self._loop is None:
            print("[Prefill] loop 미설정 — 스킵")
            return
        if self._prefill_in_flight:
            return
        self._prefill_in_flight = True

        def _run():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    build_messages(
                        system_prompt=self.soul.build_system_prompt(),
                        memory=self.memory,
                        user_text="",
                        token_budget=config.LLM_CONTEXT_SIZE - config.LLM_MAX_TOKENS,
                    ),
                    self._loop,
                )
                messages = fut.result(timeout=10)
                t0 = time.perf_counter()
                elapsed = self.llm.prefill_sync(messages)
                trigger_to_done = (time.perf_counter() - t0) * 1000
                print(f"[Prefill] KV cache 채움 {elapsed:.0f}ms "
                      f"(trigger→완료 {trigger_to_done:.0f}ms)")
            except Exception as e:
                print(f"[Prefill] 오류: {e}")
            finally:
                self._prefill_in_flight = False

        threading.Thread(target=_run, daemon=True).start()

    async def run(self, audio_path: str) -> dict:
        # loop 캐시 (set_loop 미호출 대비)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            if hasattr(self.tts, "set_loop"):
                self.tts.set_loop(self._loop)

        t_start = time.perf_counter()

        # ── 1. STT ──────────────────────────────────────────
        print("\n[Pipeline] ▶ 1/3 음성 인식(STT)")
        try:
            user_text = await self.stt.transcribe(audio_path)
        except Exception as e:
            await self.event_bus.publish("error", {"stage": "stt", "error": str(e)})
            raise

        t_stt_end = time.perf_counter()
        print(f"[Timing] STT: {(t_stt_end - t_start)*1000:.0f}ms")

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

        t_build = time.perf_counter()
        prompt_tok = sum(count_tokens(m["content"]) for m in messages)
        print(f"[Timing] build_messages: {(t_build - t_stt_end)*1000:.0f}ms, "
              f"prompt ~{prompt_tok}tok, messages={len(messages)}")

        full_response: list[str] = []
        pending = ""
        t_first_tts: float | None = None
        last_emotion: Optional[Emotion] = None
        loop = asyncio.get_event_loop()

        self.tts.start_workers()

        # 필러 한 번 — LLM 응답 기다리는 동안의 정적 메우기
        filler = random.choice(_THINKING_FILLERS)
        print(f"[Pipeline] → TTS 필러: {filler}")
        self.tts.enqueue(filler)

        # 말하기 시작 신호
        await broadcast({"type": "speaking", "speaking": True})

        def _clean_text(text: str) -> str:
            text = re.sub(r'[^\w\s\.,!?~\-가-힣a-zA-Z]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def on_sentence(sentence: str):
            nonlocal pending, t_first_tts, last_emotion
            clean, emotion = self.soul.parse_response(sentence)
            clean = _clean_text(clean)
            if not clean.strip():
                return
            last_emotion = emotion
            pending += clean
            if len(pending) >= MIN_SENTENCE_LEN:
                print(f"[Pipeline] → TTS: {pending}")
                full_response.append(pending)
                self.tts.enqueue(pending, emotion.value)
                if t_first_tts is None:
                    t_first_tts = time.perf_counter()
                    print(f"[Timing] 첫 음성 enqueue: {(t_first_tts - t_start)*1000:.0f}ms")
                # 감정/말풍선 페이지로 전송 (thread-safe)
                asyncio.run_coroutine_threadsafe(
                    broadcast({"type": "emotion", "emotion": emotion.value, "text": pending}),
                    self._loop,
                )
                pending = ""

        await loop.run_in_executor(
            None, self.llm.stream_sync, messages, on_sentence
        )

        # 남은 pending 처리
        if pending.strip():
            print(f"[Pipeline] → TTS (잔여): {pending}")
            full_response.append(pending)
            emo_value = (last_emotion or Emotion.NEUTRAL).value
            self.tts.enqueue(pending, emo_value)

        # 재생 완료 대기
        self.tts.wait_done()

        # 말하기 종료 신호
        await broadcast({"type": "speaking", "speaking": False})

        t_end = time.perf_counter()
        print(f"[Timing] 전체: {(t_end - t_start)*1000:.0f}ms")

        ai_text = " ".join(full_response)
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
