from __future__ import annotations
import asyncio
import re
import threading
import time
from typing import Optional
from config import config
from core.event_bus import EventBus
from domains.soul.soul_container import SoulContainer, SoulConfig
from domains.soul.memory import MemoryManager
from core.context_builder import build_messages
from core.tokenizer import count_tokens


MIN_SENTENCE_LEN = 7


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

    def trigger_prefill(self) -> None:
        """VAD 'start' 콜백 — 별도 스레드에서 system+memory prefill 던짐.
        sync 컨텍스트(recorder의 sync 스레드)에서 호출됨.
        """
        if self._loop is None:
            print("[Prefill] loop 미설정 — 스킵")
            return
        if self._prefill_in_flight:
            return
        self._prefill_in_flight = True

        def _run():
            try:
                # build_messages는 async (DB I/O) — main loop에서 실행
                fut = asyncio.run_coroutine_threadsafe(
                    build_messages(
                        system_prompt=self.soul.build_system_prompt(),
                        memory=self.memory,
                        user_text="",  # prefill은 user_text 자리 비움
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

        t_build = time.perf_counter()
        prompt_tok = sum(count_tokens(m["content"]) for m in messages)
        print(f"[Timing] build_messages: {(t_build - t_stt_end)*1000:.0f}ms, "
              f"prompt ~{prompt_tok}tok, messages={len(messages)}")

        full_response: list[str] = []
        pending = ""
        t_first_tts: float | None = None
        loop = asyncio.get_event_loop()

        self.tts.start_workers()

        def _clean_text(text: str) -> str:
            """이모지, 특수문자 제거."""
            text = re.sub(r'[^\w\s\.,!?~\-가-힣a-zA-Z]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def on_sentence(sentence: str):
            nonlocal pending, t_first_tts
            clean = self.soul.parse_response(sentence)
            clean = _clean_text(clean)
            if not clean.strip():
                return
            pending += clean
            if len(pending) >= MIN_SENTENCE_LEN:
                print(f"[Pipeline] → TTS: {pending}")
                full_response.append(pending)
                self.tts.enqueue(pending)
                if t_first_tts is None:
                    t_first_tts = time.perf_counter()
                    print(f"[Timing] 첫 음성 enqueue: {(t_first_tts - t_start)*1000:.0f}ms (전체 누적)")
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

        t_end = time.perf_counter()
        print(f"[Timing] 전체: {(t_end - t_start)*1000:.0f}ms")

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
