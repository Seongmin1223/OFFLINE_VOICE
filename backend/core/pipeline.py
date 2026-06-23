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
from api.avatar_ws import broadcast


# 첫 음성을 빨리 내보내려고 짧게 — 한 글자만 모여도 바로 TTS로 흘림
MIN_SENTENCE_LEN = 4

_THINKING_FILLERS = [
    "음...",
    "어...",
    "흠...",
    "잠깐만.",
    "음, 잠깐만.",
]


def _clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s\.,!?~\-가-힣a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
        self.last_metrics: dict = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """main event loop reference. mic_loop 시작 시 1회 호출."""
        self._loop = loop
        if hasattr(self.tts, "set_loop"):
            self.tts.set_loop(loop)

    async def _wait_http(self, url: str, label: str, timeout_s: float = 90.0) -> bool:
        """서버 health가 응답할 때까지 대기. 백엔드가 STT/LLM 서버보다 먼저 떠도
        워밍업이 콜드 실패하지 않도록 한다."""
        import requests
        loop = asyncio.get_event_loop()
        deadline = time.perf_counter() + timeout_s

        def _ping() -> bool:
            try:
                return requests.get(url, timeout=2).status_code < 500
            except Exception:
                return False

        while time.perf_counter() < deadline:
            if await loop.run_in_executor(None, _ping):
                return True
            await asyncio.sleep(1.0)
        print(f"[Warmup] {label} 서버 대기 시간 초과: {url}")
        return False

    async def warmup(self) -> None:
        """시작 시 1회: STT/LLM 서버가 뜰 때까지 기다린 뒤 두 모델을 모두 데운다.
        이게 끝나야 프론트엔드 로딩 오버레이가 사라지고 사용자가 쓸 수 있다.
        """
        print("[Warmup] 시스템 워밍업 시작...")
        from api.avatar_ws import set_status, set_models_ready
        try:
            await set_status("warming", "시스템 가동 중...")

            # 1) STT(whisper.cpp) — 서버 대기 후 무음 클립으로 encode 데우기
            if hasattr(self.stt, "warmup"):
                await set_status("warming", "음성 인식 모델 준비 중...")
                await self._wait_http(f"{config.WHISPER_SERVER_URL}/", "STT")
                await self.stt.warmup()

            # 2) LLM(llama.cpp) — 서버 대기 후 시스템 프롬프트 KV cache 사전 적재
            await set_status("warming", "언어 모델 준비 중...")
            await self._wait_http(f"{config.LLM_SERVER_URL}/health", "LLM")
            messages = await build_messages(
                system_prompt=self.soul.build_system_prompt(),
                memory=self.memory,
                user_text="",
                token_budget=config.LLM_CONTEXT_SIZE - config.LLM_MAX_TOKENS,
            )
            loop = asyncio.get_event_loop()
            elapsed = await loop.run_in_executor(None, self.llm.prefill_sync, messages)
            print(f"[Warmup] LLM 사전 발화 완료 ({elapsed:.0f}ms)")

            # 모델 준비 완료 표시 — 마이크까지 준비되면 게이트가 'ready'로 전환.
            await set_models_ready(True)
            print("[Warmup] ✅ 모델 워밍업 완료")
        except Exception as e:
            print(f"[Warmup] 오류: {e}")
            try:
                await set_models_ready(True)
            except Exception:
                pass

    def trigger_prefill(self) -> None:
        """VAD 'start' 콜백 — 별도 스레드에서 system+memory prefill 던짐.
        사용자가 말을 시작하는 순간 KV cache를 미리 채워, 발화가 끝났을 때
        LLM이 바로 토큰을 낼 수 있게 한다.
        """
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
        # main event loop 캐시 (set_loop 미호출 대비)
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
            return {"user_text": "", "ai_text": ""}

        await self.event_bus.publish("stt_complete", {"text": user_text})
        # 프론트엔드(원본 POBY)는 'stt' 메시지로 사용자 발화를 받는다.
        await broadcast({"type": "stt", "text": user_text})

        # ── 2+3. LLM 스트리밍 + TTS 오버랩 재생 ──────────────
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
        speaking_started = False

        self.tts.start_workers()

        # 첫 토큰을 기다리는 동안의 정적을 메우는 짧은 필러
        filler = random.choice(_THINKING_FILLERS)
        print(f"[Pipeline] → TTS 필러: {filler}")
        self.tts.enqueue(filler)
        await broadcast({"type": "speaking", "speaking": True})
        speaking_started = True

        # 콜백(워커 스레드)에서 만든 broadcast 코루틴을 메인 루프에 안전하게 던짐
        def _broadcast_threadsafe(payload: dict):
            try:
                asyncio.run_coroutine_threadsafe(broadcast(payload), self._loop)
            except Exception:
                pass

        def on_sentence(sentence: str):
            nonlocal pending, t_first_tts
            clean = _clean_text(self.soul.parse_response(sentence))
            if not clean.strip():
                return
            pending += clean
            if len(pending) >= MIN_SENTENCE_LEN:
                print(f"[Pipeline] → TTS: {pending}")
                full_response.append(pending)
                self.tts.enqueue(pending)
                _broadcast_threadsafe({"type": "llm", "text": " ".join(full_response)})
                if t_first_tts is None:
                    t_first_tts = time.perf_counter()
                    print(f"[Timing] 첫 음성 enqueue: {(t_first_tts - t_start)*1000:.0f}ms")
                pending = ""

        await self._loop.run_in_executor(
            None, self.llm.stream_sync, messages, on_sentence
        )

        # 남은 pending 처리
        if pending.strip():
            print(f"[Pipeline] → TTS (잔여): {pending}")
            full_response.append(pending)
            self.tts.enqueue(pending)
            await broadcast({"type": "llm", "text": " ".join(full_response)})

        # 재생 완료 대기
        await asyncio.get_running_loop().run_in_executor(None, self.tts.wait_done)

        # 말하기 종료 신호
        if speaking_started:
            await broadcast({"type": "speaking", "speaking": False})

        t_end = time.perf_counter()
        print(f"[Timing] 전체: {(t_end - t_start)*1000:.0f}ms")

        self.last_metrics = {
            "stt_ms":        round((t_stt_end - t_start) * 1000),
            "ttft_ms":       round(getattr(self.llm, "last_ttft_ms", -1)),
            "first_audio_ms": round((t_first_tts - t_start) * 1000) if t_first_tts else -1,
            "total_ms":      round((t_end - t_start) * 1000),
        }
        print(f"[Metrics] {self.last_metrics}")

        ai_text = " ".join(full_response)

        await self.memory.add_turn("user",      user_text, token_count=count_tokens(user_text))
        await self.memory.add_turn("assistant", ai_text,   token_count=count_tokens(ai_text))

        await broadcast({"type": "state", "turn": self.memory.turn_count})

        result = {
            "user_text": user_text,
            "ai_text":   ai_text,
            "turn":      self.memory.turn_count,
        }
        await self.event_bus.publish("turn_complete", result)
        return result
