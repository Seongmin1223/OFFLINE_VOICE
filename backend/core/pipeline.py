from __future__ import annotations
import asyncio
import re
import threading
import time
from typing import Optional

from config import config
from core.event_bus import EventBus
from core.context_builder import build_messages, _summarize_description
from core.case_search import search_cases
from core.grounding import strip_case_tokens, strip_unknown_cases
from core.tokenizer import count_tokens
from database.connection import get_db
import database.repository as repo
from domains.soul.soul_container import SoulContainer, SoulConfig
from domains.soul.memory import MemoryManager
from api.avatar_ws import broadcast


MIN_SENTENCE_LEN = 1
# md [시스템] 예시는 시나리오당 2~4건을 표시 — 가장 강한 시나리오 A(4건)에 맞춤
_CASE_PRESENT_LIMIT = 4

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SINO_DIGIT = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SINO_UNIT  = ["", "십", "백", "천"]
# 월은 6월→유월, 10월→시월 등 특수 발음이 있어 사전으로 처리
_MONTH_KO = {
    1: "일월", 2: "이월", 3: "삼월", 4: "사월", 5: "오월", 6: "유월",
    7: "칠월", 8: "팔월", 9: "구월", 10: "시월", 11: "십일월", 12: "십이월",
}
_CASE_COUNT_KO = {
    1: "한 건",
    2: "두 건",
    3: "세 건",
    4: "네 건",
    5: "다섯 건",
    6: "여섯 건",
    7: "일곱 건",
    8: "여덟 건",
    9: "아홉 건",
    10: "열 건",
}


def _sino_korean(n: int) -> str:
    """정수를 한글 사이시오 수사로 (2025 → '이천이십오', 23 → '이십삼')."""
    if n == 0:
        return "영"
    s = str(n)
    length = len(s)
    parts: list[str] = []
    for i, ch in enumerate(s):
        d = int(ch)
        pos = length - 1 - i
        if d == 0:
            continue
        unit = _SINO_UNIT[pos] if pos < len(_SINO_UNIT) else ""
        if d == 1 and pos >= 1:        # 일십→십, 일천→천 처럼 앞의 '일' 생략
            parts.append(unit)
        else:
            parts.append(_SINO_DIGIT[d] + unit)
    return "".join(parts)


def _date_to_korean(text: str) -> str:
    """TTS가 'YYYY-MM-DD'를 또박또박 읽도록 한글 수사로 변환(음성용).
    예: '2025-09-23' → '이천이십오년 구월 이십삼일'.
    """
    def _repl(m) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        month = _MONTH_KO.get(mo, _sino_korean(mo) + "월")
        return f"{_sino_korean(y)}년 {month} {_sino_korean(d)}일"
    return _DATE_RE.sub(_repl, text)


def _case_count_to_korean(n: int) -> str:
    return _CASE_COUNT_KO.get(n, f"{_sino_korean(n)} 건")


def _present_case_line(case: dict) -> str:
    """검색된 케이스 1건을 화면/음성용 한 줄로 표시. md [시스템] 형식과 동일.
    예: '[case:#0446] 2025-10-21 Secuwatcher — 인천교통공사 망연계 Tomcat 실행'
    """
    summary = _summarize_description(case["description"], max_len=60)
    return f"[case:{case['case_id']}] {case['case_date']} {case['solution']} — {summary}"


def _case_speech_line(case: dict) -> str:
    """TTS용 케이스 요약. 케이스 번호/날짜 메타는 애초에 포함하지 않는다."""
    text = _summarize_description(case["description"], max_len=56)
    if not text:
        return ""

    speech_rewrites = {
        "Map2D": "맵투디",
        "Tomcat": "톰캣",
        "Java": "자바",
        "PDF": "피디에프",
    }
    for src, dst in speech_rewrites.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    text = re.sub(r"\s*[-–—:]\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-–—:")
    return text


class VoicePipeline:

    def __init__(self, stt, llm, tts,
                 event_bus=None, soul=None, memory=None):
        self.stt       = stt
        self.llm       = llm
        self.tts       = tts
        self.event_bus = event_bus or EventBus()
        self.soul      = soul   or SoulContainer(SoulConfig.from_preset("fix_assistant"))
        self.memory    = memory or MemoryManager()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._prefill_in_flight = False

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """main event loop reference. mic_loop 시작 시 1회 호출."""
        self._loop = loop
        if hasattr(self.tts, "set_loop"):
            self.tts.set_loop(loop)

    async def warmup(self) -> None:
        """LLM 모델 페이지 로딩 + 시스템 프롬프트 KV cache 사전 적재.
        서버 시작 직후 1회 호출하면 사용자의 첫 발화도 콜드 페널티 없이 빠르게 응답.
        """
        print("[Warmup] LLM 사전 발화 시작...")
        from api.avatar_ws import set_status
        try:
            await set_status("warming", "시스템 가동 중...")
            messages = await build_messages(
                system_prompt=self.soul.build_system_prompt(),
                memory=self.memory,
                user_text="",
                token_budget=config.LLM_CONTEXT_SIZE - config.LLM_MAX_TOKENS,
            )
            loop = asyncio.get_event_loop()
            elapsed = await loop.run_in_executor(None, self.llm.prefill_sync, messages)
            print(f"[Warmup] LLM 사전 발화 완료 ({elapsed:.0f}ms)")
            await set_status("ready", "준비 완료")
        except Exception as e:
            print(f"[Warmup] 오류: {e}")
            try:
                await set_status("ready", "")
            except Exception:
                pass

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

    async def _synthesize_pattern(self, user_text: str, cases: list[dict]) -> str:
        """검색된 케이스들의 공통 패턴/권장사항을 한 문장으로 합성.
        케이스 목록은 이미 결정적으로 표시되므로, 충돌 없는 전용 미니 프롬프트로
        '패턴 한 줄'만 생성한다. 목록 재요약·케이스 번호 생성을 금지해 환각을 차단.
        """
        case_lines = "\n".join(_present_case_line(c) for c in cases)
        system = (
            "너는 IT 유지보수 장애 이력 분석 도구다. "
            "아래 과거 케이스 목록을 보고 공통 패턴이나 현장에서 주의할 점을 "
            "한국어 한 문장으로, 정중한 존댓말(합니다/입니다체)로만 답한다. "
            "'관련 케이스 N건' 같은 목록 요약, 케이스 번호([case:#...]), 인사말은 절대 쓰지 않는다. "
            "패턴·교훈만 간결히, 주어진 케이스 정보만 사용하고 추측하지 않는다."
        )
        user = (
            f"사용자 질문: {user_text}\n\n"
            f"과거 케이스 목록:\n{case_lines}\n\n"
            "위 케이스들의 공통 패턴이나 주의점 한 문장:"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_tok = sum(count_tokens(m["content"]) for m in messages)
        print(f"[Timing] case-included LLM prompt: ~{prompt_tok}tok, cases={len(cases)}")

        raw_parts: list[str] = []

        def collect_synthesis(sentence: str) -> None:
            raw_parts.append(sentence)

        try:
            await self.llm.stream(
                messages,
                collect_synthesis,
                label="case-included LLM",
            )
            raw = " ".join(raw_parts)
        except Exception as e:
            print(f"[Synthesis] 오류: {e}")
            return ""
        valid_ids = await repo.get_case_ids(await get_db())
        line = strip_case_tokens(strip_unknown_cases(raw, valid_ids))
        line = re.sub(r"\s+", " ", line).strip()
        # 모델이 목록 요약("관련 케이스 N건…")을 또 뱉으면 안전망으로 제거
        line = re.sub(r"^관련 케이스\s*\d*\s*건[^.!?]*[.!?]?\s*", "", line).strip()
        return line.lstrip("※").strip()

    async def run(self, audio_path: str) -> dict:
        # main event loop 캐시 (set_loop 미호출 대비)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            if hasattr(self.tts, "set_loop"):
                self.tts.set_loop(self._loop)

        t_start = time.perf_counter()

        # ── 1. STT ──────────────────────────────────────────
        print("\n[Pipeline] ▶ 1/3 음성 인식(STT)")
        await broadcast({"type": "pipeline_status", "text": "음성 인식중..."})
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
        # 채팅 UI에 사용자 질문(STT 결과)을 즉시 표시
        await broadcast({"type": "user_msg", "text": user_text})

        # ── 2+3. 케이스 목록(결정적 표시) + LLM 패턴 합성 + TTS 오버랩 ──────
        # 화면에는 search_cases(RAG) 결과를 그대로 표시 → ID 100% 정확.
        # 음성은 케이스 번호/날짜/솔루션명을 뺀 요약만 먼저 읽고, 이후 LLM 패턴 해석을 이어 읽는다.
        print("[Pipeline] ▶ 2+3/3 케이스 표시 + LLM 패턴 합성")

        self.tts.start_workers()

        full_response: list[str] = []
        t_first_tts: float | None = None
        speaking_started = False

        async def emit(display_text: str, speech_text: str):
            nonlocal t_first_tts, speaking_started
            full_response.append(display_text)
            print(f"[Pipeline] → TTS: {speech_text}")
            self.tts.enqueue(_date_to_korean(speech_text))
            await broadcast({"type": "assistant_msg", "text": "\n".join(full_response)})
            if t_first_tts is None:
                t_first_tts = time.perf_counter()
                print(f"[Timing] 첫 음성 enqueue: {(t_first_tts - t_start)*1000:.0f}ms")
                speaking_started = True
                await broadcast({"type": "speaking", "speaking": True})

        async def display(display_text: str):
            full_response.append(display_text)
            await broadcast({"type": "assistant_msg", "text": "\n".join(full_response)})

        async def speak_only(speech_text: str):
            nonlocal t_first_tts, speaking_started
            print(f"[Pipeline] → TTS: {speech_text}")
            self.tts.enqueue(_date_to_korean(speech_text))
            if t_first_tts is None:
                t_first_tts = time.perf_counter()
                print(f"[Timing] 첫 음성 enqueue: {(t_first_tts - t_start)*1000:.0f}ms")
                speaking_started = True
                await broadcast({"type": "speaking", "speaking": True})

        async def announce_progress(display_text: str, speech_text: str | None = None):
            await broadcast({"type": "assistant_msg", "text": "\n".join(full_response + [display_text])})
            if speech_text:
                await speak_only(speech_text)

        async def wait_tts_done():
            await asyncio.get_running_loop().run_in_executor(None, self.tts.wait_done)

        async def show_case_search_pending():
            await announce_progress("케이스 검색중...", "케이스 검색중입니다.")

        async def show_summary_pending():
            pending_lines = full_response + ["※ AI 요약 생성중..."]
            await broadcast({"type": "assistant_msg", "text": "\n".join(pending_lines)})

        await show_case_search_pending()
        candidate_cases = await search_cases(user_text, limit=_CASE_PRESENT_LIMIT)
        if candidate_cases:
            print(f"[Pipeline] 관련 케이스 후보: {[c['case_id'] for c in candidate_cases]}")

        if candidate_cases:
            llm_task = asyncio.create_task(self._synthesize_pattern(user_text, candidate_cases))

            # 1) 결정적 케이스 목록 — 화면에 즉시 표시하고 TTS 큐에 바로 넣는다.
            n = len(candidate_cases)
            case_intro = f"케이스 {n}건이 존재합니다. 내용은 다음과 같습니다."
            case_intro_speech = f"케이스 {_case_count_to_korean(n)}이 존재합니다. 내용은 다음과 같습니다."
            await display(case_intro)
            for c in candidate_cases:
                line = _present_case_line(c)
                await display(line)

            await speak_only(case_intro_speech)
            for c in candidate_cases:
                speech = _case_speech_line(c)
                if speech:
                    await speak_only(speech)

            # 2) LLM 패턴 합성 한 줄
            await show_summary_pending()
            synthesis = await llm_task
            if synthesis:
                await emit(f"※ {synthesis}", synthesis)
        else:
            msg = "관련된 과거 케이스를 찾지 못했습니다. 증상이나 솔루션 이름을 더 구체적으로 말씀해 주세요."
            await emit(msg, msg)

        # 재생 완료 대기
        await wait_tts_done()

        # 말하기 종료 신호
        await broadcast({"type": "speaking", "speaking": False})

        t_end = time.perf_counter()
        print(f"[Timing] 전체: {(t_end - t_start)*1000:.0f}ms")

        ai_text = "\n".join(full_response)

        await self.memory.add_turn("user",      user_text, token_count=count_tokens(user_text))
        await self.memory.add_turn("assistant", ai_text,   token_count=count_tokens(ai_text))

        result = {
            "user_text": user_text,
            "ai_text":   ai_text,
            "turn":      self.memory.turn_count,
        }
        await self.event_bus.publish("turn_complete", result)
        return result
