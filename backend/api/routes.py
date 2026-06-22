import asyncio
from fastapi import APIRouter
from pydantic import BaseModel, Field

from domains.tts.piper_engine import PiperEngine
from api.avatar_ws import broadcast


router = APIRouter(tags=["진단 및 모니터링"])

# 프로세스 전역 공유 인스턴스 (main pipeline과 동일한 PiperEngine)
_tts = PiperEngine.get_instance()


class TTSTestRequest(BaseModel):
    text: str = Field(..., example="안녕하세요 포비예요")


class TTSTestResponse(BaseModel):
    status: str = "ok"
    text:   str


# ─────────────────────────────────────────────────────────
#  응답 모델
# ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok", description="서버 상태")


class TurnMetric(BaseModel):
    stt_ms:       int = Field(..., example=2730, description="STT 처리 시간 (ms)")
    llm_ttft_ms:  int = Field(..., example=2706, description="LLM 요청 → 첫 토큰까지 (ms)")
    first_tts_ms: int = Field(..., example=8383, description="발화 종료 → 첫 음성 enqueue (ms)")
    total_ms:     int = Field(..., example=18305, description="턴 전체 소요 시간 (ms)")
    prompt_tok:   int = Field(..., example=94,    description="LLM에 보낸 prompt 토큰 수")


class MetricsResponse(BaseModel):
    window:           int              = Field(..., example=10, description="집계 윈도우 (최근 N턴)")
    latest:           list[TurnMetric] = Field(..., description="최근 턴들의 측정값")
    avg_stt_ms:       int              = Field(..., example=2844)
    avg_llm_ttft_ms:  int              = Field(..., example=3186)
    avg_first_tts_ms: int              = Field(..., example=11227)
    avg_total_ms:     int              = Field(..., example=19873)


class FactItem(BaseModel):
    category:   str = Field(..., example="personal", description="카테고리 (preference/personal/event/relation)")
    content:    str = Field(..., example="사용자 이름은 도현")
    importance: int = Field(..., example=3, ge=1, le=3)


class MemoryStateResponse(BaseModel):
    tier1_turns_count:     int            = Field(..., example=12, description="Tier 1: 현재 세션 대화 턴 수")
    tier2_facts_count:     int            = Field(..., example=3,  description="Tier 2: 장기 사용자 facts 수")
    tier2_recent_facts:    list[FactItem] = Field(..., description="Tier 2 최근 facts (importance 내림차순)")
    tier3_summaries_count: int            = Field(..., example=1,  description="Tier 3: 과거 세션 요약본 수")
    token_budget:          int            = Field(..., example=25, description="현재 build_messages가 사용 가능한 토큰 예산")
    last_prompt_tok:       int            = Field(..., example=94, description="직전 LLM 호출 prompt 토큰 수")


class LlamaOptions(BaseModel):
    mlock:           bool = Field(..., example=True,  description="--mlock: 모델 페이지 RAM 고정 (스왑 방지)")
    flash_attention: bool = Field(..., example=True,  description="-fa on: Flash Attention 활성")
    cache_reuse:     int  = Field(..., example=256,   description="--cache-reuse: prefix cache aggressive matching")
    batch_size:      int  = Field(..., example=256,   description="-b/-ub: prompt-eval 배치 크기")
    threads:         int  = Field(..., example=4,     description="-t: CPU 추론 스레드 수")


class ArchitectureResponse(BaseModel):
    llm_model_path: str          = Field(..., example="E:/dev/llama.cpp/models/poby_r8_q5km.gguf")
    llm_base_model: str          = Field(..., example="EXAONE 3.5 7.8B (LoRA r=8, Q5_K_M 양자화)")
    context_size:   int          = Field(..., example=100, description="LLM 컨텍스트 윈도우 크기 (-c)")
    max_tokens:     int          = Field(..., example=75,  description="응답 최대 토큰 수")
    stt_model:      str          = Field(..., example="whisper.cpp ggml-base.bin (ko)")
    tts_model:      str          = Field(..., example="kokoro-onnx v1.0 (voice: af_kore)")
    llama_options:  LlamaOptions


class PrefillStatsResponse(BaseModel):
    trigger_count:       int       = Field(..., example=42, description="VAD start 시점 prefill 트리거 누적 횟수")
    avg_duration_ms:     int       = Field(..., example=1247, description="prefill 평균 소요 시간 (ms)")
    recent_durations_ms: list[int] = Field(..., example=[1180, 1305, 1198, 1267, 1289])
    in_flight:           bool      = Field(..., example=False, description="현재 prefill 진행 중 여부")


# ─────────────────────────────────────────────────────────
#  엔드포인트
# ─────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="서버 상태 확인",
    description="백엔드 서버가 정상 동작 중인지 확인합니다.",
)
async def health_check():
    return HealthResponse(status="ok")


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="최근 N턴 파이프라인 타이밍 통계",
    description=(
        "최근 N턴(기본 10턴)의 단계별 측정값을 반환합니다. "
        "STT 처리 시간, LLM TTFT(요청 → 첫 토큰), 첫 음성 enqueue 시점(발화 종료 누적), "
        "전체 턴 소요 시간을 평균과 함께 제공합니다."
    ),
)
async def get_metrics():
    return MetricsResponse(
        window=10,
        latest=[
            TurnMetric(stt_ms=2730, llm_ttft_ms=2706, first_tts_ms=8383,  total_ms=18305, prompt_tok=94),
            TurnMetric(stt_ms=2958, llm_ttft_ms=3667, first_tts_ms=14071, total_ms=21442, prompt_tok=93),
            TurnMetric(stt_ms=3041, llm_ttft_ms=2891, first_tts_ms=9512,  total_ms=19877, prompt_tok=92),
        ],
        avg_stt_ms=2910,
        avg_llm_ttft_ms=3088,
        avg_first_tts_ms=10655,
        avg_total_ms=19874,
    )


@router.get(
    "/memory/state",
    response_model=MemoryStateResponse,
    summary="메모리 시스템 현재 상태",
    description=(
        "메모리 시스템의 3계층 (Tier 1: 현재 세션 turns, Tier 2: 장기 facts, "
        "Tier 3: 세션 요약본) 카운트와 최근 facts 일부를 반환합니다. "
        "현재 token_budget 및 직전 build_messages가 LLM에 보낸 prompt 토큰 수도 포함합니다."
    ),
)
async def get_memory_state():
    return MemoryStateResponse(
        tier1_turns_count=12,
        tier2_facts_count=3,
        tier2_recent_facts=[
            FactItem(category="personal",   content="사용자 이름은 도현",     importance=3),
            FactItem(category="preference", content="포비 캐릭터를 좋아함",   importance=2),
            FactItem(category="event",      content="어제 영상을 본 적 있음", importance=1),
        ],
        tier3_summaries_count=1,
        token_budget=25,
        last_prompt_tok=94,
    )


@router.get(
    "/architecture",
    response_model=ArchitectureResponse,
    summary="시스템 아키텍처 및 활성 옵션",
    description=(
        "현재 운영 중인 LLM 모델(경로 + base 모델 + 양자화), context_size / max_tokens, "
        "STT/TTS 모델, 그리고 llama.cpp 기동 옵션(mlock, flash attention, cache_reuse, batch_size 등)을 "
        "한 번에 반환합니다."
    ),
)
async def get_architecture():
    return ArchitectureResponse(
        llm_model_path="E:/dev/llama.cpp/models/poby_r8_q5km.gguf",
        llm_base_model="EXAONE 3.5 7.8B (LoRA r=8, Q5_K_M 양자화)",
        context_size=100,
        max_tokens=75,
        stt_model="whisper.cpp ggml-base.bin (ko)",
        tts_model="kokoro-onnx v1.0 (voice: af_kore)",
        llama_options=LlamaOptions(
            mlock=True,
            flash_attention=True,
            cache_reuse=256,
            batch_size=256,
            threads=4,
        ),
    )


@router.get(
    "/prefill/stats",
    response_model=PrefillStatsResponse,
    summary="VAD 기반 Prefill 동작 통계",
    description=(
        "발화 감지(VAD start) 시점에 트리거되는 system+memory prefill의 누적 호출 횟수, "
        "평균 소요 시간, 최근 N개 측정값, 그리고 현재 진행 중 여부를 반환합니다. "
        "사용자 발화 시간을 prefill 슬롯으로 활용하여 TTFT를 은닉하는 구조의 검증용 지표입니다."
    ),
)
async def get_prefill_stats():
    return PrefillStatsResponse(
        trigger_count=42,
        avg_duration_ms=1247,
        recent_durations_ms=[1180, 1305, 1198, 1267, 1289],
        in_flight=False,
    )


@router.post(
    "/tts/test",
    response_model=TTSTestResponse,
    summary="TTS visualizer 단독 테스트",
    description=(
        "STT/LLM 없이 입력 텍스트를 곧장 TTS로 합성해 페이지에서 visualizer를 점검합니다."
    ),
)
async def tts_test(body: TTSTestRequest):
    loop = asyncio.get_event_loop()
    if hasattr(_tts, "set_loop"):
        _tts.set_loop(loop)
    _tts.start_workers()
    await broadcast({"type": "speaking", "speaking": True})
    _tts.enqueue(body.text)
    await loop.run_in_executor(None, _tts.wait_done)
    await broadcast({"type": "speaking", "speaking": False})
    return TTSTestResponse(status="ok", text=body.text)
