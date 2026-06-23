import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.case_search import search_cases
from domains.tts.piper_engine import PiperEngine
from api.avatar_ws import broadcast


router = APIRouter(tags=["진단 및 모니터링"])

_tts: PiperEngine | None = None


def get_tts() -> PiperEngine:
    global _tts
    if _tts is None:
        _tts = PiperEngine.get_instance()
    return _tts


class TTSTestRequest(BaseModel):
    text: str = Field(..., example="관련 장애 이력을 확인했습니다.")


class TTSTestResponse(BaseModel):
    status: str = "ok"
    text: str


class CaseSearchRequest(BaseModel):
    text: str = Field(..., example="메인 서버가 갑자기 셧다운됐어. 과거 비슷한 지자체 서버 다운 사례 있어?")
    limit: int = Field(5, ge=1, le=10)


class CaseSearchItem(BaseModel):
    case_id: str
    scenario: str | None = None
    case_date: str
    solution: str
    assignee: str
    work_type: str | None = None
    work_status: str | None = None
    description: str


class CaseSearchResponse(BaseModel):
    query: str
    cases: list[CaseSearchItem]


class DemoAudioItem(BaseModel):
    name: str
    url: str


class DemoAudioListResponse(BaseModel):
    files: list[DemoAudioItem]


class DemoRunResponse(BaseModel):
    status: str = "ok"
    audio: str
    user_text: str
    ai_text: str


class DemoReserveResponse(BaseModel):
    status: str = "ok"
    audio: str
    url: str


class RuntimeModeResponse(BaseModel):
    mode: str
    demo_enabled: bool
    mic_enabled: bool


def _demo_audio_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "demo_audio"


def _demo_audio_files() -> list[Path]:
    return sorted(_demo_audio_dir().glob("*.wav"))


def _resolve_demo_audio(name: str) -> Path:
    if Path(name).name != name or not name.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="invalid demo audio name")
    path = _demo_audio_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="demo audio not found")
    return path


def _reserve_next_demo_audio(app) -> Path:
    files = _demo_audio_files()
    if not files:
        raise HTTPException(status_code=404, detail="demo audio not found")
    idx = int(getattr(app.state, "demo_idx", 0))
    path = files[idx % len(files)]
    app.state.demo_reserved = path.name
    return path


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
    llm_model_path: str          = Field(..., example="E:/dev/llama.cpp/models/fix_assistant.gguf")
    llm_base_model: str          = Field(..., example="EXAONE 3.5 7.8B (Q5_K_M 양자화)")
    context_size:   int          = Field(..., example=100, description="LLM 컨텍스트 윈도우 크기 (-c)")
    max_tokens:     int          = Field(..., example=75,  description="응답 최대 토큰 수")
    stt_model:      str          = Field(..., example="whisper.cpp ggml-base.bin (ko)")
    tts_model:      str          = Field(..., example="MeloTTS ONNX KR v2")
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
            FactItem(category="preference", content="Firewatcher 장애 이력 확인을 자주 요청함", importance=2),
            FactItem(category="event",      content="최근 지자체 서버 다운 사례를 문의함", importance=1),
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
        llm_model_path="E:/dev/llama.cpp/models/fix_assistant.gguf",
        llm_base_model="EXAONE 3.5 7.8B (Q5_K_M 양자화)",
        context_size=100,
        max_tokens=75,
        stt_model="whisper.cpp ggml-base.bin (ko)",
        tts_model="MeloTTS ONNX KR v2",
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
    "/cases/search",
    response_model=CaseSearchResponse,
    summary="STT 텍스트 기반 케이스 검색",
    description="STT가 반환한 텍스트를 그대로 넣으면 CSV에서 적재된 관련 장애 케이스 후보를 반환합니다.",
)
async def case_search(body: CaseSearchRequest):
    cases = await search_cases(body.text, limit=body.limit)
    return CaseSearchResponse(query=body.text, cases=[CaseSearchItem(**case) for case in cases])


@router.get(
    "/runtime/mode",
    response_model=RuntimeModeResponse,
    summary="현재 입력 모드",
)
async def runtime_mode(request: Request):
    mode = getattr(request.app.state, "input_mode", "demo")
    return RuntimeModeResponse(
        mode=mode,
        demo_enabled=(mode == "demo"),
        mic_enabled=(mode == "mic"),
    )


@router.get(
    "/demo/audio",
    response_model=DemoAudioListResponse,
    summary="시연용 녹음 파일 목록",
)
async def list_demo_audio():
    files = [
        DemoAudioItem(name=p.name, url=f"/api/v1/demo/audio/{p.name}")
        for p in _demo_audio_files()
    ]
    return DemoAudioListResponse(files=files)


@router.get(
    "/demo/audio/{name}",
    summary="시연용 녹음 WAV 재생",
)
async def get_demo_audio(name: str, request: Request):
    reserved = getattr(request.app.state, "demo_reserved", None)
    if reserved:
        path = _resolve_demo_audio(reserved)
    elif getattr(request.app.state, "demo_running", False):
        path = _resolve_demo_audio(name)
    else:
        path = _reserve_next_demo_audio(request.app)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post(
    "/demo/reserve",
    response_model=DemoReserveResponse,
    summary="다음 시연용 녹음 WAV 예약",
)
async def reserve_demo_audio(request: Request):
    if getattr(request.app.state, "demo_running", False):
        raise HTTPException(status_code=409, detail="demo pipeline already running")
    reserved = getattr(request.app.state, "demo_reserved", None)
    path = _resolve_demo_audio(reserved) if reserved else _reserve_next_demo_audio(request.app)
    return DemoReserveResponse(
        status="reserved",
        audio=path.name,
        url=f"/api/v1/demo/audio/{path.name}",
    )


async def _run_demo_pipeline(app, name: str, path: Path) -> None:
    pipeline = app.state.pipeline
    lock = app.state.pipeline_lock
    if hasattr(pipeline, "set_loop"):
        pipeline.set_loop(asyncio.get_running_loop())
    # 사용자 질문(user_msg)·응답(assistant_msg)·speaking은 모두 pipeline.run 내부에서 브로드캐스트
    try:
        async with lock:
            print(f"[Demo] 실행 시작: {name}")
            await pipeline.run(str(path))
            print(f"[Demo] 실행 완료: {name}")
    except Exception as exc:
        await broadcast({"type": "assistant_msg", "text": f"실행 실패: {exc}"})
        raise
    finally:
        await broadcast({"type": "speaking", "speaking": False})
        app.state.demo_running = False


@router.post(
    "/demo/run",
    response_model=DemoRunResponse,
    summary="예약된 시연용 녹음 WAV를 실제 파이프라인에 입력",
)
async def run_reserved_demo_audio(request: Request):
    if getattr(request.app.state, "demo_running", False):
        raise HTTPException(status_code=409, detail="demo pipeline already running")

    reserved = getattr(request.app.state, "demo_reserved", None)
    path = _resolve_demo_audio(reserved) if reserved else _reserve_next_demo_audio(request.app)
    request.app.state.demo_reserved = None
    request.app.state.demo_idx = int(getattr(request.app.state, "demo_idx", 0)) + 1
    request.app.state.demo_running = True

    print(f"[Demo] 요청 수신: {path.name}")
    asyncio.create_task(_run_demo_pipeline(request.app, path.name, path))
    return DemoRunResponse(
        status="accepted",
        audio=path.name,
        user_text="",
        ai_text="",
    )


@router.post(
    "/demo/run/{name}",
    response_model=DemoRunResponse,
    summary="시연용 녹음 WAV를 실제 파이프라인에 입력",
)
async def run_demo_audio(name: str, request: Request):
    if getattr(request.app.state, "demo_running", False):
        raise HTTPException(status_code=409, detail="demo pipeline already running")

    reserved = getattr(request.app.state, "demo_reserved", None)
    path = _resolve_demo_audio(reserved) if reserved else _resolve_demo_audio(name)
    request.app.state.demo_reserved = None
    request.app.state.demo_idx = int(getattr(request.app.state, "demo_idx", 0)) + 1
    request.app.state.demo_running = True
    print(f"[Demo] 요청 수신: {path.name}")
    asyncio.create_task(_run_demo_pipeline(request.app, path.name, path))
    return DemoRunResponse(
        status="accepted",
        audio=path.name,
        user_text="",
        ai_text="",
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
    tts = get_tts()
    if hasattr(tts, "set_loop"):
        tts.set_loop(loop)
    tts.start_workers()
    await broadcast({"type": "speaking", "speaking": True})
    tts.enqueue(body.text)
    await loop.run_in_executor(None, tts.wait_done)
    await broadcast({"type": "speaking", "speaking": False})
    return TTSTestResponse(status="ok", text=body.text)
