import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from domains.tts.piper_engine import PiperEngine
from api.avatar_ws import broadcast


router = APIRouter(tags=["POBY"])

_tts: PiperEngine | None = None


def get_tts() -> PiperEngine:
    global _tts
    if _tts is None:
        _tts = PiperEngine.get_instance()
    return _tts


# ─────────────────────────────────────────────────────────
#  응답 모델
# ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("ok", example="ok")


class TTSTestRequest(BaseModel):
    text: str = Field(..., example="안녕! 나는 포비야.")


class TTSTestResponse(BaseModel):
    status: str = "ok"
    text: str


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


# ─────────────────────────────────────────────────────────
#  데모 음성 헬퍼
# ─────────────────────────────────────────────────────────

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
#  엔드포인트
# ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="서버 상태 확인")
async def health_check():
    return HealthResponse(status="ok")


@router.get("/runtime/mode", response_model=RuntimeModeResponse, summary="현재 입력 모드")
async def runtime_mode(request: Request):
    mode = getattr(request.app.state, "input_mode", "demo")
    return RuntimeModeResponse(
        mode=mode,
        demo_enabled=(mode == "demo"),
        mic_enabled=(mode == "mic"),
    )


@router.get("/demo/audio", response_model=DemoAudioListResponse, summary="시연용 녹음 파일 목록")
async def list_demo_audio():
    files = [
        DemoAudioItem(name=p.name, url=f"/api/v1/demo/audio/{p.name}")
        for p in _demo_audio_files()
    ]
    return DemoAudioListResponse(files=files)


@router.get("/demo/audio/{name}", summary="시연용 녹음 WAV 재생")
async def get_demo_audio(name: str, request: Request):
    reserved = getattr(request.app.state, "demo_reserved", None)
    if reserved:
        path = _resolve_demo_audio(reserved)
    elif getattr(request.app.state, "demo_running", False):
        path = _resolve_demo_audio(name)
    else:
        path = _reserve_next_demo_audio(request.app)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/demo/reserve", response_model=DemoReserveResponse, summary="다음 시연용 WAV 예약")
async def reserve_demo_audio(request: Request):
    if getattr(request.app.state, "demo_running", False):
        raise HTTPException(status_code=409, detail="demo pipeline already running")
    reserved = getattr(request.app.state, "demo_reserved", None)
    path = _resolve_demo_audio(reserved) if reserved else _reserve_next_demo_audio(request.app)
    return DemoReserveResponse(status="reserved", audio=path.name, url=f"/api/v1/demo/audio/{path.name}")


async def _run_demo_pipeline(app, name: str, path: Path) -> None:
    pipeline = app.state.pipeline
    lock = app.state.pipeline_lock
    if hasattr(pipeline, "set_loop"):
        pipeline.set_loop(asyncio.get_running_loop())
    # 사용자 질문(stt)·응답(llm)·speaking은 모두 pipeline.run 내부에서 브로드캐스트
    try:
        async with lock:
            print(f"[Demo] 실행 시작: {name}")
            await pipeline.run(str(path))
            print(f"[Demo] 실행 완료: {name}")
    except Exception as exc:
        await broadcast({"type": "llm", "text": f"실행 실패: {exc}"})
        raise
    finally:
        await broadcast({"type": "speaking", "speaking": False})
        app.state.demo_running = False


@router.post("/demo/run", response_model=DemoRunResponse, summary="예약된 시연용 WAV를 파이프라인에 입력")
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
    return DemoRunResponse(status="accepted", audio=path.name, user_text="", ai_text="")


@router.post("/demo/run/{name}", response_model=DemoRunResponse, summary="시연용 WAV를 파이프라인에 입력")
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
    return DemoRunResponse(status="accepted", audio=path.name, user_text="", ai_text="")


@router.post("/tts/test", response_model=TTSTestResponse, summary="TTS 단독 테스트")
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
