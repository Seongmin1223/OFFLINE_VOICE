from __future__ import annotations

import asyncio
import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from domains.stt.whisper_engine import WhisperEngine
from domains.llm.llama_engine import LlamaEngine
from domains.tts.piper_engine import PiperEngine
from core.pipeline import VoicePipeline
from core.event_bus import EventBus
from database.connection import connect, disconnect
from core.session_batch import run_session_batch
from config import config


def create_app(pipeline: VoicePipeline, input_mode: str = "demo") -> FastAPI:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from api.routes import router
    from api.avatar_ws import avatar_websocket_endpoint

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 서버 시작 전 db 세팅
        await connect()
        mic_task = None
        if input_mode == "mic":
            mic_task = asyncio.create_task(mic_loop(pipeline))
            app.state.mic_task = mic_task
        # DB 준비 후 워밍업 트리거 (connect 전에 돌면 build_messages가 DB 미초기화로 실패).
        # create_task로 던져 서버 기동을 막지 않음 — 그동안 프론트엔드는 "시스템 가동 중" 표시.
        asyncio.create_task(pipeline.warmup())
        # 서버 유지
        yield
        # 서버 닫히기 전 훅
        if mic_task:
            mic_task.cancel()
            try:
                await mic_task
            except asyncio.CancelledError:
                pass
        await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()

    app = FastAPI(
        title="Offline Voice Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api/v1")
    app.add_api_websocket_route("/ws/avatar", avatar_websocket_endpoint)
    app.state.pipeline = pipeline
    app.state.pipeline_lock = asyncio.Lock()
    app.state.demo_running = False
    app.state.demo_idx = 0
    app.state.demo_reserved = None
    app.state.input_mode = input_mode

    frontend_path = Path(__file__).resolve().parents[1] / "frontend" / "avatar.html"

    @app.get("/")
    async def root():
        return FileResponse(frontend_path, headers={"Cache-Control": "no-store"})

    return app


async def mic_loop(pipeline: VoicePipeline):
    from domains.audio_input.recorder import AudioRecorder, input_device_available
    from api.avatar_ws import set_mic_ready
    loop = asyncio.get_running_loop()
    pipeline.set_loop(loop)

    async def _ensure_mic(prev_ok: bool) -> bool:
        """마이크 가용성 확인. 상태가 바뀔 때만 프론트에 알림."""
        ok = await loop.run_in_executor(None, input_device_available)
        if ok != prev_ok:
            await set_mic_ready(ok)
        if not ok:
            print("[마이크] 연결 대기 중...")
            await asyncio.sleep(2)
        return ok

    # 마이크가 연결될 때까지 대기 (없으면 프론트에 'no_mic' 표시)
    mic_ok = False
    while not mic_ok:
        mic_ok = await _ensure_mic(mic_ok)

    # VAD 모델 로딩은 무거우므로 executor에서 — 이벤트 루프(웹 서빙)를 막지 않음
    recorder = await loop.run_in_executor(
        None, lambda: AudioRecorder(on_speech_start=pipeline.trigger_prefill)
    )
    print("=" * 50)
    print("  마이크 루프 시작 (백그라운드)")
    print("=" * 50)

    while True:
        try:
            mic_ok = await _ensure_mic(mic_ok)
            if not mic_ok:
                continue
            audio_path = await recorder.record_async()
            if not audio_path:
                continue
            result = await pipeline.run(audio_path)
            if result["user_text"]:
                print(f"\n사용자: {result['user_text']}")
                print(f"AI    : {result['ai_text']}\n")
        except asyncio.CancelledError:
            print("\n[마이크 루프] 종료")
            break
        except Exception as e:
            print(f"[마이크 루프 오류] {e}")
            await asyncio.sleep(1)  # 오류 시 1초 대기 후 재시도


async def run_once(audio_path: str):
    await connect()
    pipeline = None
    try:
        stt       = WhisperEngine()
        llm       = LlamaEngine()
        tts       = PiperEngine()
        event_bus = EventBus()
        pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)
        result = await pipeline.run(audio_path)
        print(f"\n사용자: {result['user_text']}")
        print(f"AI    : {result['ai_text']}")
    finally:
        if pipeline:
            await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()


async def run_replay(folder: str):
    """녹음 파일 폴더를 정렬 순서대로 실제 파이프라인에 흘려보냄.
    마이크 없이 시연 영상을 찍을 때 — 사전 녹음한 질문 음성을 폴더에 넣고
    실제 STT→LLM→TTS 전체 경로를 그대로 재현."""
    wav_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".wav")
    )
    if not wav_files:
        print(f"[Replay] {folder}에 .wav 파일이 없습니다.")
        return

    await connect()
    pipeline = None
    try:
        stt       = WhisperEngine()
        llm       = LlamaEngine()
        tts       = PiperEngine()
        event_bus = EventBus()
        pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)
        for name in wav_files:
            path = os.path.join(folder, name)
            print(f"\n{'='*50}\n[Replay] ▶ {name}\n{'='*50}")
            result = await pipeline.run(path)
            print(f"\n사용자: {result['user_text']}")
            print(f"AI    : {result['ai_text']}")
    finally:
        if pipeline:
            await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()


async def run_loop():
    from domains.audio_input.recorder import AudioRecorder
    await connect()
    pipeline = None
    try:
        stt       = WhisperEngine()
        llm       = LlamaEngine()
        tts       = PiperEngine()
        event_bus = EventBus()
        pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)
        recorder  = AudioRecorder()
        print("=" * 50)
        print("  오프라인 음성 어시스턴트 시작")
        print("  종료: Ctrl+C")
        print("=" * 50)
        while True:
            try:
                audio_path = await recorder.record_async()
                result     = await pipeline.run(audio_path)
                if result["user_text"]:
                    print(f"\n사용자: {result['user_text']}")
                    print(f"AI    : {result['ai_text']}\n")
            except KeyboardInterrupt:
                print("\n종료합니다.")
                break
            except Exception as e:
                print(f"[오류] {e}")
    finally:
        if pipeline:
            await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()


async def run_server(host: str, port: int, input_mode: str = "mic"):
    """FastAPI 서버 실행. 기본은 mic 모드 — 아이가 마이크로 말하면 VAD가
    잡아 파이프라인을 돌리고 응답을 브라우저로 브로드캐스트(원본 POBY 방식).
    워밍업은 create_app lifespan(DB connect 이후)에서 1회 트리거된다."""
    import uvicorn

    stt       = WhisperEngine()
    llm       = LlamaEngine()
    tts       = PiperEngine()
    event_bus = EventBus()
    pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)

    app = create_app(pipeline, input_mode=input_mode)

    # uvicorn 설정
    uvicorn_config = uvicorn.Config(
        app, host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)

    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오프라인 음성 어시스턴트")
    sub    = parser.add_subparsers(dest="mode")

    srv = sub.add_parser("server", help="FastAPI 서버 실행")
    srv.add_argument("--host", default=config.API_HOST)
    srv.add_argument("--port", type=int, default=config.API_PORT)
    srv.add_argument(
        "--input", choices=["mic", "demo"], default="mic",
        help="mic: 본 서비스(마이크 실시간) / demo: 녹음된 데모 음성 재생",
    )

    once = sub.add_parser("once", help="WAV 파일 1회 처리")
    once.add_argument("audio", help="처리할 WAV 파일 경로")

    replay = sub.add_parser("replay", help="폴더 내 녹음 파일을 정렬 순서대로 실제 파이프라인에 재생 (마이크 없는 시연 영상용)")
    replay.add_argument("folder", help="WAV 파일들이 들어있는 폴더 경로")

    sub.add_parser("loop", help="마이크 실시간 루프")

    args = parser.parse_args()

    if args.mode == "server":
        asyncio.run(run_server(args.host, args.port, args.input))
    elif args.mode == "once":
        asyncio.run(run_once(args.audio))
    elif args.mode == "replay":
        asyncio.run(run_replay(args.folder))
    elif args.mode == "loop":
        asyncio.run(run_loop())
    else:
        asyncio.run(run_loop())
