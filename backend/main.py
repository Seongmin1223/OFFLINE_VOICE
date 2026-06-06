import asyncio
import argparse
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from domains.stt.whisper_engine import WhisperEngine
from domains.llm.llama_engine import LlamaEngine
from domains.tts.piper_engine import PiperEngine
from core.pipeline import VoicePipeline
from core.event_bus import EventBus
from database.connection import connect, disconnect
from core.session_batch import run_session_batch
from config import config


def create_app(pipeline: VoicePipeline) -> FastAPI:
    from api.routes import router
    from api.websocket import websocket_endpoint
    from fastapi.responses import HTMLResponse

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 서버 시작 전 db 세팅
        await connect()
        # 서버 유지
        yield
        # 서버 닫히기 전 훅
        await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()

    app = FastAPI(
        title="오프라인 음성 서비스 명세서",
        description="경북대학교 종합설계프로젝트2",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/v2/api-docs.json",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api/v1")
    app.add_api_websocket_route("/ws", websocket_endpoint)

    # Standalone preset(헤더 + 로고)을 살리고 URL 입력바만 CSS로 숨김
    _SWAGGER_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>오프라인 음성 서비스 명세서</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    .swagger-ui .topbar .download-url-wrapper { display: none !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: '/v2/api-docs.json',
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
      layout: 'StandaloneLayout',
      deepLinking: true,
    });
  </script>
</body>
</html>"""

    @app.get("/v2/api-docs", include_in_schema=False)
    async def custom_swagger():
        return HTMLResponse(_SWAGGER_HTML)

    return app


async def mic_loop(pipeline: VoicePipeline):
    from domains.audio_input.recorder import AudioRecorder
    pipeline.set_loop(asyncio.get_running_loop())
    recorder = AudioRecorder(on_speech_start=pipeline.trigger_prefill)
    print("=" * 50)
    print("  마이크 루프 시작 (백그라운드)")
    print("=" * 50)
    while True:
        try:
            audio_path = await recorder.record_async()
            result     = await pipeline.run(audio_path)
            if result["user_text"]:
                print(f"\n사용자: {result['user_text']}")
                print(f"AI    : {result['ai_text']}\n")
        except KeyboardInterrupt:
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
        pipeline.set_loop(asyncio.get_running_loop())
        recorder  = AudioRecorder(on_speech_start=pipeline.trigger_prefill)
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


async def run_server(host: str, port: int):
    """마이크 루프 + FastAPI 서버 동시 실행."""
    stt       = WhisperEngine()
    llm       = LlamaEngine()
    tts       = PiperEngine()
    event_bus = EventBus()
    pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)

    app = create_app(pipeline)

    # uvicorn 설정
    uvicorn_config = uvicorn.Config(
        app, host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)

    # 마이크 루프 + 서버 동시 실행
    await asyncio.gather(
        server.serve(),
        mic_loop(pipeline),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오프라인 음성 어시스턴트")
    sub    = parser.add_subparsers(dest="mode")

    srv = sub.add_parser("server", help="FastAPI 서버 + 마이크 루프 실행")
    srv.add_argument("--host", default=config.API_HOST)
    srv.add_argument("--port", type=int, default=config.API_PORT)

    once = sub.add_parser("once", help="WAV 파일 1회 처리")
    once.add_argument("audio", help="처리할 WAV 파일 경로")

    sub.add_parser("loop", help="마이크 실시간 루프")

    args = parser.parse_args()

    if args.mode == "server":
        asyncio.run(run_server(args.host, args.port))
    elif args.mode == "once":
        asyncio.run(run_once(args.audio))
    elif args.mode == "loop":
        asyncio.run(run_loop())
    else:
        asyncio.run(run_loop())