from __future__ import annotations
import argparse
import asyncio
import os
import webbrowser

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.avatar_ws import avatar_websocket_endpoint, broadcast
from core.event_bus import EventBus
from core.pipeline import VoicePipeline
from core.session_batch import run_session_batch
from database.connection import connect, disconnect
from domains.stt.whisper_engine import WhisperEngine
from domains.llm.llama_engine import LlamaEngine
from domains.tts.typecast_engine import TypecastEngine
from config import config


# ── 시연용 라우터 (routes.py 의 /tts/test 와 동일 시그니처) ─────
class TTSTestRequest(BaseModel):
    text: str = Field(..., example="안녕하세요 포비예요")


class TTSTestResponse(BaseModel):
    status: str = "ok"
    text:   str


def _build_app(pipeline: VoicePipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await connect()
        yield
        await run_session_batch(pipeline.llm, pipeline.memory)
        await disconnect()

    app = FastAPI(
        title="POBY 풀 파이프라인 시연 (Typecast TTS)",
        description="STT(Whisper) + LLM(Llama) + Typecast 온라인 TTS",
        version="demo-full",
        docs_url=None,
        redoc_url=None,
        openapi_url="/v2/api-docs.json",
        lifespan=lifespan,
    )

    @app.post("/api/v1/tts/test", response_model=TTSTestResponse)
    async def tts_test(body: TTSTestRequest):
        loop = asyncio.get_event_loop()
        pipeline.tts.set_loop(loop)
        pipeline.tts.start_workers()
        await broadcast({"type": "speaking", "speaking": True})
        pipeline.tts.enqueue(body.text)
        await loop.run_in_executor(None, pipeline.tts.wait_done)
        await broadcast({"type": "speaking", "speaking": False})
        return TTSTestResponse(status="ok", text=body.text)

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "mode": "typecast-full"}

    app.add_api_websocket_route("/ws/avatar", avatar_websocket_endpoint)

    _frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="ui")
    return app


async def _mic_loop(pipeline: VoicePipeline):
    from domains.audio_input.recorder import AudioRecorder
    pipeline.set_loop(asyncio.get_running_loop())
    recorder = AudioRecorder(on_speech_start=pipeline.trigger_prefill)
    print("=" * 50)
    print("  마이크 루프 시작 (Typecast TTS)")
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
            await asyncio.sleep(1)


async def _serve(host: str, port: int):
    stt       = WhisperEngine()
    llm       = LlamaEngine()
    tts       = TypecastEngine.get_instance()
    event_bus = EventBus()
    pipeline  = VoicePipeline(stt, llm, tts, event_bus=event_bus)

    app = _build_app(pipeline)
    uvicorn_config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uvicorn_config)

    async def _open_browser():
        await asyncio.sleep(1.5)
        browser_host = "localhost" if host in ("0.0.0.0", "") else host
        url = f"http://{browser_host}:{port}/ui/avatar.html"
        print(f"[Browser] 자동 열기: {url}")
        webbrowser.open(url)

    print("=" * 50)
    print("  POBY 풀 파이프라인 시연")
    print("    STT(Whisper) + LLM(Llama) + Typecast TTS")
    print("  마이크로 말하거나, 페이지 하단 입력칸에 타이핑 가능")
    print("=" * 50)

    await asyncio.gather(server.serve(), _mic_loop(pipeline), _open_browser())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POBY 풀 파이프라인 시연 (Typecast)")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    args = parser.parse_args()
    asyncio.run(_serve(args.host, args.port))
