
from __future__ import annotations
import argparse
import asyncio
import os
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# config 를 가장 먼저 import — load_dotenv 가 typecast_engine 의
# 모듈 레벨 os.getenv() 평가보다 반드시 먼저 실행되어야 함
from config import config
from api.avatar_ws import avatar_websocket_endpoint, broadcast
from domains.tts.typecast_engine import TypecastEngine


# ── 시연용 라우터 (routes.py 와 동일 시그니처 / 동일 경로) ─────────
class TTSTestRequest(BaseModel):
    text: str = Field(..., example="안녕하세요 포비예요")


class TTSTestResponse(BaseModel):
    status: str = "ok"
    text:   str


def _build_app() -> FastAPI:
    app = FastAPI(
        title="POBY 시연 (Typecast TTS)",
        description="시연용 - Typecast 온라인 TTS 만 띄우는 경량 서버",
        version="demo",
        docs_url=None,
        redoc_url=None,
        openapi_url="/v2/api-docs.json",
    )

    tts = TypecastEngine.get_instance()

    @app.on_event("startup")
    async def _startup():
        tts.set_loop(asyncio.get_running_loop())
        tts.start_workers()

    @app.post("/api/v1/tts/test", response_model=TTSTestResponse)
    async def tts_test(body: TTSTestRequest):
        loop = asyncio.get_event_loop()
        tts.set_loop(loop)
        tts.start_workers()
        await broadcast({"type": "speaking", "speaking": True})
        tts.enqueue(body.text)
        await loop.run_in_executor(None, tts.wait_done)
        await broadcast({"type": "speaking", "speaking": False})
        return TTSTestResponse(status="ok", text=body.text)

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "mode": "typecast-demo"}

    app.add_api_websocket_route("/ws/avatar", avatar_websocket_endpoint)

    _frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="ui")

    _SWAGGER_HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>POBY 시연 (Typecast)</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
<style>.swagger-ui .topbar .download-url-wrapper{display:none!important;}</style>
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
<script>window.ui=SwaggerUIBundle({url:'/v2/api-docs.json',dom_id:'#swagger-ui',
presets:[SwaggerUIBundle.presets.apis,SwaggerUIStandalonePreset],
layout:'StandaloneLayout',deepLinking:true});</script></body></html>"""

    @app.get("/v2/api-docs", include_in_schema=False)
    async def swagger():
        return HTMLResponse(_SWAGGER_HTML)

    return app


async def _serve(host: str, port: int):
    app = _build_app()
    uvicorn_config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uvicorn_config)

    async def _open_browser():
        await asyncio.sleep(1.5)
        browser_host = "localhost" if host in ("0.0.0.0", "") else host
        url = f"http://{browser_host}:{port}/ui/avatar.html"
        print(f"[Browser] 자동 열기: {url}")
        webbrowser.open(url)

    print("=" * 50)
    print("  POBY 시연 모드 - Typecast 온라인 TTS")
    print("  하단 입력칸에 텍스트를 넣으면 포비가 읽어줘요")
    print("=" * 50)

    await asyncio.gather(server.serve(), _open_browser())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POBY 시연 (Typecast)")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    args = parser.parse_args()
    asyncio.run(_serve(args.host, args.port))
