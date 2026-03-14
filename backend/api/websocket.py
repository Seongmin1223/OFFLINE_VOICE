import json
import tempfile
import os
from fastapi import WebSocket, WebSocketDisconnect

from domains.stt.whisper_engine import WhisperEngine
from domains.llm.llama_engine import LlamaEngine
from domains.tts.piper_engine import PiperEngine
from domains.conversation.manager import ConversationManager
from core.pipeline import VoicePipeline
from core.event_bus import EventBus


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 음성 채팅 엔드포인트.

    클라이언트 → 서버:
      - text 메시지: {"type": "text", "content": "안녕하세요"}
      - binary 메시지: WAV 파일 raw bytes

    서버 → 클라이언트:
      - {"type": "stt",    "text": "인식된 텍스트"}
      - {"type": "llm",    "text": "AI 응답"}
      - {"type": "status", "message": "..."}
      - {"type": "error",  "message": "..."}
    """
    await websocket.accept()

    conversation = ConversationManager()
    event_bus    = EventBus()
    pipeline     = VoicePipeline(
        WhisperEngine(), LlamaEngine(), PiperEngine(),
        conversation, event_bus
    )
    async def on_stt(data: dict):
        await websocket.send_text(json.dumps(
            {"type": "stt", "text": data["text"]}, ensure_ascii=False
        ))

    async def on_llm(data: dict):
        await websocket.send_text(json.dumps(
            {"type": "llm", "text": data["text"]}, ensure_ascii=False
        ))

    async def on_error(data: dict):
        await websocket.send_text(json.dumps(
            {"type": "error", "stage": data["stage"], "message": data["error"]},
            ensure_ascii=False
        ))

    event_bus.subscribe("stt_complete",  on_stt)
    event_bus.subscribe("llm_complete",  on_llm)
    event_bus.subscribe("error",         on_error)

    await websocket.send_text(json.dumps(
        {"type": "status", "message": "연결되었습니다. 말씀해 주세요."},
        ensure_ascii=False
    ))

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                payload = json.loads(message["text"])
                if payload.get("type") == "reset":
                    conversation.reset()
                    await websocket.send_text(json.dumps(
                        {"type": "status", "message": "대화가 초기화되었습니다."},
                        ensure_ascii=False
                    ))
                    continue

                user_text = payload.get("content", "").strip()
                if not user_text:
                    continue

                conversation.add_user(user_text)
                prompt  = conversation.get_prompt()
                ai_text = await pipeline.llm.generate(prompt)
                conversation.add_ai(ai_text)
                await pipeline.tts.speak(ai_text)

            elif "bytes" in message:
                audio_bytes = message["bytes"]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    await pipeline.run(tmp_path)
                finally:
                    os.unlink(tmp_path)

    except WebSocketDisconnect:
        print("[WebSocket] 클라이언트 연결 종료")
    except Exception as e:
        try:
            await websocket.send_text(json.dumps(
                {"type": "error", "message": str(e)}, ensure_ascii=False
            ))
        except Exception:
            pass