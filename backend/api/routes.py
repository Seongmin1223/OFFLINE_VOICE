import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from domains.stt.whisper_engine import WhisperEngine
from domains.llm.llama_engine import LlamaEngine
from domains.tts.piper_engine import PiperEngine
from domains.conversation.manager import ConversationManager
from core.pipeline import VoicePipeline
from core.event_bus import EventBus

router = APIRouter()

_stt          = WhisperEngine()
_llm          = LlamaEngine()
_tts          = PiperEngine()
_conversation = ConversationManager()
_event_bus    = EventBus()
_pipeline     = VoicePipeline(_stt, _llm, _tts, _conversation, _event_bus)


class TextRequest(BaseModel):
    text: str

class PipelineResponse(BaseModel):
    user_text: str
    ai_text:   str
    turn:      int

@router.get("/health")
async def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "turn": _conversation.turn_count()}


@router.post("/pipeline/audio", response_model=PipelineResponse)
async def run_pipeline_audio(file: UploadFile = File(...)):
    """
    WAV 파일을 업로드하면 STT → LLM → TTS 전체 파이프라인을 실행합니다.
    TTS 결과는 서버 스피커로 재생됩니다.
    """
    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = await _pipeline.run(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return PipelineResponse(
        user_text=result["user_text"],
        ai_text=result["ai_text"],
        turn=_conversation.turn_count(),
    )


@router.post("/pipeline/text", response_model=PipelineResponse)
async def run_pipeline_text(body: TextRequest):
    """
    텍스트를 직접 입력받아 LLM → TTS 파이프라인을 실행합니다.
    (STT 단계 스킵)
    """
    _conversation.add_user(body.text)
    prompt = _conversation.get_prompt()

    try:
        ai_text = await _llm.generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 오류: {e}")

    _conversation.add_ai(ai_text)

    try:
        await _tts.speak(ai_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 오류: {e}")

    return PipelineResponse(
        user_text=body.text,
        ai_text=ai_text,
        turn=_conversation.turn_count(),
    )


@router.post("/conversation/reset")
async def reset_conversation():
    """대화 히스토리를 초기화합니다."""
    _conversation.reset()
    return {"status": "reset"}


@router.get("/conversation/history")
async def get_history():
    """현재 대화 히스토리를 반환합니다."""
    return {"history": _conversation.history, "turn": _conversation.turn_count()}