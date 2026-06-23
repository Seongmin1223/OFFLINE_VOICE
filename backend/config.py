import os
from pathlib import Path
from dotenv import load_dotenv

# .env 로드 (backend/ 또는 프로젝트 루트 모두 탐색)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

def get_env_strict(key, default=None, is_path=False):
    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(f"필수 환경 변수 '{key}' 미설정. .env 파일 수정 필요")

    if is_path and not value.strip():
        raise RuntimeError(f"환경 변수 '{key}'의 경로가 비었음")
        
    return value

class Config:
    #STT 
    WHISPER_MODEL = get_env_strict("WHISPER_MODEL_PATH", is_path=True)
    WHISPER_BIN = get_env_strict("WHISPER_BIN_PATH", is_path=True)
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")
    WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "4"))
    WHISPER_SERVER_URL = os.getenv("STT_SERVER_URL", "http://127.0.0.1:8081")

    #LLM
    LLAMA_MODEL = get_env_strict("LLAMA_MODEL_PATH", is_path=True)
    LLAMA_BIN = get_env_strict("LLAMA_BIN_PATH", is_path=True)
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "50"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
    LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))
    LLM_THREADS = int(os.getenv("LLM_THREADS", "4"))
    LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "512"))
    LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080")

    # TTS — MeloTTS KR ONNX. TTS_CONFIG_PATH는 호환용으로만 남아있다.
    TTS_MODEL = os.getenv("TTS_MODEL_PATH", "unused")
    TTS_CONFIG = os.getenv("TTS_CONFIG_PATH", "unused")
    TTS_VOICE = os.getenv("TTS_VOICE", "af_kore")
    PIPER_BIN = os.getenv("PIPER_BIN_PATH", "unused")
    TTS_OUTPUT_FILE = os.getenv("TTS_OUTPUT_FILE", "unused")

    #AUDIO
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_CHUNK_SIZE = int(os.getenv("AUDIO_CHUNK_SIZE", "512"))
    AUDIO_SILENCE_THRESH = float(os.getenv("AUDIO_SILENCE_THRESH", "0.08"))
    AUDIO_VAD_THRESHOLD = float(os.getenv("AUDIO_VAD_THRESHOLD", "0.8"))
    AUDIO_SILENCE_SEC = float(os.getenv("AUDIO_SILENCE_SEC", "2.5"))
    AUDIO_MAX_SEC = float(os.getenv("AUDIO_MAX_SEC", "30.0"))
    AUDIO_RECORD_FILE = get_env_strict("AUDIO_RECORD_FILE", is_path=True)

    # ETC
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    CONVERSATION_MAX_HISTORY = int(os.getenv("CONVERSATION_MAX_HISTORY", "10"))

    # DATABASE
    # 기본값: backend/ 디렉토리 안의 airi.db
    DB_PATH = os.getenv(
        "DB_PATH",
        str(os.path.join(os.path.dirname(__file__), "airi.db"))
    )

config = Config()
