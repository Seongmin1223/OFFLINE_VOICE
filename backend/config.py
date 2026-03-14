import os

class Config:
    WHISPER_MODEL = r"C:\dev\whisper.cpp\models\ggml-base.bin"
    LLAMA_MODEL = r"C:\dev\llama.cpp\models\tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    TTS_MODEL  = r"C:\dev\piper\en_US-lessac-medium.onnx"
    TTS_CONFIG = r"C:\dev\piper\en_US-lessac-medium.onnx.json"
    WHISPER_BIN = r"C:\dev\whisper.cpp\build\bin\Release\whisper-cli.exe"
    LLAMA_BIN = r"C:\dev\llama.cpp\build\bin\Release\llama-completion.exe"
    PIPER_BIN  = r"C:\dev\piper\piper.exe"
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ko")
    WHISPER_THREADS  = int(os.getenv("WHISPER_THREADS", "4"))

    LLM_MAX_TOKENS   = 50
    LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_THREADS      = int(os.getenv("LLM_THREADS", "4"))
    LLM_CONTEXT_SIZE = 512
    LLM_SYSTEM_PROMPT = "너는 아이리야. 17세, 밝고 친근해. 한국어로 짧게 답해."
    TTS_OUTPUT_FILE = r"C:\dev\tts_output.wav"
    AUDIO_SAMPLE_RATE    = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS       = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_CHUNK_SIZE     = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
    AUDIO_SILENCE_THRESH = 0.08
    AUDIO_SILENCE_SEC    = 2.5
    AUDIO_MAX_SEC        = float(os.getenv("AUDIO_MAX_SEC", "30.0"))
    AUDIO_RECORD_FILE = r"C:\dev\recorded.wav"

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    CONVERSATION_MAX_HISTORY = int(os.getenv("CONVERSATION_MAX_HISTORY", "10"))


config = Config()