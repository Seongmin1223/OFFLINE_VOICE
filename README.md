# POBY Offline Voice Assistant

POBY는 5살 아이의 친구처럼 짧고 다정하게 대답하는 로컬 음성 AI 데모입니다. 마이크 또는 녹음된 WAV를 입력으로 받아 STT, LLM 스트리밍, TTS를 모두 로컬에서 실행합니다.

```text
WAV / Microphone
  -> whisper.cpp server STT
  -> FastAPI pipeline
  -> llama.cpp server LLM streaming
  -> MeloTTS KR ONNX TTS
  -> browser avatar UI + WebSocket status
```

외부 API 호출 없이 실행하는 것을 목표로 합니다. 단, 최초 설치와 모델/바이너리 다운로드 시에는 네트워크가 필요합니다.

## 주요 기능

- 아동 음성용 Whisper-small 파인튜닝 모델을 `whisper.cpp` 서버로 추론
- `llama.cpp` 서버의 OpenAI-compatible `/v1/completions` 스트리밍 사용
- 포비 캐릭터 시스템 프롬프트와 짧은 응답 스타일 적용
- LLM 스트리밍 중 TTS를 겹쳐 실행해 첫 발화 지연 최소화
- `turns`, `facts`, `session_summaries` 3-tier SQLite 메모리 구조
- 마이크 실시간 모드와 녹음 WAV 데모 모드 분리
- 브라우저 UI에서 상태, 사용자 발화, LLM 응답, TTS 시각화 표시

## 디렉터리 구조

```text
OFFLINE_VOICE/
  RUN_POBY.bat              # 본 프로젝트 실행: 마이크 실시간 모드
  RUN_POBY_DEMO.bat         # 데모 실행: demo_audio WAV 순차 실행
  requirements.txt
  .env.example
  backend/
    main.py                 # server / once / replay / loop 진입점
    config.py               # 루트 .env 로드
    api/
      routes.py             # health, runtime mode, demo, tts test API
      avatar_ws.py          # 브라우저 WebSocket broadcast
    core/
      pipeline.py           # STT -> LLM streaming -> TTS overlap
      context_builder.py    # 메모리 기반 prompt messages 구성
      session_batch.py      # 세션 종료 시 요약/fact 추출
    database/
      schema.py             # SQLite schema
      repository.py         # turns/facts/summaries repository
    domains/
      audio_input/          # PyAudio + Silero VAD 마이크 녹음
      stt/                  # whisper.cpp HTTP client
      llm/                  # llama.cpp HTTP client
      tts/                  # MeloTTS KR ONNX worker queue
      soul/                 # 포비 캐릭터/메모리
    demo_audio/             # 데모용 아동 질문 WAV
  frontend/
    avatar.html             # 포비 UI
```

## 실행 환경

현재 검증한 환경은 다음과 같습니다.

- OS: Windows
- Python: 3.10.11
- CPU: AMD Ryzen 5 7400 6-Core Processor, 6C/12T
- RAM: 16 GB급
- STT server: `127.0.0.1:8081`
- LLM server: `127.0.0.1:8080`
- Backend/UI: `127.0.0.1:8000`

Python 의존성 설치:

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`PyAudio` 설치가 실패하면 Windows용 wheel 또는 PortAudio 설치가 필요할 수 있습니다. 마이크 모드에서만 `PyAudio`가 직접 필요하고, 데모 WAV replay는 마이크 없이도 실행할 수 있습니다.

## 모델 및 바이너리 다운로드

모델 파일은 용량이 커서 Git에 포함하지 않습니다. 아래 파일을 팀 공유 저장소, 릴리스 아티팩트, 또는 별도 전달받은 압축 파일에서 내려받아 같은 파일명으로 배치합니다.

### 1. whisper.cpp STT

필요 파일:

```text
C:\dev\whisper.cpp\Release\whisper-server.exe
C:\dev\whisper.cpp\models\ggml-poby-stt-q8_0.bin
```

설치 방법:

1. `whisper.cpp` Windows x64 바이너리를 다운로드하거나 직접 빌드합니다.
2. `whisper-server.exe`를 `C:\dev\whisper.cpp\Release\`에 둡니다.
3. 아동 음성용으로 변환/양자화된 `ggml-poby-stt-q8_0.bin`을 `C:\dev\whisper.cpp\models\`에 둡니다.

모델 파일명이 다르면 `.env`와 `RUN_POBY*.bat`의 `WHISPER_MODEL` 경로를 같이 수정해야 합니다.

### 2. llama.cpp LLM

필요 파일:

```text
C:\dev\llama.cpp\llama-server.exe
C:\dev\llama.cpp\models\poby_r8_q5km.gguf
```

설치 방법:

1. `llama.cpp` Windows CPU 또는 GPU 바이너리를 다운로드하거나 직접 빌드합니다.
2. `llama-server.exe`를 `C:\dev\llama.cpp\`에 둡니다.
3. 포비 응답용 GGUF 모델 `poby_r8_q5km.gguf`를 `C:\dev\llama.cpp\models\`에 둡니다.

현재 프롬프트 포맷은 EXAONE 계열 채팅 템플릿(`[|system|]`, `[|user|]`, `[|assistant|]`)에 맞춰 조립됩니다. 다른 모델로 교체하면 `backend/domains/llm/llama_engine.py`의 `_format_prompt()`를 확인해야 합니다.

### 3. MeloTTS KR ONNX

필요 파일:

```text
backend\models\tts\melotts\melotts_base_kr_v2.onnx
backend\vendor\MeloTTS-Windows\
```

설치 방법:

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE
git clone https://github.com/myshell-ai/MeloTTS backend\vendor\MeloTTS-Windows
python -m pip install -e backend\vendor\MeloTTS-Windows
python -m unidic download
```

그 다음 `melotts_base_kr_v2.onnx`를 `backend\models\tts\melotts\`에 둡니다. Windows에서 MeCab DLL을 찾지 못하면 `MECAB_BIN_PATH`를 `C:\Program Files\MeCab\bin`처럼 지정합니다.

### 4. 경로 확인

```powershell
Test-Path C:\dev\whisper.cpp\Release\whisper-server.exe
Test-Path C:\dev\whisper.cpp\models\ggml-poby-stt-q8_0.bin
Test-Path C:\dev\llama.cpp\llama-server.exe
Test-Path C:\dev\llama.cpp\models\poby_r8_q5km.gguf
Test-Path .\backend\models\tts\melotts\melotts_base_kr_v2.onnx
```

모두 `True`가 나와야 합니다.

## 환경 변수

```bat
copy .env.example .env
notepad .env
```

기본 실행 경로를 쓰는 경우 핵심 값은 다음과 같습니다.

```env
STT_SERVER_URL=http://127.0.0.1:8081
LLM_SERVER_URL=http://127.0.0.1:8080
WHISPER_BIN_PATH=C:/dev/whisper.cpp/Release/whisper-server.exe
WHISPER_MODEL_PATH=C:/dev/whisper.cpp/models/ggml-poby-stt-q8_0.bin
LLAMA_BIN_PATH=C:/dev/llama.cpp/llama-server.exe
LLAMA_MODEL_PATH=C:/dev/llama.cpp/models/poby_r8_q5km.gguf
TTS_MODEL_PATH=C:/Users/user/Desktop/OFFLINE_VOICE/backend/models/tts/melotts/melotts_base_kr_v2.onnx
MELOTTS_REPO_PATH=C:/Users/user/Desktop/OFFLINE_VOICE/backend/vendor/MeloTTS-Windows
```

배치 파일은 위와 같은 `C:\dev` 경로를 직접 사용합니다. 다른 위치에 설치했다면 `.env`뿐 아니라 `RUN_POBY.bat`, `RUN_POBY_DEMO.bat`의 상단 경로도 수정합니다.

## 데모 실행

데모는 `backend/demo_audio/*.wav`를 브라우저의 "데모 실행" 버튼으로 한 건씩 파이프라인에 넣습니다. 마이크 없이 STT, LLM, TTS 전체 흐름을 재현할 수 있습니다.

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE
RUN_POBY_DEMO.bat
```

실행 순서:

1. 기존 `whisper-server.exe`, `llama-server.exe`, `python.exe` 프로세스를 정리합니다.
2. STT 서버를 `8081` 포트에 띄웁니다.
3. LLM 서버를 `8080` 포트에 띄웁니다.
4. FastAPI 백엔드를 `demo` 모드로 `8000` 포트에 띄웁니다.
5. 브라우저에서 `http://localhost:8000/`을 엽니다.
6. 로딩 오버레이가 사라지면 좌하단의 "데모 실행" 버튼을 누릅니다.

수동으로 실행하려면 터미널 3개를 사용합니다.

```bat
C:\dev\whisper.cpp\Release\whisper-server.exe -m C:\dev\whisper.cpp\models\ggml-poby-stt-q8_0.bin -l ko -t 4 --port 8081
```

```bat
C:\dev\llama.cpp\llama-server.exe -m C:\dev\llama.cpp\models\poby_r8_q5km.gguf -c 1024 -t 4 -b 256 -ub 256 -fa on --mlock --cache-reuse 256 --port 8080
```

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE\backend
python main.py server --input demo
```

## 본 프로젝트 실행

마이크 실시간 모드는 다음 배치 파일을 사용합니다.

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE
RUN_POBY.bat
```

마이크가 연결되어 있지 않으면 UI가 "마이크 연결을 기다리는 중..." 상태를 표시합니다. 준비가 끝나면 아이가 말하기 시작할 때 VAD가 녹음을 시작하고, 무음 구간이 지나면 STT -> LLM -> TTS가 실행됩니다.

수동 실행은 STT/LLM 서버를 먼저 띄운 뒤 백엔드를 `mic` 모드로 실행합니다.

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE\backend
python main.py server --input mic
```

CLI 검증 명령:

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE\backend
python main.py once "demo_audio\포비는 좋아하는 음식이 뭐야 _.wav"
python main.py replay demo_audio
python main.py loop
```

## API

- `GET /api/v1/health`: 백엔드 상태 확인
- `GET /api/v1/runtime/mode`: 현재 입력 모드 확인
- `GET /api/v1/demo/audio`: 데모 WAV 목록
- `POST /api/v1/demo/reserve`: 다음 데모 WAV 예약
- `POST /api/v1/demo/run`: 예약된 데모 WAV 실행
- `POST /api/v1/tts/test`: TTS 단독 테스트
- `WS /ws/avatar`: UI 상태, STT 텍스트, LLM 텍스트, TTS 시각화 broadcast

## 데모 성능 지표

측정 명령:

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE\backend
python -u main.py replay demo_audio
```

측정은 2026-06-23에 STT/LLM을 4코어 설정으로 실행한 뒤 진행했습니다. `LLM TTFT`는 첫 토큰 시간, `최초 응답 음성`은 포비 응답이 처음 TTS 큐에 들어간 시간입니다.

| 데모 음성 | STT | LLM TTFT | 최초 응답 음성 | 전체 |
| --- | ---: | ---: | ---: | ---: |
| `포비 죽어.wav` | 981 ms | 7,294 ms | 8,504 ms | 18,999 ms |
| `포비는 좋아하는 음식이 뭐야 _.wav` | 948 ms | 2,512 ms | 4,486 ms | 11,386 ms |
| `포비야 ! 나는 딸기가 제일 좋아.wav` | 900 ms | 3,521 ms | 5,241 ms | 14,096 ms |
| `포비야 나 강아지 키우고 싶어.wav` | 939 ms | 4,227 ms | 6,040 ms | 13,890 ms |
| `포비야 나 오늘 유치원에서 친구랑 싸웠어.wav` | 985 ms | 3,626 ms | 6,008 ms | 13,079 ms |
| `포비야 비는 왜 내리는거야 _.wav` | 908 ms | 4,245 ms | 6,810 ms | 15,018 ms |
| 평균 | 944 ms | 4,238 ms | 6,182 ms | 14,411 ms |
| 2-6회 평균 | 936 ms | 3,626 ms | 5,717 ms | 13,494 ms |

서버 최초 기동, 모델 로딩, 워밍업 전 첫 요청은 이보다 길 수 있습니다. UI 모드에서는 백엔드 lifespan에서 STT 무음 추론과 LLM prefill 워밍업을 수행하고, 완료 전까지 로딩 오버레이를 유지합니다.

## 문제 해결

- `llama-server가 실행되지 않았습니다!`: `http://127.0.0.1:8080/health`가 200을 반환하는지 확인합니다.
- `whisper-server 연결 실패`: `http://127.0.0.1:8081/`가 응답하는지 확인합니다.
- `MeloTTS ONNX model not found`: `TTS_MODEL_PATH`와 `backend\models\tts\melotts\melotts_base_kr_v2.onnx` 위치를 확인합니다.
- 브라우저가 계속 로딩 중이면 백엔드 콘솔에서 warmup 로그와 WebSocket 오류를 확인합니다.
- 배치 파일은 시작/종료 시 `python.exe`를 종료합니다. 다른 Python 작업을 동시에 돌리고 있으면 수동 실행 방식을 사용합니다.
- 포트가 이미 사용 중이면 기존 `whisper-server.exe`, `llama-server.exe`, 백엔드 프로세스를 종료하거나 포트를 바꿉니다.

## Git 관리

커밋 대상:

- 소스 코드
- `frontend/avatar.html`
- `backend/demo_audio/*.wav`
- `.env.example`
- 실행 배치 파일

커밋 제외:

- `.env`
- SQLite 런타임 DB: `*.db`, `*.db-shm`, `*.db-wal`
- 로컬 모델: `*.gguf`, `*.onnx`, `*.safetensors`, `*.bin`
- `backend/models/`, `backend/vendor/`
- 학습 산출물과 대용량 데이터
