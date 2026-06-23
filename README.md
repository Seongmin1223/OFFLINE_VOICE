# Offline Voice Fix Assistant

기업 시스템팀 엔지니어를 위한 오프라인 장애 이력 회상 음성 어시스턴트입니다.
마이크 또는 데모 wav 입력을 STT로 텍스트화하고, 과거 장애 케이스를 검색한 뒤 LLM 요약과 TTS 응답을 제공합니다.

## 현재 구조

```text
STT(in-process Whisper)
  -> case search(SQLite CSV seed)
  -> LLM(llama.cpp server)
  -> grounding / summary
  -> MeloTTS KR ONNX
  -> FastAPI + WebSocket UI
```

데모 모드와 본 실행 모드는 분리되어 있습니다.

- `run_all.bat`: 본 프로그램. FastAPI UI와 마이크 루프를 함께 실행합니다.
- `RUN_DEMO.bat`: 시연용. `backend/demo_audio/*.wav`를 순서대로 재생하고 같은 wav를 실제 STT에 넣습니다.

## 주요 파일

```text
backend/
  main.py                       # server/loop/once/replay 진입점
  config.py                     # 루트 .env 로더
  api/
    routes.py                   # health, runtime mode, demo, case search, tts test
    avatar_ws.py                # 프론트 상태/TTS visualizer WebSocket
  core/
    pipeline.py                 # STT -> case search -> LLM -> TTS 파이프라인
    case_search.py              # 키워드/2-gram IDF 기반 케이스 검색
    context_builder.py          # 메모리/케이스 컨텍스트 조립
    grounding.py                # 케이스 번호 hallucination 방지
  database/
    seed_cases.csv              # 샘플 케이스 CSV
    case_importer.py            # CSV -> SQLite 적재
  domains/
    audio_input/recorder.py     # PyAudio + Silero VAD 마이크 녹음
    stt/whisper_engine.py       # Transformers Whisper 추론
    llm/llama_engine.py         # llama.cpp HTTP client
    tts/piper_engine.py         # MeloTTS KR ONNX worker queue
frontend/
  avatar.html                   # 운영/데모 공용 UI
```

## 모델과 로컬 자산

모델 파일은 GitHub에 올리지 않습니다. `.env`에서 로컬 경로를 지정합니다.

필요한 로컬 자산:

- Whisper 모델 디렉터리: `WHISPER_MODEL_PATH`
- llama.cpp `llama-server.exe`: `LLAMA_BIN_PATH`
- GGUF LLM 모델: `LLAMA_MODEL_PATH`
- MeloTTS KR ONNX: `TTS_MODEL_PATH`
- MeloTTS Windows 소스 체크아웃: `MELOTTS_REPO_PATH`

`backend/vendor/`, `backend/models/`, `모델학습/`은 `.gitignore`로 제외됩니다.

## 설치

Python 3.10 환경에서 설치합니다.

```bat
cd /d C:\Users\user\Desktop\OFFLINE_VOICE
python -m pip install -r requirements.txt
```

MeloTTS는 Windows에서 로컬 소스 체크아웃을 권장합니다.

```bat
git clone https://github.com/myshell-ai/MeloTTS backend\vendor\MeloTTS-Windows
python -m pip install -e backend\vendor\MeloTTS-Windows
python -m unidic download
```

그 다음 `.env.example`을 `.env`로 복사하고 로컬 모델 경로를 채웁니다.

## 실행

본 프로그램, 마이크 입력:

```bat
run_all.bat
```

실행되면 다음이 뜹니다.

- `Llama LLM Server`: `localhost:8080`
- `FastAPI Backend + Mic`: `localhost:8000`
- 브라우저 UI: `http://localhost:8000`

데모:

```bat
RUN_DEMO.bat
```

데모 모드는 브라우저의 실행 버튼을 누를 때마다 `backend/demo_audio`의 wav를 순서대로 재생하고, 같은 파일을 실제 STT 파이프라인에 넣습니다.

CLI 단일 wav 검증:

```bat
cd backend
python -u main.py once demo_audio\01.wav
```

CLI 폴더 replay:

```bat
cd backend
python -u main.py replay demo_audio
```

CLI 마이크 루프만:

```bat
cd backend
python -u main.py loop
```

## UI 동작

- 마이크 모드에서는 실행 버튼이 숨겨지고, 마이크 입력을 대기합니다.
- 데모 모드에서는 실행 버튼이 보입니다.
- 처리 상태는 채팅 화면에 표시됩니다.
  - `음성 인식중...`
  - `케이스 검색중...`
  - `AI 요약 생성중...`
- 케이스 검색 중/케이스 개수 안내도 TTS로 말합니다.
- 케이스 내용 TTS는 케이스 번호와 날짜를 읽지 않고 요약 내용을 읽습니다.

## 케이스 데이터

초기 샘플 데이터는 `backend/database/seed_cases.csv`입니다. DB가 비어 있으면 서버 시작 시 자동 적재됩니다.

실제 696건 CSV로 교체할 때는 같은 컬럼 체계를 유지합니다.

주요 컬럼:

- `case_id`
- `시나리오` 또는 `scenario`
- `date`
- `솔루션` 또는 `solution`
- `담당자` 또는 `name`
- `workType`
- `workStatus`
- `description`

CSV 교체 후 반드시 실행합니다.

```bat
cd backend
python case_scenario_test.py
```

## 테스트

```bat
cd backend
python -m py_compile main.py api\routes.py core\pipeline.py
python db_test.py
python synthetic_test.py
python case_scenario_test.py
```

## GitHub 배포 메모

커밋 대상:

- 소스 코드
- `backend/database/seed_cases.csv`
- `backend/demo_audio/*.wav`
- `.env.example`
- 실행 배치 파일

커밋 제외:

- `.env`
- SQLite DB/WAL 파일
- 로컬 모델 파일
- `backend/vendor/`
- 학습 산출물
- 보고서/PDF 추출물

POBY 이전 상태는 `archive/poby-before-fix-assistant` 브랜치에 보존해두면 나중에 복구할 수 있습니다.
