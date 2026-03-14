# 🎙️ Project AIRI - 오프라인 음성 어시스턴트

> 외부 API 없이 로컬에서 완전히 동작하는 AI 음성 어시스턴트.  
> STT → LLM → TTS 전체 파이프라인이 오프라인으로 동작하며, Live2D 캐릭터와 연동됩니다.

---

## 📋 목차

- [기능](#기능)
- [시스템 요구사항](#시스템-요구사항)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [실행 방법](#실행-방법)
- [구성 요소](#구성-요소)
- [설정](#설정)
- [아키텍처](#아키텍처)

---

## ✨ 기능

- 🎤 **실시간 음성 인식** (whisper.cpp 기반, 한국어 지원)
- 🤖 **한국어 AI 응답** (EXAONE 3.5 2.4B 모델)
- 🔊 **음성 합성** (kokoro-onnx TTS)
- 🎭 **Live2D 캐릭터** (감정 연동, 립싱크)
- ⚡ **스트리밍 응답** (LLM 응답과 TTS 재생 동시 진행)
- 🧠 **대화 기억** (단기/장기 메모리)
- 😊 **감정 인식** (EMOTION 태그 파싱)
- 🌐 **WebSocket 실시간 연동** (브라우저 ↔ Python)

---

## 💻 시스템 요구사항

- Windows 10/11 (x64)
- Python 3.10+ (Anaconda 권장)
- RAM 8GB 이상 권장
- CPU: AVX2 지원 필요
- GPU: 선택 사항 (없어도 동작하나 느림)

---

## 📁 프로젝트 구조

```
offline_voice/
├── backend/
│   ├── config.py                  # 전체 설정
│   ├── main.py                    # 진입점 (server/loop/once 모드)
│   ├── core/
│   │   ├── pipeline.py            # STT→LLM→TTS 메인 파이프라인
│   │   └── event_bus.py           # 이벤트 버스
│   ├── domains/
│   │   ├── audio_input/
│   │   │   └── recorder.py        # 마이크 녹음 (pyaudio)
│   │   ├── stt/
│   │   │   ├── whisper_engine.py  # whisper-server HTTP 클라이언트
│   │   │   └── models.py
│   │   ├── llm/
│   │   │   ├── llama_engine.py    # llama-server HTTP 클라이언트 (스트리밍)
│   │   │   └── models.py
│   │   ├── tts/
│   │   │   ├── piper_engine.py    # kokoro-onnx TTS (싱글턴, 큐 재생)
│   │   │   └── models.py
│   │   ├── conversation/
│   │   │   └── manager.py         # 대화 히스토리 관리
│   │   └── soul/
│   │       ├── soul_container.py  # 캐릭터 성격/시스템 프롬프트
│   │       ├── emotion.py         # 감정 상태 트래킹
│   │       ├── memory.py          # 단기/장기 메모리
│   │       └── avatar_bridge.py   # Live2D 브릿지
│   └── api/
│       ├── routes.py              # REST API 엔드포인트
│       ├── websocket.py           # 일반 WebSocket
│       └── avatar_ws.py           # Live2D WebSocket 브로드캐스트
└── CubismSdkForWeb/               # Live2D SDK (C:\dev\에 위치)
    └── Samples/TypeScript/Demo/   # 빌드된 프론트엔드
```

---

## 🚀 설치 방법

### 1. 바이너리 빌드

#### whisper.cpp
```powershell
cd C:\dev
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j4
```

#### llama.cpp
```powershell
cd C:\dev
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j4
```

### 2. 모델 다운로드

| 모델 | 경로 | 용도 |
|------|------|------|
| `ggml-tiny.bin` | `C:\dev\whisper.cpp\models\` | STT (빠름) |
| `ggml-base.bin` | `C:\dev\whisper.cpp\models\` | STT (정확) |
| `EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf` | `C:\dev\llama.cpp\models\` | LLM (한국어) |
| `kokoro-v1.0.onnx` | `C:\dev\` | TTS 모델 |
| `voices-v1.0.bin` | `C:\dev\` | TTS 음성 |

#### 모델 다운로드 링크
- whisper 모델: https://huggingface.co/ggerganov/whisper.cpp
- EXAONE 모델: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF
- kokoro 모델: https://github.com/thewh1teagle/kokoro-onnx/releases

### 3. Python 패키지 설치

```powershell
pip install fastapi uvicorn requests kokoro-onnx sounddevice aiofiles pyaudio
```

### 4. piper (TTS 바이너리, 현재 미사용)
- https://github.com/rhasspy/piper/releases 에서 `piper_windows_amd64.zip` 다운로드
- `C:\dev\piper\` 에 압축 해제

### 5. Live2D SDK 설치
```powershell
# C:\dev\CubismSdkForWeb\ 에 SDK 압축 해제 후
cd C:\dev\CubismSdkForWeb\Samples\TypeScript\Demo
npm install
npm run build
# dist/index.html 에서 src="/assets/..." → src="./assets/..." 수정
```

---

## ▶️ 실행 방법

### 서버 실행 (권장)

터미널 3개 필요합니다:

**터미널 1 - whisper-server:**
```powershell
C:\dev\whisper.cpp\build\bin\Release\whisper-server.exe `
  -m C:\dev\whisper.cpp\models\ggml-tiny.bin `
  -l ko --port 8081
```

**터미널 2 - llama-server:**
```powershell
C:\dev\llama.cpp\build\bin\Release\llama-server.exe `
  -m C:\dev\llama.cpp\models\EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf `
  -c 512 -t 4 --port 8080
```

**터미널 3 - Python 서버:**
```powershell
cd C:\Users\하성민\Desktop\offline_voice\backend
python main.py server
```

브라우저에서 http://localhost:8000/live2d 접속하면 Live2D 캐릭터가 표시됩니다.

### 기타 실행 모드

```powershell
# WAV 파일 1회 테스트
python main.py once C:\dev\audio_16k.wav

# 마이크 루프 (Live2D 없이)
python main.py loop
```

---

## 🔧 구성 요소

### STT - whisper-server
- whisper.cpp 기반 HTTP 서버
- 포트: 8081
- 모델: ggml-tiny.bin (빠름) 또는 ggml-base.bin (정확)
- 노이즈 필터링 내장 (한글/영문 없는 결과 자동 무시)

### LLM - llama-server
- llama.cpp 기반 HTTP 서버
- 포트: 8080
- 모델: EXAONE-3.5-2.4B (한국어 특화)
- 스트리밍 응답 지원 (`/v1/chat/completions`)

### TTS - kokoro-onnx
- Python 라이브러리 (서버 불필요)
- 음성: `af_kore` (한국어 근사 지원)
- 싱글턴 패턴으로 최초 1회만 모델 로드
- 합성/재생 분리 큐로 오버랩 재생

### Live2D
- Cubism SDK for Web 기반
- 모델: Haru (기본), Hiyori 등 지원
- WebSocket `/ws/avatar` 로 Python과 실시간 연동
- 감정별 모션 자동 전환
- 외부 립싱크 값 주입으로 입 움직임 구현

---

## ⚙️ 설정

`config.py` 주요 설정값:

```python
# LLM
LLAMA_MODEL      = r"C:\dev\llama.cpp\models\EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"
LLM_MAX_TOKENS   = 60
LLM_TEMPERATURE  = 0.7
LLM_THREADS      = 4
LLM_CONTEXT_SIZE = 512
LLM_SYSTEM_PROMPT = "너는 아이리야. 17세, 밝고 친근해. 한국어로 짧게 답해."

# STT
WHISPER_MODEL    = r"C:\dev\whisper.cpp\models\ggml-tiny.bin"
WHISPER_LANGUAGE = "ko"

# TTS
PIPER_BIN        = r"C:\dev\piper\piper.exe"       # 미사용 (kokoro로 대체)
TTS_MODEL        = r"C:\dev\piper\en_US-lessac-medium.onnx"  # 미사용

# 오디오
AUDIO_SILENCE_SEC    = 2.5   # 침묵 감지 시간 (초)
AUDIO_SILENCE_THRESH = 0.008 # 침묵 임계값
AUDIO_SAMPLE_RATE    = 16000
AUDIO_MAX_SEC        = 30

# 서버
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## 🏗️ 아키텍처

```
마이크 입력
    ↓
AudioRecorder (pyaudio)
    ↓ WAV 파일
WhisperEngine → whisper-server (8081) → 텍스트
    ↓
VoicePipeline
    ↓ 스트리밍
LlamaEngine → llama-server (8080) → 토큰 스트림
    ↓ 문장 단위
PiperEngine (kokoro-onnx)
    ├── 합성 워커 (백그라운드)
    └── 재생 워커 (백그라운드, 오버랩)
    ↓
스피커 출력

동시에:
AvatarWS → WebSocket /ws/avatar → 브라우저
    ↓
Live2D 캐릭터 (감정 모션 + 립싱크)
```

---

## 📝 현재 한계 및 TODO

- [ ] 한국어 전용 TTS 모델 (현재 `af_kore`로 근사)
- [ ] GPU 가속 (CUDA 빌드 시 대폭 빠름)
- [ ] Hiyori 모델 연동 완성
- [ ] whisper-server 자동 시작 스크립트
- [ ] llama-server 자동 시작 스크립트
- [ ] 장기 메모리 UI
- [ ] 감정 파라미터 세밀화

---

## 🛠️ 트러블슈팅

### LLM 응답이 비어있음
- 프롬프트 파일 경로에 한글이 포함되면 안 됨 → `C:\dev\` 에 임시 파일 저장

### STT가 노이즈를 인식함
- `ggml-tiny.bin` 대신 `ggml-base.bin` 사용
- `whisper_engine.py`의 `_is_noise()` 필터 조정

### LLM이 너무 느림
- Edge 등 무거운 앱 종료 (RAM 확보)
- `LLM_CONTEXT_SIZE = 512` 로 줄이기
- `LLM_MAX_TOKENS = 50` 으로 줄이기
- GPU가 있으면 CUDA 빌드 적용

### Live2D JS 404 오류
- `dist/index.html` 에서 `src="/assets/..."` → `src="./assets/..."` 수정
- `npm run build` 후 매번 확인 필요
