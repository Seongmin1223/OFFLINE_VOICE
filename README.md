# 🎙️ 오프라인 실시간 음성 대화 AI 데스크톱 애플리케이션

> 완전 오프라인(On-Premise) 폐쇄망 환경에서 동작하는 **유지보수 컨텍스트 회상 음성 AI 어시스턴트**.
> CPU only (Intel i5-U 4코어 + 16GB DDR4) 환경에서 **STT → SLM → TTS** 통합 파이프라인을
> **비동기 오케스트레이션 + Prefill Hiding** 아키텍처로 운영.

**경북대학교 컴퓨터학부 산학프로젝트 · 참여기업 ㈜스피어AX · 2026.03 ~ 2026.06**

[![Paper](https://img.shields.io/badge/Paper-KIIT%202026-blue)](#-학술-자료)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-lightgrey)](#-시스템-요구사항)
[![CPU](https://img.shields.io/badge/CPU--only-i5--U%204core-orange)](#-시스템-요구사항)
[![License](https://img.shields.io/badge/License-TBD-green)](#-라이선스)

---

## 📋 목차

- [핵심 기능](#-핵심-기능)
- [선행 연구 대비 차별점](#-선행-연구-대비-차별점)
- [시스템 요구사항](#-시스템-요구사항)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [실행 방법](#️-실행-방법)
- [구성 요소](#-구성-요소)
- [3-Tier 메모리 아키텍처](#-3-tier-메모리-아키텍처)
- [Prefill Hiding](#-prefill-hiding)
- [설정](#️-설정)
- [성능 지표](#-성능-지표)
- [학술 자료](#-학술-자료)
- [트러블슈팅](#️-트러블슈팅)

---

## ✨ 핵심 기능

- 🎤 **강건성 STT** — Whisper-Small + LoRA, CTranslate2 INT8. 아동 음성 52K로 Stress-Test 훈련하여 현장 노이즈/불명확 발음 환경에서도 동작
- 🤖 **도메인 특화 SLM** — EXAONE 3.5 7.8B + QLoRA, GGUF Q5_K_M (5.15GB)
- 🔊 **경량 TTS** — MeloTTS + LoRA, ONNX FP32 정적 그래프 + OpenVoice ToneColorConverter v2 (톤 정합)
- 🧠 **3-Tier 영속 메모리** — SQLite 기반 단기/중기/장기 계층 분리 + 토큰 예산 컷오프
- ⚡ **Prefill Hiding** — 사용자 발화 시간을 KV cache 사전 적재 슬롯으로 활용 (TTFT 8.52s → 4.43s)
- 🌐 **비동기 오케스트레이션** — FastAPI(asyncio) + ThreadPoolExecutor로 코어 경합 해소
- 🪟 **웹 visualizer** — TTS 재생 시 실시간 FFT 막대그래프 (WebSocket broadcast)
- 🔒 **완전 오프라인** — 외부 클라우드/API 의존 없음. 망분리 환경 100% 동작

---

## 🎯 선행 연구 대비 차별점

| 항목 | 선행 연구 | 본 연구 |
|------|----------|--------|
| Prompt caching | 단일 system prompt 캐싱 (Anthropic, llama.cpp `--cache-reuse`) | 음성 도메인 **발화 시간을 KV cache 사전 적재 슬롯**으로 활용 |
| Long-context Memory | 클라우드 LLM 전제 RAG / 단일 임베딩 검색 (LangChain, LlamaIndex, MemGPT) | 엣지 환경 **3-Tier 영속 메모리** (단기/중기/장기 분리) |
| 음성 비서 latency | streaming TTS 위주 (Coqui XTTS streaming 등) | **발화 시간 prefill + LLM↔TTS 비동기 오버랩** 결합 |
| 온디바이스 SLM | EXAONE/Qwen 단순 양자화 | **LoRA r=8 + Q5_K_M 결합 최적화** (COLQ) |

---

## 💻 시스템 요구사항

| 항목 | 권장 사양 | 비고 |
|------|---------|------|
| OS | Windows 10/11 (x64) | 폐쇄망 / 망분리 환경 가능 |
| Python | 3.10+ (Anaconda 권장) | `py310` 환경 |
| CPU | Intel Core **i5-U series 이상 (4코어 저전력)** | AVX2 필수 |
| RAM | **16GB DDR4** | SLM Q5_K_M ≈ 5.15GB + TTS ≈ 1.5GB + OS + 캐시 |
| GPU | **불필요** | CPU only로 전 파이프라인 동작 |
| 디스크 | 15GB 이상 | 모델 + 의존 패키지 + SQLite DB |

> 본 시스템의 타겟은 **기업 시스템팀 엔지니어의 기본 지급 노트북 하한선**입니다.

---

## 📁 프로젝트 구조

```
OFFLINE_VOICE/
├── backend/
│   ├── config.py                    # 설정 (.env 로더)
│   ├── main.py                      # 진입점 (server/loop/once/tts-only 모드)
│   ├── airi.db                      # SQLite 메모리 DB (런타임 생성)
│   ├── .env                         # 환경 변수
│   │
│   ├── core/
│   │   ├── pipeline.py              # STT→SLM→TTS 비동기 파이프라인 + Prefill 트리거
│   │   ├── event_bus.py             # 이벤트 버스
│   │   ├── context_builder.py       # 3-Tier 토큰 예산 컷오프 + messages 조립
│   │   ├── session_batch.py         # 세션 종료 시 facts 추출 + summary 생성
│   │   ├── idle_watchdog.py         # 1시간 무발화 시 배치 자동 트리거
│   │   └── tokenizer.py             # llama-server /tokenize 호출 + 길이 폴백
│   │
│   ├── database/
│   │   ├── connection.py            # aiosqlite 연결 + 세션 UUID
│   │   ├── schema.py                # turns/facts/session_summaries DDL
│   │   ├── repository.py            # Tier 1/2/3 CRUD
│   │   └── SCHEMA.md                # DB 스키마 명세서
│   │
│   ├── domains/
│   │   ├── audio_input/recorder.py  # Silero VAD 기반 마이크 (start/end 콜백)
│   │   ├── stt/whisper_engine.py    # whisper-server HTTP 클라이언트
│   │   ├── llm/llama_engine.py      # llama-server 스트리밍 + Prefill API
│   │   ├── tts/
│   │   │   ├── piper_engine.py      # MeloTTS + OpenVoice 워커 큐 (싱글톤)
│   │   │   └── visualizer.py        # PCM → FFT 막대그래프 프레임
│   │   └── soul/
│   │       ├── soul_container.py    # SLM 페르소나 / 시스템 프롬프트
│   │       └── memory.py            # MemoryManager (Tier 1/2 read/write)
│   │
│   ├── api/
│   │   ├── routes.py                # 진단/모니터링 REST + /tts/test
│   │   ├── websocket.py             # 일반 WebSocket
│   │   └── avatar_ws.py             # visualizer broadcast (audio_frames/speaking)
│   │
│   ├── db_test.py                   # Phase 1/2 기본 동작 검증
│   └── synthetic_test.py            # Phase 6 정량 테스트
│
├── 모델학습/
│   ├── STT/
│   │   └── whisper_small_011_ct2/   # 학습된 Whisper-Small + LoRA (CT2 변환본)
│   ├── TTS/
│   │   ├── config.json              # OpenVoice ToneColorConverter v2 config
│   │   ├── checkpoint.pth           # OpenVoice ToneColorConverter v2 weights (131MB)
│   │   ├── neutral.wav              # 톤 정합용 reference 음성
│   │   ├── download.py              # 실험: MMS-TTS-Korean 다운로드
│   │   ├── tts_demo.py              # 실험: VITS 합성 데모
│   │   └── requirements.txt         # 학습 환경 의존성
│   └── LLM/                         # SLM 학습 자산
│
├── frontend/
│   └── avatar.html                  # POBY visualizer 페이지 (FastAPI /ui 마운트)
│
├── run_all.bat                      # 통합 실행 스크립트 (whisper + llama + FastAPI)
├── requirements.txt
└── README.md
```

---

## 🚀 설치 방법

### 1. 의존 바이너리 빌드 (Visual Studio 2022 + CMake 필요)

```cmd
cd E:\dev
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j4

cd E:\dev
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j4
```

### 2. ffmpeg 설치

`C:\ffmpeg\bin\ffmpeg.exe` 위치에 두기 (OpenVoice 톤 변환이 내부적으로 ffmpeg 호출).

```cmd
# https://www.gyan.dev/ffmpeg/builds/ 에서 release essentials 다운로드
# 압축 해제 후 bin 폴더를 C:\ffmpeg\bin 으로 이동
```

### 3. 모델 배치

| 모델 | 경로 | 출처 / 비고 |
|------|------|------|
| **STT** `ggml-base.bin` | `E:/dev/whisper.cpp/models/` | https://huggingface.co/ggerganov/whisper.cpp |
| **SLM** `poby_r8_q5km.gguf` | `E:/dev/llama.cpp/models/` | 자체 학습 — https://huggingface.co/hyeonwowo/POBY |
| **TTS** `melotts_base_kr_v2.onnx` | `C:/voice/` | MeloTTS Korean v2 + LoRA(r=16, α=32) 병합 후 ONNX 변환 |
| **OpenVoice** `config.json` + `checkpoint.pth` | `C:/OpenVoice/checkpoints_v2/converter/` | `모델학습/TTS/` 에서 복사 |
| **톤 reference** `neutral.wav` | `C:/dev/voices/` | `모델학습/TTS/` 에서 복사 |

> `모델학습/TTS/` 아래에 OpenVoice v2 config/checkpoint와 `neutral.wav`가 함께 들어있습니다. 위 경로로 복사하거나 (필요 시) symbolic link.

### 4. Python 가상환경 + 패키지

```cmd
conda create -n py310 python=3.10 -y
conda activate py310
cd OFFLINE_VOICE\backend
pip install -r requirements.txt
```

추가로 TTS 의존 패키지 (MeloTTS + OpenVoice):

```cmd
# MeloTTS (Windows 빌드 소스)
git clone https://github.com/myshell-ai/MeloTTS C:\MeloTTS-Windows
cd C:\MeloTTS-Windows
pip install -e .
python -m unidic download

# OpenVoice
pip install git+https://github.com/myshell-ai/OpenVoice.git
```

### 5. `.env` 구성

`backend/.env` 생성 (예시는 [설정](#️-설정) 섹션 참조).

---

## ▶️ 실행 방법

### 통합 실행 (권장)

```cmd
.\run_all.bat
```

자동으로 다음 3개 창이 띄워짐:
- whisper-server (포트 **8081**)
- llama-server (포트 **8080**, `--mlock` / `--cache-reuse 256` / `-fa on` 활성)
- FastAPI Backend (포트 **8000**, `server` 모드)

기동 후 약 1.5초 뒤 **자동으로 브라우저가 `http://localhost:8000/ui/avatar.html`을 열어** POBY visualizer 페이지가 표시됩니다.

### 진단 / 모니터링 콘솔

```
http://localhost:8000/v2/api-docs
```

- 시스템 상태, 최근 N턴 메트릭, 3-Tier 메모리 현황, Prefill 통계 등을 Swagger UI로 노출
- `POST /api/v1/tts/test` — 마이크 없이 텍스트 입력만으로 TTS + visualizer 단독 테스트

### 개별 모드

```cmd
python main.py once path\to\audio.wav    # WAV 파일 1회 처리
python main.py loop                      # 마이크 루프만 (API 서버 X)
python main.py tts-only                  # STT/LLM 없이 TTS visualizer 전용
```

---

## 🔧 구성 요소

### STT — whisper-server
- llama.cpp 자매 프로젝트 `whisper.cpp` HTTP 서버 (포트 8081)
- 학습 모델: **Whisper-Small + LoRA** (CT2 변환본, INT8 양자화)
- 학습 데이터: AI Hub 한국어 아동 음성 52K (Stress-Test Training)
- 노이즈 필터링 내장 (한글/영문 없는 결과 자동 무시)

### SLM — llama-server
- llama.cpp HTTP 서버 (포트 8080)
- 모델: **EXAONE 3.5 7.8B + QLoRA (r=8, α=2)**, GGUF **Q5_K_M** (5.15GB)
- 기동 옵션 (in `run_all.bat`):
  ```
  -c 100 -t 4 -b 256 -ub 256 -fa on --mlock --cache-reuse 256
  ```
- 스트리밍 응답 (`/v1/completions`, EXAONE 채팅 템플릿 강제 조립)
- 응답 `timings` 활용한 prompt_n / predicted_n 측정

### TTS — MeloTTS + OpenVoice
- **합성**: MeloTTS Korean v2, LoRA(r=16, α=32) → Speaker Embedding 병합 → ONNX FP32 정적 그래프
- **톤 정합**: OpenVoice ToneColorConverter v2 (`neutral.wav` reference 기반 SE 추출 후 변환)
- **싱글톤 패턴**: 모델/큐/스레드 프로세스당 1세트 (`PiperEngine.get_instance()`)
- **워커 큐 분리**: 합성 워커 + 재생 워커 → SLM 다음 토큰 생성과 오버랩
- **PCM → FFT broadcast**: 재생 시점 `visualizer.py`가 막대그래프 프레임을 WebSocket으로 송출 → 페이지에서 실시간 시각화

### 3-Tier 영속 메모리
- SQLite (aiosqlite, **WAL 모드**)
- 자세한 스키마: [`backend/database/SCHEMA.md`](backend/database/SCHEMA.md)

### Web UI — `frontend/avatar.html`
- 캐릭터 + 상태 표시 + 실시간 FFT visualizer + 텍스트 입력 TTS 테스트 박스
- FastAPI가 `/ui/avatar.html` 로 정적 마운트
- WebSocket `ws://localhost:8000/ws/avatar` 로 `speaking` / `audio_frames` 신호 수신

---

## 🧠 3-Tier 메모리 아키텍처

| Tier | 테이블 | 보존 범위 | 압축률 | 생성 시점 |
|------|--------|----------|--------|----------|
| **Tier 1 (단기)** | `turns` | 현재 세션 대화 턴 원본 | 0 (원본) | 매 발화/응답마다 즉시 |
| **Tier 2 (중기)** | `facts` | 세션 경계 없는 장기 사용자 사실 | 중간 (LLM 추출 JSON) | 세션 종료 / idle 1시간 배치 |
| **Tier 3 (장기)** | `session_summaries` | 과거 세션 요약본 | 높음 (LLM 요약) | 세션 종료 / idle 1시간 배치 |

매 LLM 호출 시 `context_builder.py`가 토큰 예산 안에서 **압축률 높은 순서(Tier 3 → 2 → 1)** 로 컷오프 조립.
초과 시 즉시 break, Tier 1은 오래된 턴부터 cutoff.

---

## ⚡ Prefill Hiding

```
사용자 발화 시작 (VAD 'start' 이벤트)
       │
       ├──→ [Prefill 트리거] 별도 스레드 발사
       │         │
       │         ↓
       │     llama-server에 system + memory prefix 사전 적재
       │     (max_tokens=1, KV cache만 채움)
       │
       └──→ 사용자 발화 진행 (수 초)
                 │
                 ↓
            VAD 'end' / STT 처리
                 │
                 ↓
            본 LLM 요청 (이미 cache hit 상태)
                 │
                 ↓
            첫 토큰 즉시 생성 → 첫 문장 단위 TTS enqueue
                 │
                 ↓
            합성 + 재생 (LLM 다음 토큰 생성과 오버랩)
```

CPU 환경에서 가장 비용이 큰 **prefill 연산**을 음성 도메인 고유의 유휴 슬롯(**발화 시간**)에 숨김.
GPU 프로젝트에서는 안 보이는 문제 — CPU에서 prefill 비용이 토큰 수에 거의 선형이라 cache hit률 한 자리수 차이가 TTFT 초 단위 직격.

---

## ⚙️ 설정

### `backend/.env` 예시

```env
# Server
API_HOST=0.0.0.0
API_PORT=8000
STT_SERVER_URL=http://127.0.0.1:8081
LLM_SERVER_URL=http://127.0.0.1:8080

# Paths
WHISPER_BIN_PATH=E:/dev/whisper.cpp/build/bin/Release/whisper-server.exe
WHISPER_MODEL_PATH=E:/dev/whisper.cpp/models/ggml-base.bin
LLAMA_BIN_PATH=E:/dev/llama.cpp/build/bin/Release/llama-server.exe
LLAMA_MODEL_PATH=E:/dev/llama.cpp/models/poby_r8_q5km.gguf
AUDIO_RECORD_FILE=E:/dev/recorded.wav

# VAD
AUDIO_SILENCE_THRESH=0.08
AUDIO_SILENCE_SEC=2.5

# SLM
LLM_MAX_TOKENS=50          # 응답 토큰 한도
LLM_TEMPERATURE=0.65
LLM_TOP_P=0.9
LLM_TOP_K=40
LLM_REPEAT_PENALTY=1.1
LLM_THREADS=4
LLM_CONTEXT_SIZE=256       # 컨텍스트 윈도우 (max: 32768)

# STT
WHISPER_LANGUAGE=ko
WHISPER_THREADS=4
```

### TTS 경로 (현재는 `piper_engine.py` 상수)

MeloTTS / OpenVoice 관련 경로는 현재 `backend/domains/tts/piper_engine.py` 상단에 상수로 박혀있습니다 (위 [설치 방법 §3](#-설치-방법) 표 그대로):

```python
ONNX_PATH    = r'C:\voice\melotts_base_kr_v2.onnx'
CKPT_PATH    = r'C:\OpenVoice\checkpoints_v2\converter'
NEUTRAL_REF  = r'C:\dev\voices\neutral.wav'
```

> `LLM_CONTEXT_SIZE`는 llama-server `-c` 옵션과 동기화 — `.env` 변경 시 서버 재기동 필요.

---

## 📊 성능 지표

타겟 디바이스(**i5-U 4코어 + 16GB DDR4 + CPU only**) 기준:

| 지표 | Base | Proposed | 데이터셋 |
|------|------|----------|---------|
| **CER** (문자 오류율) | 11.94% | **9.64%** | AI Hub 아동 음성 (52K) |
| **TTFT** (첫 토큰 지연) | 8.520s | **4.431s** | 자체 멀티턴 합성 (1K) |
| **C_hit** (KV cache 적중률) | 0% | **96.4%** | 자체 멀티턴 합성 (1K) |
| **Recall@3** (장애 케이스 재현율) | 42.5% | **92.8%** | 스피어AX 696건 |
| **H_rate** (환각률) | 21.4% | **3.2%** | 스피어AX 696건 |
| **RAM 점유 (런타임 최대)** | — | **14.2GB** | 100턴 연속 부하 시뮬레이션 |

> 추가 지표: RTF (STT 0.451, TTS 0.271), 페르소나 일관성 P 84.1%, DB 조회 오버헤드 O_mem 0.3%

---

## 📚 학술 자료

본 프로젝트의 핵심 아키텍처는 다음 논문으로 정리되어 투고되었습니다:

1. **오프라인 실시간 음성 대화를 위한 인지-기억-발화 통합 비동기 아키텍처 설계**
   _2026 한국정보기술학회 하계 종합학술대회_

2. **COLQ: 온디바이스 페르소나 SLM을 위한 LoRA Rank-양자화 조합 최적화**
   _2026 한국정보기술학회 하계 종합학술대회_

---

## 🛠️ 트러블슈팅

### `ModuleNotFoundError: No module named 'melo.api'`
- `C:\MeloTTS-Windows` 경로에 MeloTTS 소스가 없거나 `pip install -e .`가 안 됨.
- [설치 방법 §4](#-설치-방법) 의 MeloTTS 단계 재확인.

### `FileNotFoundError: neutral.wav`
- OpenVoice 톤 정합용 reference 누락.
- `모델학습/TTS/neutral.wav`를 `C:\dev\voices\neutral.wav`로 복사.

### 첫 발화 TTFT가 매우 긺 (콜드 페널티)
- 모델 mmap 페이지 + KV cache가 비어있어 발생.
- `--mlock` 옵션이 활성화됐는지 확인 (`run_all.bat` 기본 적용).
- 서버 부팅 후 **사전 발화 1회**로 워밍업 권장.

### STT 인식률 저하
- 환경 소음 임계값 조정: `AUDIO_SILENCE_THRESH`
- 더 큰 모델 / 학습본으로 교체:
  - `ggml-base.bin` → `ggml-small.bin`
  - 또는 `모델학습/STT/whisper_small_011_ct2/`의 학습본 활용

### LLM 응답이 짧거나 잘림
- `LLM_MAX_TOKENS` 상향
- `LLM_CONTEXT_SIZE` 상향 (단, prefill 비용 증가 → llama-server 재기동 필요)

### 코어 경합으로 시스템 프리징
- `LLM_THREADS`를 줄여 TTS와의 경합 완화
- 다른 무거운 프로세스 종료 (브라우저 등)

### 메모리(Tier 1/2/3)가 LLM 컨텍스트에 안 들어감
- `LLM_CONTEXT_SIZE − LLM_MAX_TOKENS = token_budget` 확인.
- 음수/0이면 `context_builder.py` 컷오프 루프가 즉시 break → 메모리 0 토큰 주입.
- `.env`에서 둘의 차이를 양수로 유지.

### visualizer 페이지가 안 열림 / 빈 화면
- FastAPI `server` 모드로 띄웠는지 확인 (`run_all.bat` 마지막 줄에 `main.py server`).
- `frontend/avatar.html` 존재 확인 (`/ui` static mount의 root 디렉토리).
- 브라우저 콘솔에서 WebSocket `ws://localhost:8000/ws/avatar` 연결 상태 확인.
