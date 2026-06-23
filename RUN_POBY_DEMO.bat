@echo off
setlocal
title POBY Demo Launcher
chcp 65001 >nul

REM ============================================================
REM  POBY 데모 모드 — 녹음된 아동 질문 음성(demo_audio/*.wav)을
REM  파이프라인에 투입. 화면의 "데모 실행" 버튼으로 한 건씩 재생한다.
REM  (본 서비스는 RUN_POBY.bat = 마이크 실시간)
REM ============================================================

set "ROOT=%~dp0"
set "WHISPER_EXE=C:\dev\whisper.cpp\Release\whisper-server.exe"
set "WHISPER_MODEL=C:\dev\whisper.cpp\models\ggml-poby-stt-q8_0.bin"
set "LLAMA_EXE=C:\dev\llama.cpp\llama-server.exe"
set "LLAMA_MODEL=C:\dev\llama.cpp\models\poby_r8_q5km.gguf"

echo [POBY-DEMO] 이전 서버 정리...
taskkill /f /im whisper-server.exe >nul 2>&1
taskkill /f /im llama-server.exe   >nul 2>&1
taskkill /f /im python.exe         >nul 2>&1

echo [POBY-DEMO] 1/3 음성 인식 서버(STT, 8081)
start "POBY STT"  /min "%WHISPER_EXE%" -m "%WHISPER_MODEL%" -l ko -t 4 --port 8081

echo [POBY-DEMO] 2/3 언어 모델 서버(LLM, 8080)
start "POBY LLM"  /min "%LLAMA_EXE%" -m "%LLAMA_MODEL%" -c 1024 -t 4 -b 256 -ub 256 -fa on --mlock --cache-reuse 256 --port 8080

echo [POBY-DEMO] 3/3 백엔드(8000, demo 모드)
start "POBY Backend" /min cmd /c "cd /d "%ROOT%backend" && python main.py server --input demo"

echo.
echo [POBY-DEMO] 백엔드 기동 대기 중...
:waitloop
curl -s -o nul http://localhost:8000/ 2>nul
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo [POBY-DEMO] 브라우저 열기 (로딩 후 좌하단 '데모 실행' 버튼 클릭)
start "" http://localhost:8000/

echo.
echo ============================================================
echo  POBY 데모 실행 중. 로딩이 끝나면 '데모 실행' 버튼을 누르세요.
echo  종료하려면 이 창에서 아무 키나 누르세요(서버 정리).
echo ============================================================
pause >nul

echo [POBY-DEMO] 종료 — 서버 정리...
taskkill /f /im whisper-server.exe >nul 2>&1
taskkill /f /im llama-server.exe   >nul 2>&1
taskkill /f /im python.exe         >nul 2>&1
endlocal
