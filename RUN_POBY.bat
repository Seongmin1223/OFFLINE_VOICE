@echo off
setlocal
title POBY Launcher
chcp 65001 >nul

REM ============================================================
REM  POBY 원클릭 실행 — STT(8081) + LLM(8080) + Backend(8000)
REM  세 서버를 최소화로 띄우고, 백엔드가 응답하면 브라우저를 연다.
REM  warm(모델 로딩+사전적재)은 백엔드가 백그라운드로 수행하고,
REM  완료될 때까지 화면에는 로딩 오버레이가 표시된다.
REM ============================================================

set "ROOT=%~dp0"
set "WHISPER_EXE=C:\dev\whisper.cpp\Release\whisper-server.exe"
set "WHISPER_MODEL=C:\dev\whisper.cpp\models\ggml-poby-stt-q8_0.bin"
set "LLAMA_EXE=C:\dev\llama.cpp\llama-server.exe"
set "LLAMA_MODEL=C:\dev\llama.cpp\models\poby_r8_q5km.gguf"

echo [POBY] 이전 서버 정리...
taskkill /f /im whisper-server.exe >nul 2>&1
taskkill /f /im llama-server.exe   >nul 2>&1
taskkill /f /im python.exe         >nul 2>&1

echo [POBY] 1/3 음성 인식 서버(STT, 8081)
start "POBY STT"  /min "%WHISPER_EXE%" -m "%WHISPER_MODEL%" -l ko -t 4 --port 8081

echo [POBY] 2/3 언어 모델 서버(LLM, 8080)
start "POBY LLM"  /min "%LLAMA_EXE%" -m "%LLAMA_MODEL%" -c 1024 -t 4 -b 256 -ub 256 -fa on --mlock --cache-reuse 256 --port 8080

echo [POBY] 3/3 백엔드(8000)
start "POBY Backend" /min cmd /c "cd /d "%ROOT%backend" && python main.py server"

echo.
echo [POBY] 백엔드 기동 대기 중...
:waitloop
curl -s -o nul http://localhost:8000/ 2>nul
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo [POBY] 브라우저 열기 (로딩 화면이 warm 완료까지 표시됩니다)
start "" http://localhost:8000/

echo.
echo ============================================================
echo  POBY 실행 중. 로딩 화면이 사라지면 사용 준비 완료입니다.
echo  종료하려면 이 창을 닫거나 아무 키나 누르세요(서버 정리).
echo ============================================================
pause >nul

echo [POBY] 종료 — 서버 정리...
taskkill /f /im whisper-server.exe >nul 2>&1
taskkill /f /im llama-server.exe   >nul 2>&1
taskkill /f /im python.exe         >nul 2>&1
endlocal
