@echo off
chcp 65001 >nul
title Fix Assistant - Demo Launcher
setlocal enabledelayedexpansion

REM ============================================================
REM  Fix Assistant - one-click demo launcher
REM  - Cleans old processes, starts LLM + backend, opens browser
REM  - STT is in-process, no whisper-server needed
REM  - Reads .env for model paths/options
REM ============================================================

set "ROOT=%~dp0"
set "ENV_FILE=%ROOT%.env"
set "PY=C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe"

if not exist "%ENV_FILE%" (
  echo [ERROR] .env not found: %ENV_FILE%
  pause & exit /b 1
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
  set "k=%%A"
  set "v=%%B"
  if defined v (
    for /f "tokens=1 delims=#" %%C in ("!v!") do set "v=%%C"
    for /l %%p in (1,1,20) do if "!v:~-1!"==" " set "v=!v:~0,-1!"
    set "!k!=!v!"
  )
)

set "LLAMA=%LLAMA_BIN_PATH%"s
set "LLAMA_MODEL=%LLAMA_MODEL_PATH%"

if defined PY310_PYTHON if exist "%PY310_PYTHON%" set "PY=%PY310_PYTHON%"

if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  pause & exit /b 1
)
if not exist "%LLAMA%" (
  echo [ERROR] llama-server not found: %LLAMA%
  pause & exit /b 1
)
if not exist "%LLAMA_MODEL%" (
  echo [ERROR] LLM model not found: %LLAMA_MODEL%
  pause & exit /b 1
)
if not defined TTS_MODEL_PATH (
  echo [ERROR] missing TTS_MODEL_PATH in .env
  pause & exit /b 1
)
if not exist "%TTS_MODEL_PATH%" (
  echo [ERROR] TTS model not found: %TTS_MODEL_PATH%
  pause & exit /b 1
)

echo [1/4] Cleaning old processes...
taskkill /F /IM llama-server.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/4] Starting LLM server (port 8080)...
start "LLM Server" cmd /k ""%LLAMA%" -m "%LLAMA_MODEL%" -c %LLM_CONTEXT_SIZE% -t %LLM_THREADS% -b 128 -ub 128 -fa on --mlock --cache-reuse 256 --port 8080"
timeout /t 10 /nobreak >nul

echo [3/4] Starting backend + UI (port 8000)...
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
start "Fix Assistant Backend" cmd /k "cd /d "%ROOT%backend" && "%PY%" -u main.py server"

echo [4/4] Loading models (~40s). Browser will open shortly...
timeout /t 30 /nobreak >nul
start "" http://localhost:8000

echo.
echo ============================================================
echo   Fix Assistant is running.
echo   - Browser: http://localhost:8000  (click the Run button)
echo   - If the page shows "system warming up", just wait a moment.
echo   To STOP: close the two black windows (LLM Server / Backend).
echo ============================================================
pause
