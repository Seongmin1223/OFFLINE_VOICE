@echo off
setlocal enabledelayedexpansion
title OFFLINE_VOICE Launcher

REM ============================================================
REM  OFFLINE_VOICE main launcher
REM  - Reads .env for paths/ports/options
REM  - Starts LLM server + FastAPI UI + microphone loop
REM  - Auto-detects py310 python.exe (override with PY310_PYTHON in .env)
REM  - No Korean inside this file (cp949/utf-8 conflict avoidance)
REM ============================================================

set "ROOT=%~dp0"
set "ENV_FILE=%ROOT%.env"

if not exist "%ENV_FILE%" (
    echo [ERROR] .env not found: %ENV_FILE%
    pause & exit /b 1
)

REM ---- Parse .env ( KEY=VALUE ; strip #comments ; keep / as-is ) ----
REM    NOTE: do NOT convert '/' to '\' here. URLs (e.g. http://...) would break.
REM    whisper.cpp / llama.cpp / python all accept forward slashes on Windows.
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    set "k=%%A"
    set "v=%%B"
    if defined v (
        for /f "tokens=1 delims=#" %%C in ("!v!") do set "v=%%C"
        for /l %%p in (1,1,20) do if "!v:~-1!"==" " set "v=!v:~0,-1!"
        set "!k!=!v!"
    )
)

REM ---- Sanity check required vars ----
for %%V in (WHISPER_MODEL_PATH LLAMA_BIN_PATH LLAMA_MODEL_PATH TTS_MODEL_PATH AUDIO_RECORD_FILE) do (
    if not defined %%V (
        echo [ERROR] missing %%V in .env
        pause & exit /b 1
    )
)

REM ---- Split exe paths into dir + name ----
for %%I in ("%LLAMA_BIN_PATH%") do (
    set "LLAMA_DIR=%%~dpI"
    set "LLAMA_EXE=%%~nxI"
)

REM ---- Find py310 python.exe ----
REM    Order: PY310_PYTHON in .env  -^>  common Anaconda/Miniconda locations  -^>  PATH
set "PY="
if defined PY310_PYTHON if exist "%PY310_PYTHON%" set "PY=%PY310_PYTHON%"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\py310\python.exe"   set "PY=%USERPROFILE%\anaconda3\envs\py310\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\py310\python.exe"  set "PY=%USERPROFILE%\miniconda3\envs\py310\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\envs\py310\python.exe"  set "PY=C:\ProgramData\anaconda3\envs\py310\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\envs\py310\python.exe" set "PY=C:\ProgramData\miniconda3\envs\py310\python.exe"
if not defined PY set "PY=python"

echo ------------------------------------------------------------
echo  python    : %PY%
echo  stt       : in-process Whisper
echo  llama     : %LLAMA_BIN_PATH%
echo  api port  : %API_PORT%
echo  input     : microphone
echo ------------------------------------------------------------
echo.

echo [1/2] Llama LLM    (port 8080)
REM   --mlock        : keep model pages resident in RAM (no swap)
REM   --cache-reuse  : aggressive prefix matching across requests
REM   -b / -ub       : prompt-eval batch sizes (smaller -^> faster first token on CPU)
REM   -fa on         : flash attention (memory + speed)
set "LLM_CPU_THREADS=%LLM_THREADS%"
start "Llama LLM Server" cmd /k ""%LLAMA_DIR%%LLAMA_EXE%" -m "%LLAMA_MODEL_PATH%" -c %LLM_CONTEXT_SIZE% -t %LLM_CPU_THREADS% -b 128 -ub 128 -fa on --mlock --cache-reuse 256 --port 8080"
timeout /t 7 /nobreak >nul

echo [2/2] FastAPI Backend + microphone loop
start "FastAPI Backend + Mic" cmd /k "set PYTHONUTF8=1 && set PYTHONIOENCODING=utf-8 && cd /d "%ROOT%backend" && "%PY%" -u main.py server --mic"
timeout /t 10 /nobreak >nul
start "" http://localhost:%API_PORT%

echo.
echo Main app launched. The browser UI listens to microphone input.
echo Demo mode is separate: use RUN_DEMO.bat.
echo Close the LLM and Backend windows to stop services.
pause
