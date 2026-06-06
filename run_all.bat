@echo off
setlocal enabledelayedexpansion
title OFFLINE_VOICE Launcher

REM ============================================================
REM  OFFLINE_VOICE pipeline launcher
REM  - Reads backend\.env for paths/ports/options
REM  - Auto-detects py310 python.exe (override with PY310_PYTHON in .env)
REM  - No Korean inside this file (cp949/utf-8 conflict avoidance)
REM ============================================================

set "ROOT=%~dp0"
set "ENV_FILE=%ROOT%backend\.env"

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
for %%V in (WHISPER_BIN_PATH WHISPER_MODEL_PATH LLAMA_BIN_PATH LLAMA_MODEL_PATH) do (
    if not defined %%V (
        echo [ERROR] missing %%V in .env
        pause & exit /b 1
    )
)

REM ---- Split exe paths into dir + name ----
for %%I in ("%WHISPER_BIN_PATH%") do (
    set "WHISPER_DIR=%%~dpI"
    set "WHISPER_EXE=%%~nxI"
)
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
echo  whisper   : %WHISPER_BIN_PATH%
echo  llama     : %LLAMA_BIN_PATH%
echo  api port  : %API_PORT%
echo ------------------------------------------------------------
echo.

echo [1/3] Whisper STT  (port 8081)
start "Whisper STT Server" cmd /k ""%WHISPER_DIR%%WHISPER_EXE%" -m "%WHISPER_MODEL_PATH%" -l %WHISPER_LANGUAGE% --port 8081"
timeout /t 5 /nobreak >nul

echo [2/3] Llama LLM    (port 8080)
REM   --mlock        : keep model pages resident in RAM (no swap)
REM   --cache-reuse  : aggressive prefix matching across requests
REM   -b / -ub       : prompt-eval batch sizes (smaller -^> faster first token on CPU)
REM   -fa on         : flash attention (memory + speed)
start "Llama LLM Server" cmd /k ""%LLAMA_DIR%%LLAMA_EXE%" -m "%LLAMA_MODEL_PATH%" -c %LLM_CONTEXT_SIZE% -t %LLM_THREADS% -b 256 -ub 256 -fa on --mlock --cache-reuse 256 --port 8080"
timeout /t 7 /nobreak >nul

echo [3/3] FastAPI Backend
start "FastAPI Backend" cmd /k "cd /d "%ROOT%backend" && "%PY%" main.py"

echo.
echo All three windows launched. Close any of them to stop that service.
pause
