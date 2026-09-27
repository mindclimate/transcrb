@echo off
setlocal
cd /d "%~dp0"
REM Hebrew titles and device names break without it: Windows otherwise reads
REM and writes text as cp1252.
set PYTHONUTF8=1
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

if not exist ".venv\Scripts\python.exe" (
  echo Transcrb is not set up yet. Run "First-time setup.bat" first.
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found. Run "First-time setup.bat" again.
  pause
  exit /b 1
)

REM Open the page once the server has had a moment to start.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 4; Start-Process 'http://localhost:8756'"
echo Transcrb is running. Close this window to stop.
REM Bound to this machine explicitly: the port serves private recordings and
REM transcripts with no authentication.
.venv\Scripts\python -m uvicorn web.server:app --host 127.0.0.1 --port 8756
pause
