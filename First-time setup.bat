@echo off
REM Transcrb first-time setup for Windows. Mirrors scripts\setup.sh.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
REM winget puts ffmpeg here, and uv installs to the first; neither is on the
REM PATH of a window that was already open when they were installed.
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

echo == Transcrb first-time setup ==

REM 1. ffmpeg: reads every audio and video file, and does the recording.
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo Installing ffmpeg...
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo ! ffmpeg is not installed. Install it from https://www.gyan.dev/ffmpeg/builds/
  echo   and add its bin folder to PATH, then run this setup again.
  goto :fail
)

REM 2. Python 3.11 and the packages, in .venv.
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo ! uv did not install. See https://docs.astral.sh/uv/ and run this setup again.
  goto :fail
)
if exist ".venv\Scripts\python.exe" (
  echo Reusing the existing .venv ^(delete it for a clean rebuild^).
) else (
  uv venv --python 3.11 .venv || goto :fail
)
uv pip install --python .venv -e ".[dev]" || goto :fail

REM 3. A config file of its own to edit.
if not exist "config\config.toml" (
  copy "config\config.example.toml" "config\config.toml" >nul
  echo Created config\config.toml - add your HuggingFace token there for speaker labels.
)

REM Reuse the token from config.toml for faster model downloads. They work without it.
if not defined HF_TOKEN (
  for /f "usebackq delims=" %%T in (`.venv\Scripts\python -c "import tomllib;print(tomllib.load(open('config/config.toml','rb')).get('hf_token',''))"`) do set "HF_TOKEN=%%T"
)

REM 4. The Hebrew model, already converted by ivrit.ai.
if not exist "models\ivrit-whisper-turbo-ct2" (
  echo Downloading the Hebrew model ^(about 1.6GB, one time^)...
  .venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ivrit-ai/whisper-large-v3-turbo-ct2', local_dir='models/ivrit-whisper-turbo-ct2'); print('Hebrew model ready.')" || goto :fail
)

REM 5. The English model and the small one that tells the languages apart.
echo Downloading the English and language-detection models...
.venv\Scripts\python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')" || goto :fail
.venv\Scripts\python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" || goto :fail

echo.
echo == Setup complete. Double-click "Start Transcrb.bat" to run. ==
echo Reminder: add a HuggingFace token to config\config.toml for speaker labels - see docs\SETUP.md.
pause
exit /b 0

:fail
echo.
echo == Setup did not finish. Read the message above, then run it again. ==
pause
exit /b 1
