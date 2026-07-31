@echo off
cd /d "%~dp0"
start "" http://localhost:8756
REM Bound to this machine explicitly: the port serves private recordings and
REM transcripts with no authentication.
.venv\Scripts\python -m uvicorn web.server:app --host 127.0.0.1 --port 8756
