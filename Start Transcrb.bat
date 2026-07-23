@echo off
cd /d "%~dp0"
start "" http://localhost:8756
.venv\Scripts\python -m uvicorn web.server:app --port 8756
