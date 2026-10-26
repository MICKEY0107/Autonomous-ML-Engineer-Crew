@echo off
setlocal
rem Start the dashboard API (serves the React UI) in its own window
cd /d "%~dp0\.."
set PYTHONUTF8=1
set LLM_MODE=
set PUBLIC_BASE_URL=http://127.0.0.1:8765
echo STARTED %DATE% %TIME% > tmp\live_output.txt
start "AMEC dashboard API - close to stop" /min cmd /c ".\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --host 127.0.0.1 --port 8765 > tmp\api_server.log 2>&1"
echo LIVE_EXIT=0 >> tmp\live_output.txt
exit /b
