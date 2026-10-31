@echo off
setlocal
rem (Re)start the dashboard API on port 8765 (serves the React UI) in its own window
cd /d "%~dp0\.."
set PYTHONUTF8=1
set LLM_MODE=
set PUBLIC_BASE_URL=http://127.0.0.1:8765
echo STARTED %DATE% %TIME% > tmp\live_output.txt
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr LISTENING') do taskkill /F /PID %%p >> tmp\live_output.txt 2>&1
timeout /t 2 /nobreak >nul
start "AMEC dashboard API - close to stop" /min cmd /c ".\.venv\Scripts\python.exe -m uvicorn orchestrator_api:app --host 127.0.0.1 --port 8765 > tmp\api_server.log 2>&1"
rem Also start the MLflow viewer for the dashboard runs (port 5055) if it is not running
netstat -ano | findstr ":5055" | findstr LISTENING >nul || start "AMEC MLflow UI - close to stop" /min cmd /c ".\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/ui/mlflow.db --port 5055 --workers 1 > tmp\mlflow_ui.log 2>&1"
echo LIVE_EXIT=0 >> tmp\live_output.txt
exit /b
