@echo off
cd /d "%~dp0.."
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/ui/mlflow.db --port 5001 --workers 1
