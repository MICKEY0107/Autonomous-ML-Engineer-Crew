@echo off
cd /d "%~dp0.."
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///artifacts/e2e/mlflow.db --port 5055
