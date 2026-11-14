@echo off
title Autonomous ML Engineer Crew - Live Cloud Service
echo =====================================================================
echo  STARTING AUTONOMOUS ML ENGINEER CREW - LIVE PUBLIC CLOUD SERVICE
echo =====================================================================

cd /d "%~dp0\.."

set MODEL_DIR=artifacts\deployments\11aba26804f04096abc065a7ca06668d
set PORT=8990

echo [1/2] Starting Prediction Engine on port %PORT% ...
start "AMEC Model Service" /min .venv\Scripts\python.exe -m uvicorn serving.app:app --host 127.0.0.1 --port %PORT%
timeout /t 3 >nul

echo [2/2] Connecting to Cloudflare Global Edge Network ...
echo.
echo =====================================================================
echo LOOK FOR THE PUBLIC HTTPS URL BELOW (e.g. https://xxx.trycloudflare.com)
echo SHARE THAT URL WITH YOUR PROFESSOR / USE FOR DEMO EVALUATION
echo =====================================================================
echo.
scripts\cloudflared.exe tunnel --url http://127.0.0.1:%PORT%
