@echo off
setlocal
rem Live end-to-end run with the real LLM providers configured in .env
cd /d "%~dp0\.."
set PYTHONUTF8=1
set LLM_MODE=
set PYTHONIOENCODING=utf-8
echo STARTED %DATE% %TIME% > tmp\live_output.txt
.\.venv\Scripts\python.exe main.py --csv data\house_prices.csv --target sale_price --goal "predict house sale price" >> tmp\live_output.txt 2>&1
echo LIVE_EXIT=%ERRORLEVEL% >> tmp\live_output.txt
echo FINISHED %DATE% %TIME% >> tmp\live_output.txt
exit /b
