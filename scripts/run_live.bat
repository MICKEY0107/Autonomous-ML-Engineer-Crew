@echo off
rem Live end-to-end run with the real LLM providers from .env
cd /d "%~dp0\.."
echo STARTED %DATE% %TIME% > tmp\live_output.txt
.\.venv\Scripts\python.exe main.py --csv data\house_prices.csv --target sale_price --goal "predict house sale price" >> tmp\live_output.txt 2>&1
echo LIVE_EXIT=%ERRORLEVEL% >> tmp\live_output.txt
echo FINISHED %DATE% %TIME% >> tmp\live_output.txt
exit /b
