@echo off
setlocal
cd /d "%~dp0\.."
set LLM_MODE=offline
echo STARTED %DATE% %TIME% > tmp\check_output.txt
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider >> tmp\check_output.txt 2>&1
echo PYTEST_EXIT=%ERRORLEVEL% >> tmp\check_output.txt
cd ui
call npm run build >> ..\tmp\check_output.txt 2>&1
echo BUILD_EXIT=%ERRORLEVEL% >> ..\tmp\check_output.txt
cd ..
echo FINISHED %DATE% %TIME% >> tmp\check_output.txt
exit /b
