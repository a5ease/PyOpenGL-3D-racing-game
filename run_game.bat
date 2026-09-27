@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" main.py
echo.
echo Game exited. Press any key to close...
pause >nul
