@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
title Leonardo Visual Demos

rem Prefer the project virtual environment. Falling through to a bare "python"
rem picks up whatever interpreter is on PATH, which will not have the packages
rem installed by setup (PyTorch in particular) and produces confusing failures.
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo No .venv found. Creating one...
  where py >nul 2>nul
  if not errorlevel 1 ( py -3 -m venv .venv ) else ( python -m venv .venv )
)
if not exist "%PYTHON%" (
  echo.
  echo Could not create a Python environment.
  echo Install 64-bit Python 3.10 or newer, then run this file again.
  pause
  exit /b 1
)

"%PYTHON%" -c "import numpy, PIL, fastapi, uvicorn, pydantic" >nul 2>nul
if errorlevel 1 (
  echo Installing requirements. This only happens once.
  "%PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

echo.
echo Starting the local backend and frontend together...
echo Keep this window open while using the viewer.
echo The browser will open automatically at the local address printed below.
echo.
"%PYTHON%" app.py
pause
