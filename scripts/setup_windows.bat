@echo off
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Run scripts\start_viewer_windows.bat
pause
