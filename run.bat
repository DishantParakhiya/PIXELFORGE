@echo off
cd /d "%~dp0"
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"
echo Installing dependencies...
pip install -r requirements.txt
echo Starting PixelForge...
streamlit run app.py
pause
