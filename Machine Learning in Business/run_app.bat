@echo off
cd /d "%~dp0"

echo Checking for virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies if needed...
pip install --upgrade pip
pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run app.py

pause