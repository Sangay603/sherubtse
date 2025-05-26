@echo off
echo Setting up Stock Market Forecast Dashboard...

REM Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or later.
    pause
    exit
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

echo Setup complete! Starting the dashboard...
python dashboard_app.py

pause 