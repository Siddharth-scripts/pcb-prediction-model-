@echo off
echo PCB Component Detection Backend
echo ================================
cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating venv...
call venv\Scripts\activate.bat

if not exist "venv\Lib\site-packages\fastapi" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting server at http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.
python app.py
pause
