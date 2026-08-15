@echo off
REM Launch the Smart Waste Segregation web demo on localhost.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
    call .venv\Scripts\activate.bat
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

if exist "models\best_model.pth" (
    echo Launching app with trained model...
    py app.py --checkpoint models\best_model.pth %*
) else (
    echo No trained model found. Launching in mock (preview) mode.
    py app.py --mock %*
)
endlocal
