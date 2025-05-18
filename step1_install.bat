@echo off
:: Step 1: Install dependencies and set up the environment
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Checking for virtual environment...
if not exist "%CD%\.venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv "%CD%\.venv"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        echo Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call "%CD%\.venv\Scripts\activate.bat"

echo Installing required packages...
if not exist "%CD%\requirements.txt" (
    echo requirements.txt not found. Creating it...
    (
        echo openai>=1.76.0
        echo keyboard
        echo pyaudio
        echo sounddevice
        echo soundfile
        echo numpy
        echo pyperclip
        echo python-dotenv
        echo pillow
        echo pyautogui
        echo mss
        echo pywin32
        echo websockets>=10.1
        echo # Optional dependencies (fallbacks)
        echo groq
        echo anthropic>=0.19.1
    ) > "%CD%\requirements.txt"
)
pip install -r "%CD%\requirements.txt"

echo Installation complete. You can now run step2_run.bat (Step 2).
pause 