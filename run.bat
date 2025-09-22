@echo off
title Professional Speech-to-Text Application - Psynect Corp
echo ====================================================
echo Professional Speech-to-Text GUI Application
echo Psynect Corp - www.psynect.ai
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python detected successfully.
echo.

REM Create data directory structure
if not exist "data" mkdir data
if not exist "data\screenshots" mkdir data\screenshots
if not exist "data\recordings" mkdir data\recordings

REM Check if this is first run (no dependencies installed)
python -c "import numpy, openai, tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo First run detected - installing dependencies...
    echo.
    
    REM Upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo.
    
    echo Installing core dependencies...
    pip install numpy>=1.21.0 Pillow>=9.0.0 pyautogui>=0.9.50
    
    echo Installing AI service dependencies...
    pip install openai>=1.0.0 groq>=0.4.0 anthropic>=0.7.0
    
    echo Installing audio dependencies...
    pip install pyaudio>=0.2.11 soundfile>=0.12.0
    
    echo Installing system integration dependencies...
    pip install keyboard>=0.13.5 pyperclip>=1.8.0 python-dotenv>=0.19.0
    pip install mss>=6.1.0 pywin32>=300 websockets>=10.0
    
    echo Installing optional dependencies...
    pip install mouse>=0.7.0 pystray>=0.19.0
    
    echo.
    echo Dependencies installed successfully!
    echo.
)

REM Check for .env file and create template if needed
if not exist "data\.env" (
    echo Creating .env template file...
    echo # Professional Speech-to-Text Configuration > data\.env
    echo # Add your API keys below: >> data\.env
    echo OPENAI_API_KEY=your_openai_api_key_here >> data\.env
    echo GROQ_API_KEY=your_groq_api_key_here >> data\.env
    echo ANTHROPIC_API_KEY=your_anthropic_api_key_here >> data\.env
    echo. >> data\.env
    echo # Get your API keys from: >> data\.env
    echo # OpenAI: https://platform.openai.com/api-keys >> data\.env
    echo # Groq: https://console.groq.com/keys >> data\.env
    echo # Anthropic: https://console.anthropic.com/ >> data\.env
    echo.
    echo INFO: .env template created in data folder. You can add your API keys there,
    echo or configure them in the application settings.
    echo.
)

echo Starting Professional Speech-to-Text GUI...
echo.

REM Run the GUI application
python data/speech_to_text_gui.py

REM Only show error if it's a critical error (not clipboard issues)
if %errorlevel% neq 0 (
    echo.
    echo ====================================================
    echo Application exited unexpectedly.
    echo Check the error messages above for details.
    echo Note: Clipboard errors are non-critical and safe to ignore.
    echo ====================================================
    pause
)

echo.
echo Application closed normally.
pause
