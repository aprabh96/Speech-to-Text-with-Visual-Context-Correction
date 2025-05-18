@echo off
:: Step 2: Run the Speech-to-Text application
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Activating virtual environment...
call "%CD%\.venv\Scripts\activate.bat"

echo Running Speech-to-Text application...
python "%CD%\speech_to_text.py"

pause 