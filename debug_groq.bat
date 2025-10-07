@echo off
REM Groq API Diagnostic Script Runner
REM This script runs the Groq diagnostic tool to troubleshoot API issues

echo ====================================================================
echo GROQ API DIAGNOSTIC TOOL
echo ====================================================================
echo.
echo This will check your Groq installation and API configuration.
echo.
echo Checking Python installation...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please ensure Python is installed and in your PATH.
    echo.
    pause
    exit /b 1
)

echo Python found. Running diagnostic...
echo.

REM Run the diagnostic script
python debug_groq.py

REM Check the exit code
if %errorlevel% equ 0 (
    echo.
    echo ====================================================================
    echo DIAGNOSTIC COMPLETED SUCCESSFULLY
    echo ====================================================================
    echo Your Groq setup appears to be working correctly.
) else (
    echo.
    echo ====================================================================
    echo DIAGNOSTIC FOUND ISSUES
    echo ====================================================================
    echo Please follow the recommendations above to fix the problems.
)

echo.
echo Press any key to exit...
pause >nul
