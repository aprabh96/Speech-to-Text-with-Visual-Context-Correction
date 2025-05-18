# Get the directory where the script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptDir

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host "Checking files in the directory:" -ForegroundColor Cyan
Get-ChildItem

# Check for virtual environment
Write-Host "Checking for virtual environment..." -ForegroundColor Cyan

if (-not (Test-Path "$scriptDir\venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv "$scriptDir\venv"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        Write-Host "Please ensure Python is installed and in your PATH." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if requirements.txt exists
if (-not (Test-Path "$scriptDir\requirements.txt")) {
    Write-Host "requirements.txt not found. Creating it..." -ForegroundColor Yellow
    @"
groq
pynput
pyaudio
sounddevice
soundfile
numpy
pyperclip
python-dotenv
"@ | Out-File -FilePath "$scriptDir\requirements.txt" -Encoding utf8
    Write-Host "Created requirements.txt file." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& "$scriptDir\venv\Scripts\Activate.ps1"

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
pip install -r "$scriptDir\requirements.txt"

# Check if the application file exists
if (-not (Test-Path "$scriptDir\speech_to_text.py")) {
    Write-Host "ERROR: speech_to_text.py file not found in $scriptDir" -ForegroundColor Red
    Write-Host "Please make sure all required files are in the same directory as this script." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run application
Write-Host "Running Speech-to-Text application..." -ForegroundColor Cyan
python "$scriptDir\speech_to_text.py"

Read-Host "Press Enter to exit" 