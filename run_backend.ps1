param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

Write-Host 'TrafficBot backend launcher (no venv)' -ForegroundColor Cyan

$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python launcher 'py' was not found. Install Python 3.13 and ensure 'py' is available."
}

if (-not $SkipInstall) {
    Write-Host 'Installing/updating backend dependencies from requirements.txt...' -ForegroundColor Yellow
    py -3.13 -m pip install -r requirements.txt
}

Write-Host 'Starting backend orchestrator (main.py)...' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor DarkGray

py -3.13 main.py
