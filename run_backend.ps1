param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

Write-Host 'TrafficBot backend launcher (no venv)' -ForegroundColor Cyan

$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$pythonCmd = $null

if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
    Write-Host "Using project venv Python: $pythonCmd" -ForegroundColor Green
} else {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Error "Neither project venv python nor 'py' launcher was found."
    }
    $pythonCmd = 'py -3.13'
    Write-Host "Using system Python launcher: $pythonCmd" -ForegroundColor Yellow
}

if (-not $SkipInstall) {
    Write-Host 'Installing/updating backend dependencies from requirements.txt...' -ForegroundColor Yellow
    if ($pythonCmd -eq 'py -3.13') {
        py -3.13 -m pip install -r requirements.txt
    } else {
        & $pythonCmd -m pip install -r requirements.txt
    }
}

Write-Host 'Starting backend orchestrator (evaluate.py --live-ui)...' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor DarkGray

if ($pythonCmd -eq 'py -3.13') {
    py -3.13 evaluate.py --live-ui
} else {
    & $pythonCmd evaluate.py --live-ui
}
