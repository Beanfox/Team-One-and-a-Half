# Team-One-and-a-Half (TrafficBot)

## Current Workflow

This repo is currently run as a **frontend-first (Node.js)** project.

- UI stack: React + TypeScript + Vite in `frontend/`
- No Python virtual environment is required for normal UI work

## Frontend Commands

From repo root:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
npm --prefix frontend run build
npm --prefix frontend run lint
```

## Backend (System Python, No venv)

The team backend workflow uses system Python (no virtual environment activation).

Install backend dependencies from repo root:

```bash
py -3.13 -m pip install -r requirements.txt
```

Run backend orchestrator:

```bash
py -3.13 main.py
```

Or use the one-click PowerShell launcher:

```powershell
.\run_backend.ps1
```

If script execution policy blocks it, run:

```powershell
pwsh -ExecutionPolicy Bypass -File .\run_backend.ps1
```

To skip dependency reinstall on repeated runs:

```powershell
.\run_backend.ps1 -SkipInstall
```

Notes:

- `main.py` writes live files: `data_bridge.json` and `traffic_metrics.csv`
- `torch` is optional in current code path (the app falls back to random actions if unavailable)