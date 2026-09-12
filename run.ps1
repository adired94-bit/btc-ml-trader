# run.ps1 - start the FastAPI backend and the Streamlit dashboard together (Windows).
#
#   powershell -ExecutionPolicy Bypass -File run.ps1
#
# Trains the models first if no artifacts exist yet.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "venv not found. Create it with: python -m venv venv; venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path (Join-Path $PSScriptRoot "models\direction_ensemble.joblib"))) {
    Write-Host "[run] No trained models found - training now (about a minute)..."
    & $py -m src.models.train
}

$apiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$uiPort = if ($env:DASHBOARD_PORT) { $env:DASHBOARD_PORT } else { "8501" }

Write-Host "[run] starting API on http://127.0.0.1:$apiPort"
$api = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", $apiPort -PassThru -NoNewWindow

Start-Sleep -Seconds 3
Write-Host "[run] starting dashboard on http://localhost:$uiPort"
$ui = Start-Process -FilePath $py -ArgumentList "-m", "streamlit", "run", "app.py", "--server.port", $uiPort, "--server.headless", "true" -PassThru -NoNewWindow

Write-Host "[run] API PID $($api.Id), dashboard PID $($ui.Id). Press Ctrl+C to stop both."
try {
    Wait-Process -Id $api.Id, $ui.Id
} finally {
    foreach ($p in @($api, $ui)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
