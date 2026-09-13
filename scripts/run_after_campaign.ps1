# Waits for any running scripts/improve.py process to finish, then runs the volatility experiment.
#   powershell -ExecutionPolicy Bypass -File scripts\run_after_campaign.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = (Resolve-Path "venv\Scripts\python.exe").Path
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*improve.py*' }
    if (-not $running) { break }
    Start-Sleep -Seconds 60
}
& $py scripts/volatility_target.py --long *> logs\volatility_target.log
"VOL_EXIT=$LASTEXITCODE" | Out-File -Append logs\volatility_target.log
