# publish.ps1 - push the project to GitHub (first run creates the branch upstream).
#
#   powershell -ExecutionPolicy Bypass -File publish.ps1
#
# Prerequisite (once): create an empty repository named btc-ml-trader at
# https://github.com/new (private is fine, do NOT add a README).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    git remote add origin https://github.com/adired94-bit/btc-ml-trader.git
    $remote = "https://github.com/adired94-bit/btc-ml-trader.git"
}
Write-Host "[publish] remote: $remote"

$status = git status --porcelain
if ($status) {
    Write-Host "[publish] committing local changes"
    git add -A
    git commit -m "Update $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[publish] push failed. Most likely the repository does not exist yet:"
    Write-Host "          1. open https://github.com/new"
    Write-Host "          2. name: btc-ml-trader, leave everything else empty, click Create"
    Write-Host "          3. run this script again"
    exit 1
}

Write-Host ""
Write-Host "[publish] done. Next (once): deploy the dashboard for free at https://share.streamlit.io"
Write-Host "          Create app -> repo adired94-bit/btc-ml-trader -> branch main -> file app.py -> Deploy"
Write-Host "          Every future run of this script updates the live app automatically."
