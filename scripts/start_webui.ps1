param(
    [switch]$SkipInstall,
    [switch]$ForceCpu,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

$rootDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPython = Join-Path $rootDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host ".venv was not found. Run run.bat first." -ForegroundColor Red
    exit 1
}

Set-Location $rootDir
$env:PYTHONPATH = $rootDir
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:32"
$env:CUDNN_LOGINFO_DBG = "0"
$env:PYTHONWARNINGS = "ignore::UserWarning:streamlit"

if ($ForceCpu) {
    $env:CHATTERBOX_DEVICE = "cpu"
}

if (-not $env:CHATTERBOX_CFG_WEIGHT) {
    $env:CHATTERBOX_CFG_WEIGHT = "0.2"
}

if (-not $env:CHATTERBOX_CHUNK_THRESHOLD) {
    $env:CHATTERBOX_CHUNK_THRESHOLD = "800"
}

$venvCudnnPath = Join-Path $rootDir ".venv\Lib\site-packages\nvidia\cudnn\bin"
if (Test-Path $venvCudnnPath) {
    $env:PATH = "$venvCudnnPath;$env:PATH"
}
elseif ($env:CONDA_PREFIX) {
    $condaCudnnPath = Join-Path $env:CONDA_PREFIX "Lib\site-packages\nvidia\cudnn\bin"
    if (Test-Path $condaCudnnPath) {
        $env:PATH = "$condaCudnnPath;$env:PATH"
    }
}

Write-Step "Starting WebUI"
Write-Host "URL: http://127.0.0.1:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C in this window to stop the server." -ForegroundColor Yellow

& $venvPython -m streamlit run .\webui\Main.py --browser.gatherUsageStats=False --server.enableCORS=True
exit $LASTEXITCODE
