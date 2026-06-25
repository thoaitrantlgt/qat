param(
    [string]$Python = "C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.12 not found at: $Python"
}

if (-not (Test-Path -LiteralPath ".venv")) {
    & $Python -m venv .venv --system-site-packages
}

$venvPython = Join-Path (Resolve-Path .venv).Path "Scripts\\python.exe"
& $venvPython -c "import sys; print(sys.version)"
& $venvPython -c "import torch, yaml, numpy; print('torch', torch.__version__); print('yaml', yaml.__version__); print('numpy', numpy.__version__)"
& $venvPython -m src.training.train_fp32 --config configs/cifar100_resnet32_fp32.yaml --dry-run

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run baseline with: .\scripts\run_fp32.ps1"
