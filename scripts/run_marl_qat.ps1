param(
    [string]$Config = "configs/cifar10_resnet20_marl_qat.yaml",
    [string]$Fp32Checkpoint = "",
    [int]$Epochs = 0,
    [switch]$DryRun
)

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

$argsList = @("-m", "src.training.train_marl_qat", "--config", $Config)
if ($Fp32Checkpoint -and (Test-Path -LiteralPath $Fp32Checkpoint)) {
    $argsList += @("--fp32-checkpoint", $Fp32Checkpoint)
} elseif ($Fp32Checkpoint) {
    Write-Host "warning: fp32 checkpoint not found at $Fp32Checkpoint; training adaptive QAT from scratch."
}
if ($Epochs -gt 0) {
    $argsList += @("--epochs", "$Epochs")
}
if ($DryRun) {
    $argsList += @("--dry-run")
}

& $pythonExe @argsList
