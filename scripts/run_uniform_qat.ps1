param(
    [string]$Config = "configs/cifar100_resnet32_qat_uniform.yaml",
    [string]$Fp32Checkpoint = "",
    [int]$Epochs = 0
)

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

$argsList = @("-m", "src.training.train_qat_uniform", "--config", $Config)
if ($Fp32Checkpoint -and (Test-Path -LiteralPath $Fp32Checkpoint)) {
    $argsList += @("--fp32-checkpoint", $Fp32Checkpoint)
} elseif ($Fp32Checkpoint) {
    Write-Host "warning: fp32 checkpoint not found at $Fp32Checkpoint; training QAT from scratch."
}
if ($Epochs -gt 0) {
    $argsList += @("--epochs", "$Epochs")
}

& $pythonExe @argsList
