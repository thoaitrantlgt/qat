param(
    [string]$Config = "configs/cifar100_resnet32_fp32.yaml",
    [int]$Epochs = 0
)

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

$argsList = @("-m", "src.training.train_fp32", "--config", $Config)
if ($Epochs -gt 0) {
    $argsList += @("--epochs", "$Epochs")
}

& $pythonExe @argsList

