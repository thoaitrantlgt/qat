param(
    [string]$Config = "configs/cifar100_resnet32_fp32.yaml",
    [string]$Checkpoint = "outputs/checkpoints/resnet32_cifar100_fp32_best.pt",
    [string]$Output = "outputs/logs/fp32_cifar100_eval.json"
)

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

& $pythonExe -m src.training.evaluate --config $Config --checkpoint $Checkpoint --output $Output

