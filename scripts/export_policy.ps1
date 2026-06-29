param(
    [string]$Config = "configs/cifar10_resnet20_marl_qat.yaml",
    [string]$Policy = "outputs/policies/resnet20_cifar10_marl_qat_policy.json",
    [string]$Checkpoint = "",
    [int]$FineTuneEpochs = 0,
    [switch]$DryRun
)

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

$argsList = @("-m", "src.training.export_policy", "--config", $Config, "--policy", $Policy, "--fine-tune-epochs", "$FineTuneEpochs")
if ($Checkpoint) {
    $argsList += @("--checkpoint", $Checkpoint)
}
if ($DryRun) {
    $argsList += @("--dry-run")
}

& $pythonExe @argsList
