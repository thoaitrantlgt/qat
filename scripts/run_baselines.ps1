param(
    [ValidateSet("cifar100_resnet32", "cifar10_resnet20")]
    [string]$Suite = "cifar100_resnet32",
    [int]$Epochs = 0,
    [switch]$Train,
    [switch]$SkipFP32
)

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

if ($Suite -eq "cifar10_resnet20") {
    $baselines = @(
        @{
            Name = "fp32"
            Config = "configs/cifar10_resnet20_fp32.yaml"
            Checkpoint = "outputs/checkpoints/resnet20_cifar10_fp32_best.pt"
            TrainCmd = "scripts/run_fp32.cmd"
            DefaultWarmStart = ""
        },
        @{
            Name = "w8a8"
            Config = "configs/cifar10_resnet20_qat_uniform.yaml"
            Checkpoint = "outputs/checkpoints/resnet20_cifar10_qat_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet20_cifar10_fp32_best.pt"
        },
        @{
            Name = "w4a8"
            Config = "configs/cifar10_resnet20_qat_w4a8.yaml"
            Checkpoint = "outputs/checkpoints/resnet20_cifar10_qat_w4a8_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet20_cifar10_fp32_best.pt"
        },
        @{
            Name = "w4a4"
            Config = "configs/cifar10_resnet20_qat_w4a4.yaml"
            Checkpoint = "outputs/checkpoints/resnet20_cifar10_qat_w4a4_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet20_cifar10_fp32_best.pt"
        }
    )
} else {
    $baselines = @(
        @{
            Name = "fp32"
            Config = "configs/cifar100_resnet32_fp32.yaml"
            Checkpoint = "outputs/checkpoints/resnet32_cifar100_fp32_best.pt"
            TrainCmd = "scripts/run_fp32.cmd"
            DefaultWarmStart = ""
        },
        @{
            Name = "w8a8"
            Config = "configs/cifar100_resnet32_qat_uniform.yaml"
            Checkpoint = "outputs/checkpoints/resnet32_cifar100_qat_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet32_cifar100_fp32_best.pt"
        },
        @{
            Name = "w4a8"
            Config = "configs/cifar100_resnet32_qat_w4a8.yaml"
            Checkpoint = "outputs/checkpoints/resnet32_cifar100_qat_w4a8_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet32_cifar100_fp32_best.pt"
        },
        @{
            Name = "w4a4"
            Config = "configs/cifar100_resnet32_qat_w4a4.yaml"
            Checkpoint = "outputs/checkpoints/resnet32_cifar100_qat_w4a4_best.pt"
            TrainCmd = "scripts/run_uniform_qat.cmd"
            DefaultWarmStart = "outputs/checkpoints/resnet32_cifar100_fp32_best.pt"
        }
    )
}

$tableRoot = "outputs/tables/$Suite"
$csvPath = Join-Path $tableRoot "baselines.csv"
$markdownPath = Join-Path $tableRoot "baselines.md"
$summaryJsonPath = Join-Path $tableRoot "baselines_summary.json"
$plotPath = Join-Path $tableRoot "baselines_comparison.png"
New-Item -ItemType Directory -Force -Path $tableRoot | Out-Null
if (Test-Path -LiteralPath $csvPath) {
    Remove-Item -LiteralPath $csvPath -Force
}

if ($Train) {
    Write-Host "info: retraining all $Suite baselines before reporting."
} else {
    $missingBaselines = @()
    foreach ($baseline in $baselines) {
        if ($SkipFP32 -and $baseline.Name -eq "fp32") {
            continue
        }
        if (-not (Test-Path -LiteralPath $baseline.Checkpoint)) {
            $missingBaselines += $baseline.Name
        }
    }
    if ($missingBaselines.Count -gt 0) {
        Write-Host "info: missing $Suite checkpoints detected, training: $($missingBaselines -join ', ')"
    }
}

foreach ($baseline in $baselines) {
    if ($SkipFP32 -and $baseline.Name -eq "fp32") {
        continue
    }

    $needsTraining = $Train -or -not (Test-Path -LiteralPath $baseline.Checkpoint)
    if (-not $needsTraining) {
        continue
    }

    if ($baseline.TrainCmd -like "*run_fp32.cmd") {
        $args = @("-Config", $baseline.Config)
        if ($Epochs -gt 0) {
            $args += @("-Epochs", "$Epochs")
        }
        & (Join-Path $PSScriptRoot "run_fp32.cmd") @args
    } else {
        $args = @("-Config", $baseline.Config)
        if ($baseline.DefaultWarmStart) {
            $args += @("-Fp32Checkpoint", $baseline.DefaultWarmStart)
        }
        if ($Epochs -gt 0) {
            $args += @("-Epochs", "$Epochs")
        }
        & (Join-Path $PSScriptRoot "run_uniform_qat.cmd") @args
    }
}

foreach ($baseline in $baselines) {
    if (-not (Test-Path -LiteralPath $baseline.Checkpoint)) {
        Write-Host "warning: missing checkpoint for $($baseline.Name): $($baseline.Checkpoint)"
        continue
    }

    $jsonPath = Join-Path $tableRoot "$($baseline.Name).json"
    & $pythonExe -m src.training.report_baseline --config $baseline.Config --checkpoint $baseline.Checkpoint --name $baseline.Name --csv $csvPath --json $jsonPath
}

& $pythonExe -m src.training.summarize_baselines --csv $csvPath --markdown $markdownPath --json $summaryJsonPath
& $pythonExe -m src.training.plot_baselines --csv $csvPath --output $plotPath

Write-Host "info: wrote $Suite baseline table to $csvPath"
