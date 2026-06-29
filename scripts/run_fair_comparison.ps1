param(
    [int]$Epochs = 1,
    [int]$FineTuneEpochs = 0,
    [switch]$Train
)

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot

$pythonExe = if (Test-Path -LiteralPath ".venv\\Scripts\\python.exe") {
    ".venv\\Scripts\\python.exe"
} else {
    "python"
}

$suite = "cifar10_resnet20"
$tableRoot = "outputs/tables/$suite"
$comparisonCsv = Join-Path $tableRoot "baselines.csv"
$comparisonMd = Join-Path $tableRoot "comparison.md"
$comparisonJson = Join-Path $tableRoot "comparison_summary.json"
$comparisonPlot = Join-Path $tableRoot "comparison.png"
$policyPath = "outputs/policies/resnet20_cifar10_marl_qat_policy.json"
$marlCheckpoint = "outputs/checkpoints/resnet20_cifar10_marl_qat_best.pt"
$staticJson = "outputs/policies/resnet20_cifar10_static_policy_export.json"
$staticCsv = "outputs/tables/$suite/static_policy_metrics.csv"
$staticCheckpoint = "outputs/checkpoints/resnet20_cifar10_static_policy_finetuned.pt"

$baselineArgs = @("-Suite", $suite, "-Epochs", "$Epochs")
if ($Train) {
    $baselineArgs += "-Train"
}
& (Join-Path $PSScriptRoot "run_baselines.cmd") @baselineArgs

& (Join-Path $PSScriptRoot "run_marl_qat.cmd") -Config "configs/cifar10_resnet20_marl_qat.yaml" -Epochs $Epochs

& $pythonExe -m src.training.export_policy `
    --config configs/cifar10_resnet20_marl_qat.yaml `
    --policy $policyPath `
    --checkpoint $marlCheckpoint `
    --fine-tune-epochs $FineTuneEpochs `
    --output-json $staticJson `
    --output-csv $staticCsv `
    --output-checkpoint $staticCheckpoint `
    --comparison-csv $comparisonCsv `
    --name multiagent_static

& $pythonExe -m src.training.summarize_baselines --csv $comparisonCsv --markdown $comparisonMd --json $comparisonJson
& $pythonExe -m src.training.plot_baselines --csv $comparisonCsv --output $comparisonPlot

Write-Host "info: fair comparison CSV: $comparisonCsv"
Write-Host "info: fair comparison plot: $comparisonPlot"
