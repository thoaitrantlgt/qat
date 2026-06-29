#!/usr/bin/env bash
set -euo pipefail

EPOCHS=100
FINE_TUNE_EPOCHS=20
TRAIN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --fine-tune-epochs|-f)
            FINE_TUNE_EPOCHS="$2"
            shift 2
            ;;
        --train)
            TRAIN=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--epochs N] [--fine-tune-epochs N] [--train]"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
    PYTHON_EXE=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="python3"
else
    PYTHON_EXE="python"
fi

SUITE="cifar10_resnet20"
TABLE_ROOT="outputs/tables/$SUITE"
COMPARISON_CSV="$TABLE_ROOT/baselines.csv"
COMPARISON_MD="$TABLE_ROOT/comparison.md"
COMPARISON_JSON="$TABLE_ROOT/comparison_summary.json"
COMPARISON_PLOT="$TABLE_ROOT/comparison.png"
POLICY_PATH="outputs/policies/resnet20_cifar10_marl_qat_policy.json"
MARL_CHECKPOINT="outputs/checkpoints/resnet20_cifar10_marl_qat_best.pt"
STATIC_JSON="outputs/policies/resnet20_cifar10_static_policy_export.json"
STATIC_CSV="$TABLE_ROOT/static_policy_metrics.csv"
STATIC_CHECKPOINT="outputs/checkpoints/resnet20_cifar10_static_policy_finetuned.pt"

mkdir -p "$TABLE_ROOT"
rm -f "$COMPARISON_CSV"

run_fp32() {
    local config="$1"
    "$PYTHON_EXE" -m src.training.train_fp32 --config "$config" --epochs "$EPOCHS"
}

run_qat() {
    local config="$1"
    local warm_start="$2"
    local args=(-m src.training.train_qat_uniform --config "$config" --epochs "$EPOCHS")
    if [[ -f "$warm_start" ]]; then
        args+=(--fp32-checkpoint "$warm_start")
    else
        echo "warning: fp32 checkpoint not found at $warm_start; training QAT from scratch."
    fi
    "$PYTHON_EXE" "${args[@]}"
}

report_baseline() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    if [[ ! -f "$checkpoint" ]]; then
        echo "warning: missing checkpoint for $name: $checkpoint"
        return
    fi
    "$PYTHON_EXE" -m src.training.report_baseline \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --name "$name" \
        --csv "$COMPARISON_CSV" \
        --json "$TABLE_ROOT/$name.json"
}

FP32_CONFIG="configs/cifar10_resnet20_fp32.yaml"
W8A8_CONFIG="configs/cifar10_resnet20_qat_uniform.yaml"
W4A8_CONFIG="configs/cifar10_resnet20_qat_w4a8.yaml"
W4A4_CONFIG="configs/cifar10_resnet20_qat_w4a4.yaml"
FP32_CKPT="outputs/checkpoints/resnet20_cifar10_fp32_best.pt"
W8A8_CKPT="outputs/checkpoints/resnet20_cifar10_qat_best.pt"
W4A8_CKPT="outputs/checkpoints/resnet20_cifar10_qat_w4a8_best.pt"
W4A4_CKPT="outputs/checkpoints/resnet20_cifar10_qat_w4a4_best.pt"

if [[ "$TRAIN" -eq 1 || ! -f "$FP32_CKPT" ]]; then
    run_fp32 "$FP32_CONFIG"
fi
if [[ "$TRAIN" -eq 1 || ! -f "$W8A8_CKPT" ]]; then
    run_qat "$W8A8_CONFIG" "$FP32_CKPT"
fi
if [[ "$TRAIN" -eq 1 || ! -f "$W4A8_CKPT" ]]; then
    run_qat "$W4A8_CONFIG" "$FP32_CKPT"
fi
if [[ "$TRAIN" -eq 1 || ! -f "$W4A4_CKPT" ]]; then
    run_qat "$W4A4_CONFIG" "$FP32_CKPT"
fi

report_baseline "fp32" "$FP32_CONFIG" "$FP32_CKPT"
report_baseline "w8a8" "$W8A8_CONFIG" "$W8A8_CKPT"
report_baseline "w4a8" "$W4A8_CONFIG" "$W4A8_CKPT"
report_baseline "w4a4" "$W4A4_CONFIG" "$W4A4_CKPT"

"$PYTHON_EXE" -m src.training.train_marl_qat \
    --config configs/cifar10_resnet20_marl_qat.yaml \
    --epochs "$EPOCHS"

"$PYTHON_EXE" -m src.training.export_policy \
    --config configs/cifar10_resnet20_marl_qat.yaml \
    --policy "$POLICY_PATH" \
    --checkpoint "$MARL_CHECKPOINT" \
    --fine-tune-epochs "$FINE_TUNE_EPOCHS" \
    --output-json "$STATIC_JSON" \
    --output-csv "$STATIC_CSV" \
    --output-checkpoint "$STATIC_CHECKPOINT" \
    --comparison-csv "$COMPARISON_CSV" \
    --name multiagent_static

"$PYTHON_EXE" -m src.training.summarize_baselines \
    --csv "$COMPARISON_CSV" \
    --markdown "$COMPARISON_MD" \
    --json "$COMPARISON_JSON"
"$PYTHON_EXE" -m src.training.plot_baselines \
    --csv "$COMPARISON_CSV" \
    --output "$COMPARISON_PLOT"

echo "info: fair comparison CSV: $COMPARISON_CSV"
echo "info: fair comparison plot: $COMPARISON_PLOT"
