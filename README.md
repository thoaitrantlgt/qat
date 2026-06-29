# qat-agent

Multi-agent adaptive QAT for CNNs.

## Setup

1. Run `.\scripts\setup.ps1`
1. On a default Windows shell, run `scripts\setup.cmd` instead
2. Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Run the FP32 baseline:

```powershell
.\scripts\run_fp32.cmd -Config configs/cifar100_resnet32_fp32.yaml -Epochs 1
```

4. Run the uniform QAT baseline:

```powershell
.\scripts\run_uniform_qat.cmd -Config configs/cifar100_resnet32_qat_uniform.yaml -Epochs 1
```

5. Run the low-bit warm-start QAT baseline:

```powershell
.\scripts\run_uniform_qat.cmd -Config configs/cifar100_resnet32_qat_w4a8.yaml -Epochs 1
```

6. Run the CIFAR-10 QAT baseline:

```powershell
.\scripts\run_uniform_qat.cmd -Config configs/cifar10_resnet20_qat_uniform.yaml -Epochs 1
```

7. Run the baseline suite, summarize results, and generate a chart. The script will train any missing checkpoints first, then write the CSV/Markdown summary and chart:

```powershell
.\scripts\run_baselines.cmd
```

For a fair CIFAR-10/ResNet20 baseline suite matching the multi-agent config:

```powershell
.\scripts\run_baselines.cmd -Suite cifar10_resnet20 -Epochs 100
```

8. Run adaptive multi-agent QAT:

```powershell
.\scripts\run_marl_qat.cmd -Config configs/cifar10_resnet20_marl_qat.yaml -Epochs 100
```

9. Export and optionally fine-tune the static mixed-precision policy:

```powershell
.\scripts\export_policy.cmd -Config configs/cifar10_resnet20_marl_qat.yaml -Policy outputs/policies/resnet20_cifar10_marl_qat_policy.json -Checkpoint outputs/checkpoints/resnet20_cifar10_marl_qat_best.pt -FineTuneEpochs 20
```

10. Run the full fair comparison end-to-end:

```powershell
.\scripts\run_fair_comparison.cmd
```

This writes the combined comparison table and chart under `outputs/tables/cifar10_resnet20/`.

## Notes

- The project uses Python 3.12.
- `.venv` is created with `--system-site-packages` so it can reuse the machine's existing PyTorch install.
- Put CIFAR data under `data/` or set `dataset.download: true` in the config.
- If you prefer PowerShell directly, use `powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup.ps1`.
