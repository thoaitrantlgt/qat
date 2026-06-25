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

7. Run the baseline suite, summarize results, and generate a chart:

```powershell
.\scripts\run_baselines.cmd
```

## Notes

- The project uses Python 3.12.
- `.venv` is created with `--system-site-packages` so it can reuse the machine's existing PyTorch install.
- Put CIFAR data under `data/` or set `dataset.download: true` in the config.
- If you prefer PowerShell directly, use `powershell.exe -ExecutionPolicy Bypass -File .\scripts\setup.ps1`.
