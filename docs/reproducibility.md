# Reproducibility

Each run records resolved config, seed, Git commit, dependency versions, CUDA/device metadata, exact source revisions, and checksums. JSONL manifests are immutable; JSON, caches, checkpoints, and metrics use atomic replacement. Checksum drift aborts loading.

Epochs 1–3 train only the projector. Epochs 4–10 unfreeze vision and language parameters with differential learning rates. The effective default batch is 32.
