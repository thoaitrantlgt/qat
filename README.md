# HMAQ-VLM

Captioning-focused hierarchical multi-agent mixed-precision QAT for a pretrained ViT-Small + projector + GPT-2 Small model. This cleanly replaces the previous image-classification pipeline and has no runtime dependency on Giathoai/VLM.

## Environment

- Python 3.12; production target PyTorch 2.11 / CUDA 12.8
- AMP FP16 defaults for a 16 GB GPU (micro-batch 4, accumulation 8)
- CPU synthetic tests require no downloads

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\python -m pytest -q
```

Resolve config and record reproducibility metadata:

```powershell
python -m hmaq_vlm.cli resolve-config --config configs/default.yaml --output artifacts/metrics/resolved_run.json
```

Build immutable Flickr30k manifests from supplied files:

```powershell
python -m hmaq_vlm.cli prepare-flickr --images D:\flickr30k-images --annotations D:\dataset_flickr30k.json --output artifacts/manifests/flickr30k --seed 11
```

Download the pinned COCO image-caption metadata and images, then preserve Karpathy validation/test while deriving policy-search from train/restval:

```powershell
python -m hmaq_vlm.cli prepare-coco --cache data/coco-karpathy --output artifacts/manifests/coco-karpathy --seed 11 --policy-fraction 0.10
```

Run the complete network-free acceptance pipeline (FP16 step, generation, calibrated QAT step, five searches, static export, and reports):

```powershell
python -m hmaq_vlm.cli acceptance-smoke --output artifacts/smoke --seed 11 --search-budget 4 --timing-budget 2
```

Production commands use the same manifest/config contracts:

```powershell
python -m hmaq_vlm.cli train-fp16 --config configs/default.yaml --manifests artifacts/manifests/coco-karpathy --output artifacts/checkpoints/fp16
python -m hmaq_vlm.cli profile --config configs/default.yaml --manifests artifacts/manifests/coco-karpathy --teacher artifacts/checkpoints/fp16/teacher_best.pt --output artifacts/profiles/sensitivity.json
python -m hmaq_vlm.cli search --config configs/default.yaml --manifests artifacts/manifests/coco-karpathy --teacher artifacts/checkpoints/fp16/teacher_best.pt --method hierarchical_mappo --output artifacts/policies/hmaq
python -m hmaq_vlm.cli train-qat --config configs/default.yaml --manifests artifacts/manifests/coco-karpathy --teacher artifacts/checkpoints/fp16/teacher_best.pt --policy artifacts/policies/selected.json --output artifacts/checkpoints/hmaq
python -m hmaq_vlm.cli evaluate --config configs/default.yaml --manifests artifacts/manifests/coco-karpathy --checkpoint artifacts/checkpoints/hmaq/static_qat.pt --split test --output artifacts/metrics/hmaq_test.json
```

COCO/pretrained/CUDA runs are integration workloads and use the exact revisions in `configs/default.yaml`. Quantizable groups expose all 16 Cartesian actions from `{2,4,8,16}²`; embeddings, normalization, Softmax, caption loss, and GPT-2 output head stay FP16. A 16-bit quantizer is an exact bypass.

Server timing is always stored as `server_fake_quant_ms` and marked diagnostic-only. It is not Jetson latency, energy, or evidence of native INT2/INT4 speedup. Real Jetson execution remains behind the versioned bundle contract.
