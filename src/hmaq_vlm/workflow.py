from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hmaq_vlm.config import ExperimentConfig, load_config
from hmaq_vlm.data import CaptionCollator, CaptionImage
from hmaq_vlm.hardware import MeasurementProtocol, ServerTimingBackend
from hmaq_vlm.losses import consistency_losses
from hmaq_vlm.models.pretrained import load_pretrained_vlm
from hmaq_vlm.profiling import SensitivityProfiler
from hmaq_vlm.quantization import ACTION_SPACE, MixedPrecisionPolicy, PrecisionAction, build_quant_group_registry, inject_quantizers, temporary_policy
from hmaq_vlm.quantization.export import export_static_checkpoint, load_static_checkpoint
from hmaq_vlm.reporting import build_result_artifacts, evaluate_captions
from hmaq_vlm.reproducibility import atomic_torch_save, atomic_write_json, collect_run_metadata, seed_everything, stable_hash
from hmaq_vlm.search import SearchEnvironment, run_search
from hmaq_vlm.training import build_optimizer, calibrate_quantizers, set_trainable_stage


class _TimmProcessor:
    def __init__(self, vision_model) -> None:
        from timm.data import create_transform, resolve_model_data_config

        self.transform = create_transform(**resolve_model_data_config(vision_model), is_training=False)

    def __call__(self, *, images, return_tensors: str):
        return {"pixel_values": torch.stack([self.transform(image) for image in images])}


def _components(config: ExperimentConfig):
    from transformers import AutoTokenizer

    model = load_pretrained_vlm(config.model)
    tokenizer = AutoTokenizer.from_pretrained(config.model.language_model, revision=config.model.language_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, _TimmProcessor(model.vision_encoder)


def _records(root: str | Path, split: str) -> list[CaptionImage]:
    path = Path(root) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest split not found: {path}")
    return [CaptionImage(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def _loader(records: list[CaptionImage], processor, tokenizer, config: ExperimentConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(records, batch_size=config.train.micro_batch_size, shuffle=shuffle, generator=generator, num_workers=config.data.workers, collate_fn=CaptionCollator(processor, tokenizer), pin_memory=torch.cuda.is_available())


def _device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _train_epoch(model, loader, optimizer, config: ExperimentConfig, device: torch.device, teacher=None) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    total_steps = len(loader)
    for step, raw in enumerate(loader, 1):
        batch = _device_batch(raw, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(**batch)
            loss = output.loss
            if teacher is not None:
                teacher.eval()
                with torch.no_grad():
                    reference = teacher(**batch)
                prefix, kl = consistency_losses(output.visual_prefix, reference.visual_prefix, output.logits[:, :-1], reference.logits[:, :-1], 2.0, output.labels[:, 1:].ne(-100))
                loss = loss + prefix + kl
            scaled = loss / config.train.gradient_accumulation
        if not bool(torch.isfinite(scaled)):
            raise FloatingPointError("non-finite training loss; precision policy rejected")
        scaled.backward()
        if step % config.train.gradient_accumulation == 0 or step == total_steps:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite gradient norm; precision policy rejected")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def _caption_metrics(model, records: list[CaptionImage], processor, tokenizer, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    references, hypotheses = {}, {}
    for start in range(0, len(records), config.train.micro_batch_size):
        chunk = records[start : start + config.train.micro_batch_size]
        images = []
        from PIL import Image

        for item in chunk:
            images.append(Image.open(item.image_path).convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        prompt = torch.full((len(chunk), 1), tokenizer.bos_token_id or tokenizer.eos_token_id, dtype=torch.long, device=device)
        generated = model.generate(pixels, prompt, eos_token_id=tokenizer.eos_token_id, max_new_tokens=config.model.max_new_tokens)
        captions = tokenizer.batch_decode(generated[:, 1:], skip_special_tokens=True)
        for item, caption in zip(chunk, captions, strict=True):
            references[item.image_id] = list(item.captions)
            hypotheses[item.image_id] = caption.strip()
    return evaluate_captions(references, hypotheses)


def _policy(path: str | Path) -> MixedPrecisionPolicy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return MixedPrecisionPolicy({name: PrecisionAction(value["weight_bits"], value["activation_bits"]) for name, value in raw.items()})


def train_fp16(config_path: str | Path, manifests: str | Path, output: str | Path) -> Path:
    config = load_config(config_path)
    seed_everything(config.seed)
    model, tokenizer, processor = _components(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = _loader(_records(manifests, "train"), processor, tokenizer, config, shuffle=True)
    validation = _records(manifests, "validation")
    destination = Path(output)
    best_cider = float("-inf")
    best_path = destination / "teacher_best.pt"
    optimizer = None
    history = []
    for epoch in range(1, config.train.epochs + 1):
        set_trainable_stage(model, epoch, config.train.projector_warmup_epochs)
        if optimizer is None or epoch == config.train.projector_warmup_epochs + 1:
            optimizer = build_optimizer(model, projector_lr=config.train.projector_lr, vision_lr=config.train.vision_lr, language_lr=config.train.language_lr)
        train_loss = _train_epoch(model, train_loader, optimizer, config, device)
        metrics = _caption_metrics(model, validation, processor, tokenizer, config, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if metrics["cider"] > best_cider:
            best_cider = metrics["cider"]
            atomic_torch_save(best_path, {"schema_version": "1.0", "model_state": model.state_dict(), "validation_cider": best_cider, "epoch": epoch, "config": config.to_dict()})
    atomic_write_json(destination / "fp16_history.json", history)
    return best_path


def train_qat(config_path: str | Path, manifests: str | Path, teacher_checkpoint: str | Path, policy_path: str | Path, output: str | Path) -> Path:
    config = load_config(config_path)
    seed_everything(config.seed)
    student, tokenizer, processor = _components(config)
    teacher, _, _ = _components(config)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(checkpoint["model_state"])
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.requires_grad_(False)
    registry = build_quant_group_registry(student)
    policy = _policy(policy_path)
    inject_quantizers(student, registry, policy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)
    train_loader = _loader(_records(manifests, "train"), processor, tokenizer, config, shuffle=True)
    calibration = []
    for index, batch in enumerate(train_loader):
        calibration.append(_device_batch(batch, device))
        if index == 7:
            break
    calibrate_quantizers(student, calibration, lambda module, item: module(**item))
    set_trainable_stage(student, config.train.projector_warmup_epochs + 1, config.train.projector_warmup_epochs)
    optimizer = build_optimizer(student, projector_lr=config.train.projector_lr, vision_lr=config.train.vision_lr, language_lr=config.train.language_lr)
    history = []
    validation = _records(manifests, "validation")
    best_cider = float("-inf")
    destination = Path(output)
    checkpoint_path = destination / "static_qat.pt"
    for epoch in range(1, config.train.epochs + 1):
        train_loss = _train_epoch(student, train_loader, optimizer, config, device, teacher)
        metrics = _caption_metrics(student, validation, processor, tokenizer, config, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if metrics["cider"] > best_cider:
            best_cider = metrics["cider"]
            export_static_checkpoint(checkpoint_path, student, policy, {"seed": config.seed, "teacher_checkpoint": str(teacher_checkpoint), "validation_cider": best_cider, "epoch": epoch}, registry)
    atomic_write_json(destination / "qat_history.json", history)
    return checkpoint_path


def evaluate_checkpoint(config_path: str | Path, manifests: str | Path, checkpoint_path: str | Path, split: str, output: str | Path) -> dict[str, float]:
    config = load_config(config_path)
    model, tokenizer, processor = _components(config)
    payload = load_static_checkpoint(checkpoint_path)
    if "policy" in payload:
        registry = build_quant_group_registry(model)
        policy = MixedPrecisionPolicy({name: PrecisionAction(value["weight_bits"], value["activation_bits"]) for name, value in payload["policy"].items()})
        inject_quantizers(model, registry, policy)
    model.load_state_dict(payload["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = _caption_metrics(model, _records(manifests, split), processor, tokenizer, config, device)
    atomic_write_json(output, {"schema_version": "1.0", "split": split, **metrics})
    return metrics


def search_policies(config_path: str | Path, manifests: str | Path, teacher_checkpoint: str | Path, method: str, output: str | Path) -> Path:
    config = load_config(config_path)
    model, tokenizer, processor = _components(config)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    registry = build_quant_group_registry(model)
    records = _records(manifests, "policy_search")
    loader = _loader(records, processor, tokenizer, config, shuffle=False)
    batches = [_device_batch(batch, device) for batch in loader]
    if not batches:
        raise ValueError("policy-search manifest is empty")
    model.eval()
    with torch.no_grad():
        teacher_outputs = [model(**batch) for batch in batches]

    def costs(policy: MixedPrecisionPolicy) -> tuple[float, float]:
        total = sum(group.parameters for group in registry)
        bitops = sum(group.parameters * policy.actions[group.name].weight_bits * policy.actions[group.name].activation_bits for group in registry) / (total * 256)
        size = sum(group.parameters * policy.actions[group.name].weight_bits for group in registry) / (total * 16)
        return bitops, size

    def evaluator(policy: MixedPrecisionPolicy) -> dict[str, float]:
        with temporary_policy(model, registry, policy):
            calibrate_quantizers(model, batches[: min(8, len(batches))], lambda module, item: module(**item))
            prefixes, kls = [], []
            with torch.no_grad():
                for batch, reference in zip(batches, teacher_outputs, strict=True):
                    student = model(**batch)
                    prefix, kl = consistency_losses(student.visual_prefix, reference.visual_prefix, student.logits[:, :-1], reference.logits[:, :-1], 2.0, student.labels[:, 1:].ne(-100))
                    prefixes.append(float(prefix))
                    kls.append(float(kl))
            caption_metrics = _caption_metrics(model, records, processor, tokenizer, config, device)
            bitops, size = costs(policy)
            return {"cider": caption_metrics["cider"], "bitops_ratio": bitops, "model_size_ratio": size, "prefix_distortion": sum(prefixes) / len(prefixes), "logit_kl": sum(kls) / len(kls)}

    backend = ServerTimingBackend()

    def timing(policy: MixedPrecisionPolicy) -> dict[str, object]:
        with temporary_policy(model, registry, policy):
            calibrate_quantizers(model, batches[:1], lambda module, item: module(**item))
            return backend.measure(lambda: model(**batches[0]).loss, MeasurementProtocol()).to_dict()

    destination = Path(output)
    environment = SearchEnvironment([group.name for group in registry], evaluator, timing_evaluator=timing, cache_path=destination / "candidate_cache.json", model_hash=stable_hash({name: tuple(value.shape) for name, value in model.state_dict().items()}), dataset_hash=stable_hash([item.image_id for item in records]), runtime_hash=f"torch-{torch.__version__}", protocol_hash=stable_hash(config.search.__dict__))
    run = run_search(method, environment, budget=config.search.candidate_budget, timing_budget=config.search.timing_budget, seed=config.seed)
    path = destination / f"{method}_audit.json"
    atomic_write_json(path, {"schema_version": "1.0", "method": method, "candidates": [asdict(item) for item in run.candidates], "audit": list(run.audit)})
    return path


def profile_sensitivity(config_path: str | Path, manifests: str | Path, teacher_checkpoint: str | Path, output: str | Path) -> Path:
    config = load_config(config_path)
    model, tokenizer, processor = _components(config)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    registry = build_quant_group_registry(model)
    loader = _loader(_records(manifests, "policy_search"), processor, tokenizer, config, shuffle=False)
    try:
        batch = _device_batch(next(iter(loader)), device)
    except StopIteration as error:
        raise ValueError("policy-search manifest is empty") from error
    model.eval()
    with torch.no_grad():
        reference = model(**batch)

    def probe(probed, group, action):
        calibrate_quantizers(probed, [batch], lambda module, item: module(**item))
        result = probed(**batch)
        prefix, kl = consistency_losses(result.visual_prefix, reference.visual_prefix, result.logits[:, :-1], reference.logits[:, :-1], 2.0, result.labels[:, 1:].ne(-100))
        gradients = torch.autograd.grad(result.loss, [parameter for parameter in probed.parameters() if parameter.requires_grad], allow_unused=True)
        norm = torch.sqrt(sum((gradient.detach().float().square().sum() for gradient in gradients if gradient is not None), torch.tensor(0.0, device=device)))
        target = probed
        for part in group.module_path.split("."):
            target = getattr(target, part)
        activation_range = float(target.activation_quantizer.scale.detach().abs().max() * (2 ** (action.activation_bits - 1) - 1)) if action.activation_bits < 16 else float("nan")
        return {"caption_loss_delta": float(result.loss.detach() - reference.loss.detach()), "prefix_distortion": float(prefix.detach()), "logit_kl": float(kl.detach()), "activation_range": activation_range if torch.isfinite(torch.tensor(activation_range)) else 0.0, "gradient_norm": float(norm), "parameters": group.parameters, "bitops": group.parameters * action.weight_bits * action.activation_bits, "model_size_bytes": group.parameters * action.weight_bits / 8}

    rows = SensitivityProfiler(model, registry, probe).profile(ACTION_SPACE)
    destination = Path(output)
    atomic_write_json(destination, rows)
    return destination
