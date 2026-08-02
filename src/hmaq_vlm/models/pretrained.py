from __future__ import annotations

from pathlib import Path

from hmaq_vlm.config import ModelConfig
from .vlm import HMAQVLM


_VISION_CLASSIFIER_KEYS = frozenset({"head.weight", "head.bias"})


def _load_vision_backbone_checkpoint(timm, vision, checkpoint: Path) -> None:
    """Load a classification checkpoint into a headless vision backbone."""
    incompatible = timm.models.load_checkpoint(vision, str(checkpoint), strict=False)
    if incompatible is None:
        return

    missing = set(getattr(incompatible, "missing_keys", ()))
    unexpected = set(getattr(incompatible, "unexpected_keys", ()))
    unexpected_backbone = unexpected - _VISION_CLASSIFIER_KEYS
    if missing or unexpected_backbone:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected_backbone:
            details.append(f"unexpected={sorted(unexpected_backbone)}")
        raise RuntimeError(f"ViT checkpoint does not match the configured backbone: {'; '.join(details)}")


def load_pretrained_vlm(config: ModelConfig) -> HMAQVLM:
    """Load revision-pinned ViT-Small and GPT-2 Small without a source dependency on Giathoai/VLM."""
    try:
        import timm
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("pretrained loading requires timm, transformers and huggingface_hub") from error
    vision_dir = Path(snapshot_download(config.vision_model, revision=config.vision_revision))
    architecture = config.vision_model.split("/", 1)[-1]
    vision = timm.create_model(architecture, pretrained=False, num_classes=0)
    checkpoints = list(vision_dir.glob("*.safetensors")) or list(vision_dir.glob("*.bin"))
    if not checkpoints:
        raise RuntimeError(f"revision-pinned ViT snapshot contains no supported checkpoint: {vision_dir}")
    _load_vision_backbone_checkpoint(timm, vision, checkpoints[0])
    if config.gradient_checkpointing and hasattr(vision, "set_grad_checkpointing"):
        vision.set_grad_checkpointing(True)
    language = AutoModelForCausalLM.from_pretrained(config.language_model, revision=config.language_revision)
    if hasattr(language, "gradient_checkpointing_enable") and config.gradient_checkpointing:
        language.gradient_checkpointing_enable()
    return HMAQVLM(vision, language, vision_dim=384, language_dim=768)
