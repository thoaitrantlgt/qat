from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hmaq_vlm.losses import consistency_losses
from hmaq_vlm.models import HMAQVLM
from hmaq_vlm.quantization import (
    ACTION_SPACE,
    LSQFakeQuantizer,
    MixedPrecisionPolicy,
    PrecisionAction,
    QuantizedLinear,
    build_quant_group_registry,
    inject_quantizers,
    temporary_policy,
)
from hmaq_vlm.training import set_trainable_stage


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 6)
        self.calls = 0

    def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        pooled = pixels.mean(dim=(-1, -2))
        token = self.proj(pooled)
        return token[:, None, :].expand(-1, 5, -1)


class TinyLM(nn.Module):
    def __init__(self, force_token: int | None = None) -> None:
        super().__init__()
        self.embed = nn.Embedding(13, 8)
        self.block = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 13, bias=False)
        self.force_token = force_token
        self.last_attention_mask = None
        self.last_position_ids = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values=None, use_cache=False):
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_position_ids = position_ids.detach().clone()
        logits = self.lm_head(torch.tanh(self.block(inputs_embeds)))
        if self.force_token is not None:
            logits = torch.zeros_like(logits)
            logits[..., self.force_token] = 10
        return SimpleNamespace(logits=logits, past_key_values=(torch.tensor(1),) if use_cache else None)


def make_model(lm: TinyLM | None = None) -> HMAQVLM:
    return HMAQVLM(TinyVision(), lm or TinyLM(), vision_dim=6, language_dim=8)


def test_forward_aligns_visual_text_masks_positions_and_shifted_labels() -> None:
    model = make_model()
    pixels = torch.randn(2, 3, 4, 4)
    tokens = torch.tensor([[1, 4, 2], [1, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = tokens.masked_fill(mask == 0, -100)
    output = model(pixels, tokens, mask, labels)
    assert output.visual_prefix.shape == (2, 5, 8)
    assert output.logits.shape == (2, 8, 13)
    assert output.labels[:, :5].eq(-100).all()
    assert output.labels[1, -1].item() == -100
    assert model.language_model.last_attention_mask.tolist() == [[1] * 8, [1] * 7 + [0]]
    assert model.language_model.last_position_ids.tolist() == [list(range(8)), list(range(7)) + [0]]
    expected = F.cross_entropy(output.logits[:, :-1].reshape(-1, 13), output.labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(output.loss, expected)


def test_generation_caches_visual_features_and_stops_at_eos() -> None:
    lm = TinyLM(force_token=2)
    model = make_model(lm)
    result = model.generate(torch.randn(2, 3, 4, 4), torch.ones(2, 1, dtype=torch.long), eos_token_id=2, max_new_tokens=30)
    assert result.tolist() == [[1, 2], [1, 2]]
    assert model.vision_encoder.calls == 1


def test_lsq_has_scale_gradient_and_16_bit_is_exact_bypass() -> None:
    values = torch.tensor([[-2.0, -0.25, 0.5], [1.0, 2.0, 3.0]], requires_grad=True)
    quantizer = LSQFakeQuantizer(4, per_channel=True, channels=2, channel_axis=0)
    quantizer.calibrate(values.detach())
    quantizer(values).sum().backward()
    assert quantizer.scale.grad is not None and torch.isfinite(quantizer.scale.grad).all()
    bypass = LSQFakeQuantizer(16)
    assert torch.equal(bypass(values.detach()), values.detach())


class RegistryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.projector = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.language_model = nn.Module()
        self.language_model.transformer = nn.Sequential(nn.Linear(4, 4))
        self.language_model.lm_head = nn.Linear(4, 9, bias=False)


def test_registry_policy_injection_and_restoration_cover_all_actions() -> None:
    assert {(action.weight_bits, action.activation_bits) for action in ACTION_SPACE} == {(w, a) for w in (2, 4, 8, 16) for a in (2, 4, 8, 16)}
    model = RegistryModel()
    registry = build_quant_group_registry(model)
    assert [group.name for group in registry] == sorted(group.name for group in registry)
    assert all("lm_head" not in group.name for group in registry)
    policy = MixedPrecisionPolicy({group.name: PrecisionAction(4, 8) for group in registry})
    original = model.projector[0]
    with temporary_policy(model, registry, policy):
        assert isinstance(model.projector[0], QuantizedLinear)
        assert isinstance(model.language_model.lm_head, nn.Linear)
    assert model.projector[0] is original
    with pytest.raises(ValueError, match="exactly one action"):
        inject_quantizers(model, registry, MixedPrecisionPolicy({}))
    switched = RegistryModel()
    switched_registry = build_quant_group_registry(switched)
    inject_quantizers(switched, switched_registry, MixedPrecisionPolicy({group.name: PrecisionAction(8, 8) for group in switched_registry}))
    inject_quantizers(switched, switched_registry, MixedPrecisionPolicy({group.name: PrecisionAction(2, 4) for group in switched_registry}))
    assert switched.projector[0].weight_quantizer.bits == 2
    assert switched.projector[0].activation_quantizer.bits == 4


def test_registry_quantizes_real_gpt2_conv1d_but_not_head() -> None:
    from transformers.pytorch_utils import Conv1D

    model = RegistryModel()
    model.language_model.transformer = nn.Sequential(Conv1D(6, 4))
    registry = build_quant_group_registry(model)
    language = [group for group in registry if group.modality == "language"]
    assert len(language) == 1 and language[0].module_type == "conv1d"
    policy = MixedPrecisionPolicy({group.name: PrecisionAction(4, 8) for group in registry})
    inject_quantizers(model, registry, policy)
    assert model.language_model.transformer[0].weight_quantizer.channel_axis == 1
    assert isinstance(model.language_model.lm_head, nn.Linear)


def test_consistency_detaches_teacher_and_warmup_only_trains_projector() -> None:
    student_prefix = torch.randn(2, 3, 4, requires_grad=True)
    teacher_prefix = student_prefix.detach().clone().requires_grad_(True)
    student_logits = torch.randn(2, 3, 7, requires_grad=True)
    teacher_logits = student_logits.detach().clone().requires_grad_(True)
    prefix_loss, kl_loss = consistency_losses(student_prefix, teacher_prefix, student_logits, teacher_logits, temperature=2.0)
    assert prefix_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert kl_loss.item() == pytest.approx(0.0, abs=1e-7)
    (prefix_loss + kl_loss).backward()
    assert teacher_prefix.grad is None and teacher_logits.grad is None
    model = make_model()
    set_trainable_stage(model, epoch=1, warmup_epochs=3)
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    assert not any(parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())
    set_trainable_stage(model, epoch=4, warmup_epochs=3)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_logit_consistency_ignores_unmasked_positions() -> None:
    prefix = torch.randn(1, 2, 3)
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = student.detach().clone()
    teacher[:, 1:] += 100
    _, masked = consistency_losses(prefix, prefix, student, teacher, logit_mask=torch.tensor([[1, 0, 0]], dtype=torch.bool))
    assert masked.item() == pytest.approx(0.0, abs=1e-7)
