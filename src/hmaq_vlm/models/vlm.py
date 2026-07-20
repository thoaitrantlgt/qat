from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VLMOutput:
    logits: torch.Tensor
    visual_prefix: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    past_key_values: Any = None


class HMAQVLM(nn.Module):
    """ViT visual-token prefix projected into a causal language model."""

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, vision_dim: int = 384, language_dim: int = 768) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = nn.Sequential(nn.Linear(vision_dim, language_dim), nn.GELU(), nn.Linear(language_dim, language_dim))
        self.language_model = language_model

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder.forward_features(pixel_values)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        if features.ndim != 3:
            raise ValueError(f"vision encoder must return [batch,tokens,channels], got {tuple(features.shape)}")
        return self.projector(features)

    @staticmethod
    def _position_ids(mask: torch.Tensor) -> torch.Tensor:
        positions = (mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        return positions.masked_fill(mask == 0, 0)

    def _decode(self, embeddings: torch.Tensor, mask: torch.Tensor, position_ids: torch.Tensor, *, past_key_values: Any = None, use_cache: bool = False) -> Any:
        return self.language_model(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> VLMOutput:
        prefix = self.encode_images(pixel_values)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        embeddings = torch.cat((prefix, text_embeddings), dim=1)
        visual_mask = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat((visual_mask, attention_mask), dim=1)
        positions = self._position_ids(full_mask)
        decoded = self._decode(embeddings, full_mask, positions)
        full_labels = None
        loss = None
        if labels is not None:
            visual_labels = torch.full(prefix.shape[:2], -100, dtype=labels.dtype, device=labels.device)
            text_labels = labels.masked_fill(attention_mask == 0, -100)
            full_labels = torch.cat((visual_labels, text_labels), dim=1)
            loss = F.cross_entropy(
                decoded.logits[:, :-1].contiguous().view(-1, decoded.logits.shape[-1]),
                full_labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return VLMOutput(decoded.logits, prefix, full_mask, positions, full_labels, loss, getattr(decoded, "past_key_values", None))

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        eos_token_id: int,
        max_new_tokens: int = 30,
    ) -> torch.Tensor:
        if max_new_tokens < 0 or max_new_tokens > 30:
            raise ValueError("max_new_tokens must be between 0 and 30")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix = self.encode_images(pixel_values)
        visual_mask = torch.ones(prefix.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat((visual_mask, attention_mask), dim=1)
        embeddings = torch.cat((prefix, self.language_model.get_input_embeddings()(input_ids)), dim=1)
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        past = None
        for step in range(max_new_tokens):
            if step == 0:
                positions = self._position_ids(full_mask)
                decoded = self._decode(embeddings, full_mask, positions, use_cache=True)
            else:
                last_embedding = self.language_model.get_input_embeddings()(generated[:, -1:])
                position = self._position_ids(full_mask)[:, -1:]
                decoded = self._decode(last_embedding, full_mask, position, past_key_values=past, use_cache=True)
            past = getattr(decoded, "past_key_values", None)
            next_token = decoded.logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished |= next_token.eq(eos_token_id)
            full_mask = torch.cat((full_mask, (~finished).to(full_mask.dtype)[:, None]), dim=1)
            if bool(finished.all()):
                break
        return generated
