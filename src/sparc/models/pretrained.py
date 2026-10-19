"""Sensing model for checkpoints with architecture pretrained_v1.

VSM blocks use learned residual projections. Spectral attention left-multiplies
value features by channel affinity. Inputs are single-channel RAW measurements.
"""

import re

import torch
from torch import nn

from sparc.models.blocks import LayerNorm2d, SpectralLatentAttention, VSMBlock
from sparc.models.sparc import SPARCInference


class PretrainedVSMBlock(VSMBlock):
    """VSM block with a learned 1 × 1 residual projection."""

    def __init__(self, dim: int, backend: str):
        super().__init__(dim, backend)
        self.projection = nn.Conv2d(dim, dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        local = self.local(features) + self.projection(features)
        tokens = self.mamba(self.norm(local.flatten(2).transpose(1, 2)))
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class PretrainedLayerNorm2d(LayerNorm2d):
    """Per-pixel channel normalization using explicit mean and variance."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.permute(0, 2, 3, 1)
        mean = tokens.mean(-1, keepdim=True)
        variance = tokens.var(-1, keepdim=True, unbiased=False)
        tokens = (tokens - mean) / torch.sqrt(variance + 1e-5)
        return (tokens * self.norm.weight + self.norm.bias).permute(0, 3, 1, 2)


class PretrainedSpectralAttention(SpectralLatentAttention):
    """Spectral-channel attention with left-applied affinity."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        query, key, value = [
            projection(features).reshape(batch, self.heads, channels // self.heads, -1)
            for projection in self.projections
        ]
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)
        affinity = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.project((affinity @ value).reshape(batch, channels, height, width))


class PretrainedSPARCInference(SPARCInference):
    """Single-branch SPARC inference with projected residuals and spectral attention."""

    def __init__(self, dim: int = 32, num_classes: int = 2, mamba_backend: str = "reference"):
        super().__init__(dim, num_classes, mamba_backend)
        replacements = []
        for name, module in self.named_modules():
            if isinstance(module, VSMBlock):
                replacements.append(
                    (name, PretrainedVSMBlock(module.local[0].in_channels, mamba_backend))
                )
            elif isinstance(module, SpectralLatentAttention):
                replacements.append(
                    (name, PretrainedSpectralAttention(module.project.in_channels, module.heads))
                )
            elif isinstance(module, LayerNorm2d):
                replacements.append((name, PretrainedLayerNorm2d(module.norm.normalized_shape[0])))
        for name, replacement in replacements:
            parent_name, _, child_name = name.rpartition(".")
            parent = self.get_submodule(parent_name) if parent_name else self
            setattr(parent, child_name, replacement)
        self.model_config["architecture"] = "pretrained_v1"


def map_vsm_key(key: str) -> str:
    part, _, suffix = key.partition(".")
    names = {"res": "local", "conv": "projection", "att": "mamba", "norm": "norm"}
    if part not in names:
        raise ValueError(f"Unrecognized pretrained VSM parameter: {key}")
    return names[part] + "." + suffix


def map_sla_key(key: str) -> str:
    match = re.fullmatch(r"blocks\.0\.([0-3])\.(.+)", key)
    if match is None:
        raise ValueError(f"Unrecognized pretrained SLA parameter: {key}")
    index, suffix = match.groups()
    if index in {"0", "2"}:
        if not suffix.startswith("body."):
            raise ValueError(f"Unrecognized normalization parameter: {key}")
        norm = "attention_norm" if index == "0" else "refinement_norm"
        return norm + ".norm." + suffix.removeprefix("body.")
    if index == "1":
        if suffix == "rescale":
            return "attention.scale"
        names = {
            "to_q": "projections.0.0",
            "q_dwconv": "projections.0.1",
            "to_k": "projections.1.0",
            "k_dwconv": "projections.1.1",
            "to_v": "projections.2.0",
            "v_dwconv": "projections.2.1",
            "proj": "project",
        }
        name, _, parameter = suffix.partition(".")
        if name not in names:
            raise ValueError(f"Unrecognized attention parameter: {key}")
        return "attention." + names[name] + "." + parameter
    if suffix.startswith("out_conv."):
        return "refinement.project.1." + suffix.removeprefix("out_conv.")
    match = re.fullmatch(r"net([12])\.([02])\.(weight|bias)", suffix)
    if match is None:
        raise ValueError(f"Unrecognized refinement parameter: {key}")
    branch, layer, parameter = match.groups()
    return f"refinement.paths.{int(branch) - 1}.{layer}.{parameter}"


def map_pretrained_state(state: dict) -> tuple[dict, list[str]]:
    """Map every sensing parameter; reject unknown keys and record excluded auxiliary keys."""
    mapped, excluded = {}, []
    auxiliary_prefixes = (
        "embedding1.",
        "embedding2.",
        "embedding.",
        "down_sample.",
        "up_sample.",
        "mapping.",
        "encoder_layers.",
        "decoder_layers.",
    )
    direct = {
        "stem": "optical_adapter",
        "downsample": "sensing_encoder.downsample",
        "upsample": "sensing_decoder.upsample",
        "head": "perception_head",
    }
    shared = {
        "att1": "input_sla",
        "bottleneck": "latent_sla",
        "att4": "output_sla",
        "ds1": "downsample",
        "us1": "upsample",
        "fusion1": "fusion",
    }
    for key, value in state.items():
        if key.startswith(auxiliary_prefixes):
            excluded.append(key)
            continue
        first, _, suffix = key.partition(".")
        if first in direct:
            target = direct[first] + "." + suffix
        elif first == "encoder":
            stage, layer, suffix = suffix.split(".", 2)
            target = (
                f"sensing_encoder.blocks.{stage}." + map_vsm_key(suffix)
                if layer == "0"
                else f"sensing_encoder.downsamples.{stage}." + suffix
            )
        elif first == "decoder":
            stage, layer, suffix = suffix.split(".", 2)
            if layer == "2":
                target = f"sensing_decoder.blocks.{stage}." + map_vsm_key(suffix)
            else:
                module = {"0": "upsamples", "1": "fusions"}[layer]
                target = f"sensing_decoder.{module}.{stage}." + suffix
        elif first == "share":
            module, _, suffix = suffix.partition(".")
            if module not in shared:
                raise ValueError(f"Unknown shared module: {key}")
            if module in {"att1", "bottleneck", "att4"}:
                suffix = map_sla_key(suffix)
            target = "ssls_modulator." + shared[module] + "." + suffix
        else:
            raise ValueError(f"Unrecognized pretrained parameter: {key}")
        if target in mapped:
            raise ValueError(f"Duplicate mapped parameter: {target}")
        mapped[target] = value
    return mapped, excluded
