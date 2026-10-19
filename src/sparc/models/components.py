"""Named modules corresponding to the paper's per-stage forward pipeline."""

import torch
from torch import nn

from sparc.models.blocks import SLABlock, VSMBlock


class OpticalAdapter(nn.Conv2d):
    """F_ada: project a recorded BSEA-RAW embedding into neural features."""

    def __init__(self, dim: int):
        super().__init__(1, dim, 3, padding=1, bias=False)


def make_block(dim: int, kind: str, heads: int, backend: str) -> nn.Module:
    return VSMBlock(dim, backend) if kind == "sense" else SLABlock(dim, heads)


class BranchEncoder(nn.Module):
    """E_sense / E_recon: return deep features and branch-specific bypasses."""

    def __init__(self, dim: int, kind: str, backend: str, stages: int = 2):
        super().__init__()
        self.downsample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index in range(stages):
            width = dim * 2**index
            self.blocks.append(make_block(width, kind, 2**index, backend))
            self.downsamples.append(nn.Conv2d(width, 2 * width, 4, 2, 1, bias=False))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        bypasses = [features]
        features = self.downsample(features)
        for block, downsample in zip(self.blocks, self.downsamples):
            features = block(features)
            bypasses.append(features)
            features = downsample(features)
        return features, bypasses


class BranchDecoder(nn.Module):
    """D_sense / D_recon: decode SSLS features with same-branch bypasses."""

    def __init__(self, dim: int, kind: str, backend: str, stages: int = 2):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for index in reversed(range(stages)):
            width = dim * 2**index
            self.upsamples.append(nn.ConvTranspose2d(2 * width, width, 2, stride=2))
            self.fusions.append(nn.Conv2d(2 * width, width, 1, bias=False))
            self.blocks.append(make_block(width, kind, 2**index, backend))
        self.upsample = nn.ConvTranspose2d(dim, dim, 2, stride=2)

    def forward(self, features: torch.Tensor, bypasses: list[torch.Tensor]) -> torch.Tensor:
        for upsample, fusion, block, bypass in zip(
            self.upsamples, self.fusions, self.blocks, reversed(bypasses[1:])
        ):
            features = block(fusion(torch.cat([upsample(features), bypass], dim=1)))
        return self.upsample(features) + bypasses[0]


class SSLSModulator(nn.Module):
    """M_ssls: one shared spectral latent module, invoked once per branch.

    Three SLA blocks use a coarse latent level and concatenate the upsampled
    feature with the first SLA output before projection and final refinement.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.input_sla = SLABlock(dim, 4)
        self.downsample = nn.Conv2d(dim, 2 * dim, 4, 2, 1, bias=False)
        self.latent_sla = SLABlock(2 * dim, 8)
        self.upsample = nn.ConvTranspose2d(2 * dim, dim, 2, stride=2)
        self.fusion = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.output_sla = SLABlock(dim, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bypass = self.input_sla(features)
        latent = self.latent_sla(self.downsample(bypass))
        fused = self.fusion(torch.cat([self.upsample(latent), bypass], dim=1))
        return self.output_sla(fused)


class PerceptionHead(nn.Conv2d):
    """Semantic logits y_sense; no softmax is applied inside the model."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__(dim, num_classes, 3, padding=1, bias=False)


class AuxiliaryHead(nn.Conv2d):
    """Training-only spectral prediction y_recon (Fidelity Head in Fig. S3)."""

    def __init__(self, dim: int, spectral_bands: int):
        super().__init__(dim, spectral_bands, 3, padding=1, bias=False)
