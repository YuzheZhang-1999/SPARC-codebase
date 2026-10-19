"""VSM and Spectral Latent Attention blocks (Supplementary Section 4.2)."""

import torch
from torch import nn
from torch.nn import functional as F

from sparc.models.mamba import build_mamba


class VSMBlock(nn.Module):
    """Two local CNL units, residual addition, then spatial Mamba (S10-S11)."""

    def __init__(self, dim: int, backend: str = "mamba"):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(dim)
        self.mamba = build_mamba(dim, backend)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = (features + self.local(features)).flatten(2).transpose(1, 2)
        tokens = self.mamba(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class LayerNorm2d(nn.Module):
    """Normalize hidden channels independently at each spatial position."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpectralLatentAttention(nn.Module):
    """Channel affinity A=softmax(scale QK), followed by VA (Eq. S12)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Hidden width must be divisible by attention heads")
        self.heads = heads
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, bias=False),
                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                )
                for _ in range(3)
            ]
        )
        self.scale = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        query, key, value = [
            projection(features).reshape(batch, self.heads, channels // self.heads, -1)
            for projection in self.projections
        ]
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        affinity = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attended = value.transpose(-2, -1) @ affinity
        attended = attended.transpose(-2, -1).reshape(batch, channels, height, width)
        return self.project(attended)


class ConvolutionalRefinement(nn.Module):
    """Parallel Conv-GELU-DWConv paths and concatenation from Fig. S3."""

    def __init__(self, dim: int):
        super().__init__()
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                )
                for _ in range(2)
            ]
        )
        self.project = nn.Sequential(nn.GELU(), nn.Conv2d(2 * dim, dim, 1, bias=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(features) for branch in self.paths], dim=1))


class SLABlock(nn.Module):
    """Residual spectral-channel attention and residual convolutional refinement."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention_norm = LayerNorm2d(dim)
        self.attention = SpectralLatentAttention(dim, heads)
        self.refinement_norm = LayerNorm2d(dim)
        self.refinement = ConvolutionalRefinement(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.attention(self.attention_norm(features))
        return features + self.refinement(self.refinement_norm(features))
