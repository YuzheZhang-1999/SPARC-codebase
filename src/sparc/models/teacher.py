"""F_SR: spectral teacher used for pre-training and frozen supervision only."""

import torch
from torch import nn

from sparc.models.blocks import SLABlock
from sparc.models.components import BranchDecoder, BranchEncoder
from sparc.models.sparc import SPARCInference


class SpectralTeacher(nn.Module):
    """Calibrated spectral U-Net with three spectral-attention encoder stages.

    The teacher takes a calibrated response map alongside RAW measurements.
    """

    def __init__(self, spectral_bands: int = 60, dim: int = 32):
        super().__init__()
        if dim < 2 or dim % 2 or spectral_bands < 1:
            raise ValueError("Teacher width must be even >= 2 and spectral_bands positive")
        self.model_config = dict(spectral_bands=spectral_bands, dim=dim)
        self.raw_embedding = nn.Conv2d(1, dim, 3, padding=1, bias=False)
        self.response_embedding = nn.Conv2d(spectral_bands, dim, 3, padding=1, bias=False)
        self.fusion = nn.Conv2d(2 * dim, dim, 3, padding=1, bias=False)
        self.encoder = BranchEncoder(dim, "recon", "reference", stages=3)
        self.bottleneck = SLABlock(dim * 8, 8)
        self.decoder = BranchDecoder(dim, "recon", "reference", stages=3)
        self.spectral_head = nn.Conv2d(dim, spectral_bands, 3, padding=1, bias=False)

    def forward(self, raw: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 4 or response.shape[1] != self.model_config["spectral_bands"]:
            raise ValueError("Teacher response must have shape [B, spectral_bands, H, W]")
        if response.shape[0] != raw.shape[0] or response.shape[-2:] != raw.shape[-2:]:
            raise ValueError("RAW and calibrated response must share batch and spatial dimensions")
        raw, (height, width) = SPARCInference.prepare_input(raw)
        response = torch.nn.functional.pad(
            response, (0, raw.shape[-1] - width, 0, raw.shape[-2] - height), mode="replicate"
        )
        features = self.fusion(
            torch.cat([self.raw_embedding(raw), self.response_embedding(response)], dim=1)
        )
        deep, bypasses = self.encoder(features)
        decoded = self.decoder(self.bottleneck(deep), bypasses)
        return self.spectral_head(decoded)[..., :height, :width]

    def freeze(self) -> "SpectralTeacher":
        self.eval()
        self.requires_grad_(False)
        return self
