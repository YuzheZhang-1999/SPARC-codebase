"""Differentiable fixed-response BSEA optical encoding (Eq. S2)."""

import torch
from torch import nn


class BSEAEncoder(nn.Module):
    """Project an aligned spectral cube into one broadband RAW channel.

    Wavelength interpolation and radiometric calibration belong to data preparation.
    This operator expects cube and response on the same spectral/spatial grid.
    """

    def forward(self, cube: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        if cube.ndim != 4 or response.ndim not in (3, 4):
            raise ValueError("Expected cube [B,C,H,W] and response [C,H,W] or [B,C,H,W]")
        if response.ndim == 3:
            response = response.unsqueeze(0)
        if response.shape[1:] != cube.shape[1:] or response.shape[0] not in (1, cube.shape[0]):
            raise ValueError("Cube and BSEA response must share spectral and spatial dimensions")
        return (cube * response).sum(dim=1, keepdim=True)
