"""Dual-branch training and the retained SPARC sensing path."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from sparc.models.components import (
    AuxiliaryHead,
    BranchDecoder,
    BranchEncoder,
    OpticalAdapter,
    PerceptionHead,
    SSLSModulator,
)


class SPARCInference(nn.Module):
    """F_ada -> E_sense -> M_ssls -> D_sense -> PerceptionHead."""

    def __init__(self, dim: int = 32, num_classes: int = 2, mamba_backend: str = "mamba"):
        super().__init__()
        if dim < 2 or dim % 2 or num_classes < 2:
            raise ValueError("dim must be a positive even width >= 2; num_classes must be >= 2")
        self.model_config = dict(dim=dim, num_classes=num_classes, mamba_backend=mamba_backend)
        self.optical_adapter = OpticalAdapter(dim)
        self.sensing_encoder = BranchEncoder(dim, "sense", mamba_backend)
        self.ssls_modulator = SSLSModulator(dim * 4)
        self.sensing_decoder = BranchDecoder(dim, "sense", mamba_backend)
        self.perception_head = PerceptionHead(dim, num_classes)

    @staticmethod
    def prepare_input(raw: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if raw.ndim != 4 or raw.shape[1] != 1 or min(raw.shape[-2:]) < 2:
            raise ValueError("BSEA-RAW must have shape [B, 1, H, W], with H,W >= 2")
        height, width = raw.shape[-2:]
        # Three encoder downsamplings and one SSLS downsampling.
        raw = F.pad(raw, (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
        return raw, (height, width)

    def sense(self, features: torch.Tensor) -> torch.Tensor:
        deep, bypasses = self.sensing_encoder(features)
        modulated = self.ssls_modulator(deep)
        return self.perception_head(self.sensing_decoder(modulated, bypasses))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        raw, (height, width) = self.prepare_input(raw)
        return self.sense(self.optical_adapter(raw))[..., :height, :width]


class SPARC(SPARCInference):
    """Training decoder with shared adapter/SSLS and two prediction branches."""

    def __init__(
        self,
        dim: int = 32,
        num_classes: int = 2,
        spectral_bands: int = 60,
        mamba_backend: str = "mamba",
    ):
        super().__init__(dim, num_classes, mamba_backend)
        if spectral_bands < 1:
            raise ValueError("spectral_bands must be positive")
        self.model_config["spectral_bands"] = spectral_bands
        self.auxiliary_encoder = BranchEncoder(dim, "recon", mamba_backend)
        self.auxiliary_decoder = BranchDecoder(dim, "recon", mamba_backend)
        self.auxiliary_head = AuxiliaryHead(dim, spectral_bands)

    def forward(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, (height, width) = self.prepare_input(raw)
        features = self.optical_adapter(raw)
        sense = self.sense(features)
        deep, bypasses = self.auxiliary_encoder(features)
        recon = self.auxiliary_head(self.auxiliary_decoder(self.ssls_modulator(deep), bypasses))
        return {"sense": sense[..., :height, :width], "recon": recon[..., :height, :width]}

    def to_inference(self) -> SPARCInference:
        """Extract an independent model containing only the five retained modules."""
        inference = SPARCInference.__new__(SPARCInference)
        nn.Module.__init__(inference)
        inference.model_config = {
            k: v for k, v in self.model_config.items() if k != "spectral_bands"
        }
        for name in (
            "optical_adapter",
            "sensing_encoder",
            "ssls_modulator",
            "sensing_decoder",
            "perception_head",
        ):
            setattr(inference, name, deepcopy(getattr(self, name)))
        return inference.eval()
