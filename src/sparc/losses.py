"""Semantic sensing and spectral distillation objectives (Eqs. S13-S17)."""

import torch
from torch import nn
from torch.nn import functional as F


def sensing_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L_sense = cross entropy + class-averaged soft Dice loss."""
    probability = logits.softmax(dim=1)
    target = F.one_hot(labels.long(), num_classes=logits.shape[1]).movedim(-1, 1)
    target = target.to(probability.dtype)
    dims = (0, 2, 3)
    intersection = (probability * target).sum(dims)
    cardinality = (probability + target).sum(dims)
    dice = (2 * intersection + 1e-6) / (cardinality + 1e-6)
    return F.cross_entropy(logits, labels.long()) + 1 - dice.mean()


def spatial_total_variation(prediction: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """Isotropic TV on valid left/down neighbours of the prediction (Eq. S15).

    Batch and spectral channels are averaged. The paper uses a spatial sum;
    'mean' is an explicit optional normalization for differing image resolutions.
    """
    if prediction.ndim != 4 or min(prediction.shape[-2:]) < 2:
        raise ValueError("TV expects [B,C,H,W] with H,W >= 2")
    center = prediction[..., :-1, 1:]
    left = prediction[..., :-1, :-1] - center
    down = prediction[..., 1:, 1:] - center
    # vector_norm has a finite zero subgradient on spatially constant predictions.
    magnitude = torch.linalg.vector_norm(torch.stack([left, down], dim=-1), dim=-1)
    if reduction == "sum":
        return magnitude.sum(dim=(-2, -1)).mean()
    if reduction == "mean":
        return magnitude.mean()
    raise ValueError("tv_reduction must be 'sum' or 'mean'")


class SpectralDistillationLoss(nn.Module):
    """RMSE + alpha TV(prediction) + beta MRAE, with numerical stabilization."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        tv_reduction: str = "sum",
        epsilon: float = 1e-6,
    ):
        super().__init__()
        if alpha < 0 or beta < 0 or epsilon <= 0:
            raise ValueError("Loss weights must be nonnegative; epsilon must be positive")
        self.alpha, self.beta = alpha, beta
        self.tv_reduction, self.epsilon = tv_reduction, epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError("Spectral prediction and teacher target shapes must match")
        target = target.detach()
        error = prediction - target
        rmse = error.square().mean().add(self.epsilon**2).sqrt() - self.epsilon
        mrae = (error.abs() / target.abs().clamp_min(self.epsilon)).mean()
        tv = spatial_total_variation(prediction, self.tv_reduction)
        return {
            "distillation": rmse + self.alpha * tv + self.beta * mrae,
            "rmse": rmse,
            "tv": tv,
            "mrae": mrae,
        }


class SPARCLoss(nn.Module):
    """L = L_sense + L_distillation; teacher tensors never receive gradients."""

    def __init__(self, **kwargs):
        super().__init__()
        self.distillation = SpectralDistillationLoss(**kwargs)

    def forward(self, output: dict, labels: torch.Tensor, teacher_target: torch.Tensor) -> dict:
        components = self.distillation(output["recon"], teacher_target)
        components["sense"] = sensing_loss(output["sense"], labels)
        components["total"] = components["sense"] + components["distillation"]
        return components
