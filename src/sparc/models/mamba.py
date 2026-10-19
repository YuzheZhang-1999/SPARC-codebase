"""Mamba backend selection and a small, unfused selective-scan implementation.

The reference backend follows Mamba v1's input-dependent state-space recurrence.
It is intended for correctness checks on small CPU inputs, not throughput benchmarks.
See https://github.com/state-spaces/mamba for the accelerated implementation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMamba(nn.Module):
    """Unfused Mamba v1 with parameter names compatible with mamba_ssm.Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        width = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * width, bias=False)
        self.conv1d = nn.Conv1d(width, width, d_conv, groups=width, padding=d_conv - 1)
        self.x_proj = nn.Linear(width, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, width)
        nn.init.uniform_(self.dt_proj.weight, -(self.dt_rank**-0.5), self.dt_rank**-0.5)
        dt = torch.exp(torch.empty(width).uniform_(math.log(0.001), math.log(0.1)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(width, 1))
        self.D = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Linear(width, d_model, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        signal, gate = self.in_proj(tokens).chunk(2, dim=-1)
        signal = F.silu(self.conv1d(signal.transpose(1, 2))[..., : tokens.shape[1]])
        signal = signal.transpose(1, 2)
        delta, input_b, output_c = self.x_proj(signal).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta)).float()
        transition = -self.A_log.float().exp()
        state = signal.new_zeros(signal.shape[0], signal.shape[2], self.d_state).float()
        output = []
        for index in range(tokens.shape[1]):
            step = delta[:, index, :, None]
            state = (step * transition).exp() * state + step * input_b[
                :, index, None, :
            ].float() * signal[:, index, :, None].float()
            value = (state * output_c[:, index, None, :].float()).sum(-1)
            output.append(value + self.D.float() * signal[:, index].float())
        result = torch.stack(output, dim=1).to(signal.dtype) * F.silu(gate)
        return self.out_proj(result)


def build_mamba(dim: int, backend: str) -> nn.Module:
    if backend == "reference":
        return ReferenceMamba(dim)
    if backend != "mamba":
        raise ValueError("mamba_backend must be 'mamba' or 'reference'")
    try:
        from mamba_ssm import Mamba
    except ImportError as error:
        raise ImportError(
            "Install mamba-ssm and causal-conv1d for CUDA, or use mamba_backend: reference "
            "for small CPU checks."
        ) from error
    return Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
