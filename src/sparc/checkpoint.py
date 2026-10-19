"""Self-describing checkpoints and strict, sensing-only model loading."""

from pathlib import Path

import torch

from sparc.models import SPARC, SPARCInference, SpectralTeacher

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model,
    kind: str,
    *,
    config: dict,
    epoch: int = 0,
    optimizer=None,
    scheduler=None,
    **metadata,
):
    if kind not in {"sparc", "inference", "teacher"}:
        raise ValueError(f"Unknown checkpoint kind: {kind}")
    payload = dict(
        format_version=FORMAT_VERSION,
        kind=kind,
        model=model.state_dict(),
        model_config=model.model_config,
        config=config,
        epoch=epoch,
        **metadata,
    )
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Expected a SPARC checkpoint produced by this package")
    return payload


def load_teacher(path: str | Path, device: torch.device) -> SpectralTeacher:
    payload = read_checkpoint(path)
    if payload["kind"] != "teacher":
        raise ValueError("A spectral teacher checkpoint is required")
    teacher = SpectralTeacher(**payload["model_config"])
    teacher.load_state_dict(payload["model"], strict=True)
    return teacher.to(device).freeze()


def load_inference(
    path: str | Path, device: torch.device, backend: str | None = None
) -> tuple[SPARCInference, dict]:
    payload = read_checkpoint(path)
    if payload["kind"] not in {"sparc", "inference"}:
        raise ValueError("Expected a joint SPARC or sensing-only checkpoint")
    model_config = dict(payload["model_config"])
    model_config.pop("spectral_bands", None)
    if backend:
        model_config["mamba_backend"] = backend
    architecture = model_config.pop("architecture", "paper")
    if architecture == "pretrained_v1":
        from sparc.models.pretrained import PretrainedSPARCInference

        if payload["kind"] != "inference":
            raise ValueError("Pretrained compatibility checkpoints must be sensing-only")
        model = PretrainedSPARCInference(**model_config)
    elif architecture == "paper":
        model = SPARCInference(**model_config)
    else:
        raise ValueError(f"Unknown SPARC architecture: {architecture}")
    state = payload["model"]
    if payload["kind"] == "sparc":
        # Validate the complete training state before extracting the sensing path.
        full_config = dict(payload["model_config"])
        if backend:
            full_config["mamba_backend"] = backend
        full_model = SPARC(**full_config)
        full_model.load_state_dict(state, strict=True)
        model = full_model.to_inference()
    else:
        model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
