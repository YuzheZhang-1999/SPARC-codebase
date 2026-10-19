"""Configuration loading with paths resolved relative to the YAML file."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for section in ("model", "data", "training", "loss", "teacher"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for section, key in (
        ("data", "root"),
        ("training", "output_dir"),
        ("teacher", "checkpoint"),
        ("teacher", "output_dir"),
    ):
        value = config[section].get(key)
        if value:
            value = Path(value).expanduser()
            config[section][key] = str(value if value.is_absolute() else path.parent / value)
    model, data = config["model"], config["data"]
    if model["spectral_bands"] != data["spectral_bands"]:
        raise ValueError("model.spectral_bands must equal data.spectral_bands")
    if model["num_classes"] != data["num_classes"]:
        raise ValueError("model.num_classes must equal data.num_classes")
    if data.get("raw_normalization", "max") not in {"max", "none"}:
        raise ValueError("data.raw_normalization must be max or none")
    if config["teacher"]["source"] not in {"online", "cached"}:
        raise ValueError("teacher.source must be online or cached")
    return config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; select --device cpu with reference backend"
        )
    return torch.device(value)


def write_json(path: str | Path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
