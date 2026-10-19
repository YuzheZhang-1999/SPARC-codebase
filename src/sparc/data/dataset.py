"""Datasets with fixed splits and aligned RAW/label/spectral/calibration transforms."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sparc.data.io import load_response, normalize_raw, read_array


def read_split(root: Path, split_file: str, split: str) -> list[str]:
    with (root / split_file).open(encoding="utf-8") as handle:
        splits = json.load(handle)
    all_ids = {}
    for name, values in splits.items():
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {name!r}")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("Split entries must be lists of sample ID strings")
        ids = [Path(value).stem for value in values]
        if any(Path(value).name != value or value in {".", "..", ""} for value in values):
            raise ValueError("Sample IDs must be filenames without directory components")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate IDs in {name} split")
        for sample_id in ids:
            if sample_id in all_ids:
                raise ValueError(
                    f"Sample {sample_id} appears in both {all_ids[sample_id]} and {name}"
                )
            all_ids[sample_id] = name
    ids = [Path(value).stem for value in splits.get(split, [])]
    if not ids:
        raise ValueError(f"Split {split!r} is missing or empty")
    return ids


class BSEADataset(Dataset):
    """Load recorded RAW and optional labels/teacher targets or reconstruction cubes.

    Spatial augmentation uses flips only, applied to every aligned array including
    the calibration. No arbitrary rotation or interpolation of the BSEA mosaic.
    """

    def __init__(
        self,
        config: dict,
        split: str,
        *,
        labels: bool = True,
        target: str | None = None,
        response: bool = False,
        augment: bool = False,
    ):
        self.config = config
        self.root = Path(config["root"])
        self.ids = read_split(self.root, config.get("split_file", "splits.json"), split)
        self.labels = labels
        self.target = target
        self.augment = augment
        metadata_path = self.root / "teacher_targets" / "metadata.json"
        if target == "cached" and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("raw_normalization", "spectral_scale"):
                if metadata.get(key) != config.get(key):
                    raise ValueError(f"Cached targets disagree with data configuration on {key}")
        self.response = None
        if response:
            self.response = load_response(
                self.root / config.get("response_file", "mask.mat"), config["spectral_bands"]
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        sample_id = self.ids[index]
        raw = normalize_raw(
            read_array(self.root / "raw" / f"{sample_id}.mat", "raw", 2),
            self.config.get("raw_normalization", "max"),
        )
        arrays = {"raw": raw[None]}
        if self.labels:
            label = read_array(self.root / "gt" / f"{sample_id}.mat", "gt", 2)
            if (label != np.floor(label)).any():
                raise ValueError(f"{sample_id}: gt must contain integer class IDs")
            if label.min() < 0 or label.max() >= self.config["num_classes"]:
                raise ValueError(f"{sample_id}: gt class IDs outside configured range")
            arrays["label"] = label.astype(np.int64)
        if self.target:
            folder, variable = (
                ("hsi", "cube") if self.target == "hsi" else ("teacher_targets", "cube")
            )
            cube = read_array(self.root / folder / f"{sample_id}.mat", variable, 3)
            if cube.shape[0] != self.config["spectral_bands"] or (
                self.target == "hsi" and cube.min() < 0
            ):
                raise ValueError(f"{sample_id}: {folder} must be nonnegative [C,H,W]")
            # Spectral scale is explicit; never normalize targets using the RAW maximum.
            scale = self.config.get("spectral_scale", 1.0) if self.target == "hsi" else 1.0
            arrays["spectral_target"] = cube.astype(np.float32) * scale
        if self.response is not None:
            arrays["response"] = self.response
        if any(array.shape[-2:] != raw.shape for array in arrays.values()):
            raise ValueError(f"{sample_id}: RAW, gt, cube and response must be spatially aligned")
        if self.augment:
            for axis in (-2, -1):
                if torch.rand(()).item() < 0.5:
                    arrays = {key: np.flip(array, axis=axis) for key, array in arrays.items()}
        sample = {
            key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()
        }
        sample["id"] = sample_id
        return sample
