"""Explicit MATLAB v5/v7.2 array contracts and normalization."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def read_array(path: str | Path, key: str, ndim: int) -> np.ndarray:
    path = Path(path)
    try:
        content = loadmat(path)
    except NotImplementedError as error:
        raise ValueError(f"{path}: export MATLAB v7.3 files as v7/v7.2 first") from error
    if key not in content:
        raise ValueError(f"{path}: missing MATLAB variable {key!r}")
    array = np.asarray(content[key])
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path}:{key} must be a numeric {ndim}D array; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}:{key} contains NaN or infinity")
    return array


def normalize_raw(raw: np.ndarray, mode: str = "max") -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 2 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("RAW must be a finite, nonnegative [H,W] array with H,W >= 2")
    if mode == "max":
        scale = float(raw.max())
        if scale > 0:
            raw = raw / scale
    elif mode != "none":
        raise ValueError("raw_normalization must be 'max' or 'none'")
    return np.ascontiguousarray(raw)


def load_response(path: str | Path, bands: int) -> np.ndarray:
    response = read_array(path, "mask", 3).astype(np.float32)
    if response.shape[0] != bands:
        raise ValueError(f"Calibration must be [C,H,W] with C={bands}; got {response.shape}")
    response = np.maximum(response, 0)
    peak = float(response.max())
    if peak == 0:
        raise ValueError("Calibration response cannot be all zero")
    return np.ascontiguousarray(response / peak)
