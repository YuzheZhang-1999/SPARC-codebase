# Reproduction guide

## Environment

The package requires Python 3.10 or later. Install PyTorch for the target CPU or CUDA environment, then install the package with `python -m pip install -e .`.

The CUDA optional dependency group provides accelerated sequence and convolution kernels:

```bash
python -m pip install -e '.[cuda]' --no-build-isolation
```

A compatible Linux CUDA compiler toolchain is required to build the acceleration packages. Dependency names and version ranges are declared in [pyproject.toml](../pyproject.toml). Backend identifiers are `mamba` for accelerated execution and `reference` for the unfused PyTorch implementation. Select `--backend reference --device cpu` for CPU inference.

## Data layout

Paths in YAML files are relative to the configuration file. Set `data.root` to a directory containing:

```text
data/
├── splits.json
├── mask.mat
├── raw/<sample_id>.mat
├── gt/<sample_id>.mat
├── hsi/<sample_id>.mat
└── teacher_targets/<sample_id>.mat
```

| File | Variable | Shape | Usage |
| --- | --- | --- | --- |
| `raw/*.mat` | `raw` | H × W | Recorded optical embeddings |
| `gt/*.mat` | `gt` | H × W | Integer class labels; background = 0 |
| `mask.mat` | `mask` | C × H × W | Teacher input and optical synthesis |
| `hsi/*.mat` | `cube` | C × H × W | Teacher pre-training targets |
| `teacher_targets/*.mat` | `cube` | C × H × W | Cached teacher supervision |

Use MATLAB v5/v7/v7.2 files and channel-first spectral arrays. Paired arrays must share their spatial grid. Training batches require a common spatial size.

Split IDs must be disjoint:

```json
{
  "train": ["sample_001", "sample_002"],
  "val": ["sample_003"],
  "test": ["sample_004"]
}
```

Keep acquisitions from the same specimen in one split. Duplicate IDs and overlap between splits are rejected.

## Calibration and normalization

Restore the example calibration from its lossless parts:

```bash
python tools/restore_calibration.py
```

The script verifies SHA-256 checksums and creates `examples/calibration/mask.mat`. Copy this file to the dataset root when the calibration matches the data. Pretrained sensing inference does not require calibration.

RAW uses per-image positive-maximum normalization by default. All-zero inputs remain zero. Teacher spectral targets use their input scale multiplied by `data.spectral_scale`. Cached teacher predictions already use teacher-output units. Teacher response maps are clipped at zero and normalized by their global positive maximum.

The default spectral width is 60. The decoder, teacher, paired cubes and calibration must use matching spectral grids. Wavelength resampling requires the acquisition wavelengths.

Training augmentation applies synchronized flips to RAW, labels, spectral targets and response maps.

## Optimization

Teacher pre-training requires paired RAW, spectra and calibration. Joint decoder training requires semantic labels and a trained teacher. The teacher operates with frozen parameters.

The default configuration uses Adam for 200 epochs, an initial learning rate of 1e-4 and beta values (0.9, 0.999). The scheduler halves the learning rate after 10 epochs without validation-loss improvement. Distillation weights are alpha = 1 and beta = 0.2. Four accumulation steps with batch size 4 give a nominal effective batch size of 16.

Runs contain `best.pt`, `last.pt`, `history.json` and the resolved `config.yaml`. Model selection uses validation loss. Test evaluation is separate. Each run requires a new output directory.

For cached spectral supervision:

```bash
sparc cache-teacher --config configs/sparc.yaml \
  --checkpoint runs/teacher/best.pt --split train --device cuda:0
sparc cache-teacher --config configs/sparc.yaml \
  --checkpoint runs/teacher/best.pt --split val --device cuda:0
```

Set `teacher.source: cached` before joint training. Cache metadata records the teacher checksum and normalization. Online supervision evaluates the teacher on augmented inputs; cached supervision transforms stored targets.

## Evaluation and checkpoint loading

Evaluation accumulates confusion counts across the split. Aggregate Dice and IoU exclude background and average foreground classes. Per-class scores and the confusion matrix are also saved. A class absent from both prediction and ground truth receives 1. Scores are in [0, 1].

Use `--data-root /path/to/data` to evaluate data at a different location.

Checkpoint metadata selects the model architecture. State dictionaries are loaded strictly. The `pretrained_v1` architecture uses learned residual projections and left-applied channel affinity; the training decoder uses identity residuals and right-applied channel affinity.

Encoder/decoder state dictionaries can be converted with:

```bash
sparc import-pretrained --checkpoint /path/to/weights.pt \
  --output checkpoints/imported.pt --backend reference
```

## Optical synthesis

The optical projection operator expects calibrated response values and a cube on the same spectral and spatial grid:

```bash
sparc encode --cube data/hsi/sample_001.mat \
  --response data/mask.mat --output data/raw/sample_001.mat
```

The operator computes a response-weighted sum over spectral bands. It does not interpolate wavelengths or normalize response values.

## Python interface

```python
import torch
from sparc.models import SPARC

model = SPARC(dim=32, num_classes=2, spectral_bands=60).cuda()
measurement = torch.rand(1, 1, 512, 640, device="cuda")
outputs = model(measurement)

sensing_model = model.to_inference()
with torch.inference_mode():
    logits = sensing_model(measurement)
```

`outputs["sense"]` contains semantic logits; `outputs["recon"]` contains auxiliary spectral predictions. Exported sensing checkpoints exclude the teacher, auxiliary branch and optimizer state. Binary prediction PNGs use 0 and 255; multiclass PNGs contain class IDs.
