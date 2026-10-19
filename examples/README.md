# Example data

`raw/1.mat` through `raw/5.mat` contain BSEA-RAW measurements. Each file stores a float32 variable named `raw` with shape 512 × 640.

`calibration/mask.mat.001` and `calibration/mask.mat.002` contain lossless parts of the BSEA calibration. Run the following command from the repository root:

```bash
python tools/restore_calibration.py
```

The reconstructed `calibration/mask.mat` contains a float32 variable named `mask` with shape 60 × 512 × 640. The script verifies each part and the complete file using SHA-256.

The RAW files are accepted by `sparc predict`. Calibration is an input to the spectral teacher and optical projection operator. The sensing model accepts RAW without calibration.

Segmentation labels and paired spectral targets are not included in this directory.
