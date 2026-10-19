# Pretrained sensing checkpoint

`sparc_pretrained.pt` contains a two-class SPARC sensing model with 32 base channels and 2,016,944 parameters. Input tensors have shape [B, 1, H, W]. Output tensors contain two semantic logits per pixel.

Checkpoint metadata specifies `architecture: pretrained_v1`. `PretrainedSPARCInference` uses learned residual projections, spectral-channel attention and the shared spectral latent modulator. Prediction requires neither the auxiliary branch nor spectral calibration.

The `source_sha256` metadata field identifies the parameter source:

```text
fdf6047777b27d13a4c2afa6d3e44e990dc725d099355c25b56b2166a2b0a493
```

The main README documents prediction commands and preprocessing. Example RAW inputs do not include segmentation labels.
