# Implementation details

## Forward pipeline

The sensing path implements main-text Eq. 1 and Supplementary Eqs. S4, S6 and S8:

```python
features = optical_adapter(raw)
deep, bypasses = sensing_encoder(features)
modulated = ssls_modulator(deep)
logits = perception_head(sensing_decoder(modulated, bypasses))
```

Joint training adds the auxiliary encoder, a second call to the same SSLS module, the auxiliary decoder and the auxiliary head. Both encoders receive the adapter features. Each SSLS call processes one branch's features. Encoder-to-decoder bypasses remain within their respective branches.

The default hidden widths are 32, 64 and 128. An initial spatial downsampling precedes two encoder stages. SSLS contains three SLA blocks with a 256-channel coarse level. The first block's features are concatenated with the upsampled latent features before projection and final refinement. Full-resolution adapter features provide the final residual bypass.

VSM blocks combine local convolutional processing, a residual pathway and selective spatial modelling. The default state width is 16, local sequence convolution width is 4, and expansion factor is 2.

Spectral Latent Attention uses pointwise and depthwise convolutional projections. Query and key vectors are normalized over spatial positions. A learnable per-head scale controls channel affinity. Values right-multiply the affinity matrix as specified in Eq. S12. SLA combines residual attention with two parallel convolutional refinement paths.

## Loss definitions

- Semantic supervision is cross entropy plus soft Dice loss averaged over classes, including background.
- Spectral distillation combines RMSE, isotropic spatial total variation of the auxiliary prediction, and MRAE.
- The TV stencil uses positions with both a valid left neighbour and a valid downward neighbour. The default sums spatial positions and averages batch and spectral dimensions. `tv_reduction: mean` averages all dimensions.
- RMSE uses `sqrt(MSE + epsilon²) - epsilon`. MRAE divides absolute error by `max(abs(target), epsilon)`.
- The teacher operates in evaluation mode with gradients disabled and is excluded from the optimizer.

## Teacher inputs

`SpectralTeacher` is a spectral U-Net with three encoder stages and spectral-channel attention. RAW and calibrated-response embeddings are concatenated and projected into encoder features.

Teacher pre-training requires spatially paired spectra, RAW measurements and calibrated response maps. `data.spectral_scale` sets the spectral target scale. Joint training requires the same RAW normalization and spectral scale as teacher pre-training. Cached teacher predictions already use teacher-output units.

Separate configuration files can specify teacher and perception datasets. Their spectral grid and normalization must agree.

## Checkpoint formats

`format_version` identifies the checkpoint schema. `kind` is `sparc`, `teacher` or `inference`. `model_config` specifies dimensions and backend selection; `model` contains the state dictionary.

`SPARC` uses a shared optical adapter, identity VSM residuals and right-applied channel affinity. It supports dual-branch optimization. `SPARC.to_inference()` returns an independent sensing model containing the adapter, sensing encoder, SSLS modulator, sensing decoder and perception head.

The bundled sensing checkpoint uses `architecture: pretrained_v1`. `PretrainedSPARCInference` contains learned 1 × 1 VSM residual projections, left-applied channel affinity and explicit per-pixel channel normalization. The loader selects this class from checkpoint metadata. State dictionaries are loaded strictly.

`import-pretrained` accepts encoder/decoder state dictionaries with `stem`, `encoder`, `decoder`, `share` and `head` keys. It validates sensing tensors, maps their names to the inference schema, records a source checksum and excludes auxiliary tensors.

Inference exports contain architecture and preprocessing metadata. They exclude teacher parameters, auxiliary parameters, optimizer state and scheduler state. Accelerator-specific engine conversion requires separate validation.

## Input normalization and spatial size

RAW measurements use per-image maximum normalization by default. All-zero inputs remain zero. The calibrated response used by the teacher is clipped at zero and divided by its positive global maximum.

RAW is padded on the right and bottom with edge values to a multiple of 16. Output logits are cropped to the input extent. Prediction uses no demosaicing or spectral reconstruction. The reference and accelerated implementations share parameter layouts.

## Optical projection

`BSEAEncoder` computes the fixed-response weighted sum over aligned spectral bands. Cubes and response maps must share their spectral and spatial grids. The command-line interface does not perform wavelength resampling.

The digital training and prediction commands operate on RAW measurements or paired spectral data. They do not perform sensor acquisition, fabrication, RCWA simulation or metasurface geometry optimization.
