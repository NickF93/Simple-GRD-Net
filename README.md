# Simple-GRD-Net (Official Rebuild)

Official configurable implementation of:
- **GRD-Net (2023)**: Generative-Reconstructive-Discriminative anomaly detection with ROI-aware attention.
- **DeepIndustrial-SN (2026)**: runtime-optimized configuration focused on the generative branch for fast patch-level decisions.

This repository now provides one **YAML-configurable pipeline** with two paper profiles, a full **PyTorch backend**, and a **TensorFlow/Keras scaffold API** for future parity.

## 1. Paper Mapping

Source documents in this repository:
- `docs/International_Journal_of_Intelligent_Systems_-_2023_-_Ferrari_-_GRD‐Net.pdf`
- `docs/DeepIndustrial_SN.pdf`

### 1.1 GRD-Net 2023 profile (`grdnet_2023_full`)

Core design:
1. Generator `G` with encoder-decoder-encoder topology.
2. Discriminator `C` for adversarial feature matching.
3. Discriminative U-Net `Δ` for anomaly map prediction.
4. ROI attention through intersection in discriminative loss.

Paper-level equations implemented as profile defaults:

Synthetic perturbation:

$$
(X_n, M) = P_q(X)
$$

Adversarial feature loss:

$$
L_{adv} = L_2\big(F_C(X), F_C(\hat{X})\big)
$$

Contextual loss:

$$
L_{con} = w_a L_1(X, \hat{X}) + w_b L_{SSIM}(X, \hat{X})
$$

Encoder consistency:

$$
L_{enc} = L_1(z, \hat{z})
$$

Generator objective:

$$
L_{gen} = w_1 L_{adv} + w_2 L_{con} + w_3 L_{enc}
$$

ROI intersection for discriminative supervision:

$$
I = A_{discr} \odot ROI_{input}
$$

Total full-profile objective:

$$
L_{tot} = L_{gen} + FL(I, M)
$$

where `FL` is focal loss.

### 1.2 DeepIndustrial-SN 2026 profile (`deepindustrial_sn_2026`)

This profile keeps the generative adversarial core, with deployment-oriented constraints and score computation.

Perturbation mechanism:

$$
X^* = (1 - M)\odot X + (1-\beta)M\odot X + \beta N, \quad \beta\sim U(0.5, 1.0)
$$

Noise regularization term:

$$
L_{nse} = w_4 \cdot L_2\left(\left|(1-\beta)\hat{M}\odot\hat{X} - M\odot X\right|,\, \beta N\right)
$$

Contextual update (Huber + SSIM):

$$
L_{con} = w_a L_{Huber}(X, \hat{X}) + w_b L_{SSIM}(X, \hat{X})
$$

Anomaly score:

$$
\phi = 1 - SSIM(X, \hat{X})
$$

Heatmap:

$$
H = \lvert X - \hat{X} \rvert
$$

with per-sample min-max normalization.

## 2. Architecture

Top-level package (`grdnet/`) is layered and modular:

- `grdnet/config`: strict typed schema + YAML loader.
- `grdnet/core`: deterministic logging, reproducibility, exceptions.
- `grdnet/data`: MVTec-like adapter, custom MVTec-like extension point, patch extraction.
- `grdnet/models/pytorch`: residual generator/discriminator and U-Net segmentator.
- `grdnet/models/tensorflow`: scaffold placeholders for API parity planning.
- `grdnet/losses`: centralized objective definitions.
- `grdnet/backends`: strategy/factory backend abstraction.
- `grdnet/training`: loop + checkpoints.
- `grdnet/inference`: calibrate/eval/infer workflows.
- `grdnet/metrics`: thresholding and score metrics.
- `grdnet/reporting`: structured console + CSV outputs.
- `grdnet/pipeline`: command runners.

## 3. Dataset Contract

Current official adapter expects **MVTec-like layout**:

```text
root/
  good/
    *.png
  defect_type_a/
    *.png
  defect_type_b/
    *.png
```

Optional ROI and mask roots must mirror the same class/file structure:

```text
roi_root/
  good/
    file.png
mask_root/
  defect_type_a/
    file.png
```

Notes:
- Training can be `nominal_train_only=true`.
- `roi_root` missing defaults ROI to all ones.
- `mask_root` missing defaults ground-truth mask to zeros.

## 4. Config Profiles

Provided configs:
- `configs/profiles/grdnet_2023_full.yaml`
- `configs/profiles/deepindustrial_sn_2026.yaml`

Backend stubs:
- `configs/backends/pytorch.yaml`
- `configs/backends/tensorflow_scaffold.yaml`

## 5. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 6. CLI Usage

Validate configuration:

```bash
grdnet validate-config -c configs/profiles/grdnet_2023_full.yaml
```

Train:

```bash
grdnet train -c configs/profiles/grdnet_2023_full.yaml
```

Calibrate threshold:

```bash
grdnet calibrate -c configs/profiles/deepindustrial_sn_2026.yaml --checkpoint artifacts/checkpoints/deepindustrial_sn_2026/epoch_0010.pt
```

Evaluate:

```bash
grdnet eval -c configs/profiles/deepindustrial_sn_2026.yaml --checkpoint artifacts/checkpoints/deepindustrial_sn_2026/epoch_0010.pt
```

Infer:

```bash
grdnet infer -c configs/profiles/deepindustrial_sn_2026.yaml --checkpoint artifacts/checkpoints/deepindustrial_sn_2026/epoch_0010.pt
```

## 7. Output Artifacts

Artifacts are written under `training.output_dir` and `training.checkpoint_dir` from YAML:

- `metrics.csv`: epoch and evaluation metrics.
- `predictions.csv`: patch-wise scores/predictions.
- `epoch_XXXX.pt`: full training checkpoints.

## 8. Backend Status

- `pytorch`: full v1 implementation.
- `tensorflow_scaffold`: API placeholder only. Commands requiring compute raise deterministic `NotImplementedError` with explicit guidance.

## 9. Reproducibility

Determinism controls are first-class config fields:

- `system.seed`
- `system.deterministic`

The implementation seeds Python, NumPy, and PyTorch RNGs and configures deterministic backend behavior when requested.

## 10. Theory References

Primary references implemented in this repo:

1. **Ferrari et al. (2023)**, *GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module*, International Journal of Intelligent Systems. PDF in `docs/International_Journal_of_Intelligent_Systems_-_2023_-_Ferrari_-_GRD‐Net.pdf`.
2. **Ferrari et al. (2026)**, *DeepIndustrial_SN* (runtime-focused GRD-Net variant). PDF in `docs/DeepIndustrial_SN.pdf`.

Secondary conceptual references used by both papers and reflected in this codebase:
- GANomaly-style E-D-E reconstruction consistency.
- DRÆM-inspired synthetic perturbation via Perlin-mask anomaly simulation.
- SSIM-guided reconstruction quality and heatmap-based anomaly localization.

## 11. Quality Gates (project policy)

The rebuild is structured for strict gates:
- lint (`ruff`)
- type checks (`mypy`)
- tests (`pytest`)
- vulnerability and unsafe-code checks (planned CI integration)
- coverage target policy on core scope (planned CI integration)

## 12. Important Current Scope

- Full scientific/engineering implementation is currently **PyTorch-first**.
- TensorFlow/Keras is scaffolded for interface stability and future parity work.
- Public dataset support currently targets MVTec-style structures.
