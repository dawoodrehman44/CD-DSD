# CD-DSD: Counterfactual Diffusion-based Domain Shift Detection for Chest X-ray Analysis

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/dawoodrehman44/CD-DSD/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/ICASSP-2026-green.svg" alt="ICASSP 2026">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
</p>

> **CD-DSD: Counterfactual Diffusion-based Domain Shift Detection for Chest X-ray Analysis**
>
> *ICASSP 2026 — under review*

---

## Overview

**CD-DSD** is a domain shift detection framework for chest X-ray (CXR) analysis. It addresses a core clinical deployment challenge: a model trained on one hospital's CXR distribution (e.g., CheXpert) may silently degrade when applied to images from a different scanner, acquisition protocol, or institution. CD-DSD detects this degradation *before* prediction errors occur, and attributes the shift to interpretable clinical factors.

The framework decomposes per-image uncertainty into two complementary signals:

- **U_domain** — diffusion denoising error + high-frequency energy ratio (HFER), measuring how far the test image lies from the training distribution in both reconstruction and spectral space.
- **U_semantic** — MC-Dropout predictive entropy, measuring the classifier's inherent ambiguity about the pathology prediction.

A lightweight **FactorMLP** then attributes the detected domain shift to one of four interpretable factors: *brightness/contrast*, *noise/texture*, *scanner artefact*, or *global structure*. Finally, an **SDEdit-based counterfactual** partially corrects the test image back toward the source domain, providing visual evidence of what changed.

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="CD-DSD Architecture and Qualitative Results">
</p>

**Figure A (top):** The CD-DSD inference pipeline. A test CXR is processed in parallel by (i) a frozen DenseNet-121 with MC-Dropout for U_semantic, and (ii) a frozen DDPM that computes multi-timestep denoising error and HFER for U_domain. The two signals are fused into U_total, and FactorMLP attributes the shift to interpretable factors. An SDEdit partial correction produces the domain-corrected counterfactual x*.

**Figure B (bottom):** Qualitative results across three corruption types (blur, noise, contrast). Each row shows the original test image followed by five factor-level partial corrections, ending with the full counterfactual correction (purple border). The framework correctly localises each corruption type.

---

## Key Results

### Domain Shift Detection AUROC vs. 12 Baselines

| Method | Brightness | Darkness | Noise | Blur | Contrast | Combined | **Mean** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MSP (2017) | 0.825 | 0.830 | 0.735 | 0.866 | 0.806 | 0.804 | 0.811 |
| Mahalanobis (2018) | 0.560 | 0.585 | 1.000 | 0.996 | 0.386 | 0.938 | 0.744 |
| Energy (2020) | 0.823 | 0.829 | 0.667 | 0.862 | 0.804 | 0.801 | 0.798 |
| ReAct (2021) | 0.832 | 0.819 | 0.884 | 0.871 | 0.763 | 0.787 | 0.826 |
| ViM (2022) | 0.873 | 0.876 | 1.000 | 0.986 | 0.820 | 0.910 | 0.911 |
| KNN (2022) | 0.739 | 0.814 | 1.000 | 0.989 | 0.701 | 0.953 | 0.866 |
| DICE (2022) | 0.820 | 0.818 | 0.930 | 0.856 | 0.805 | 0.789 | 0.836 |
| GEN (2023) | 0.179 | 0.235 | 0.005 | 0.154 | 0.269 | 0.135 | 0.163 |
| ASH (2023) | 0.826 | 0.834 | 0.821 | 0.895 | 0.813 | 0.834 | 0.837 |
| DDA (2023) | 0.566 | 0.613 | 0.963 | 0.692 | 0.564 | 0.589 | 0.664 |
| DiffPath (2024)* | 0.524 | 0.506 | 0.677 | 0.119 | 0.998 | 0.250 | 0.512 |
| EigenScore (2026)* | 0.639 | 0.570 | 0.622 | 0.345 | 0.964 | 0.424 | 0.594 |
| **CD-DSD (ours)** | **1.000** | **0.993** | **1.000** | **0.995** | **1.000** | **0.986** | **0.996** |

*Simplified reproductions of the published method. All methods use the same frozen DenseNet-121 backbone.*

### Bootstrap 95% CI on AUROC (B = 1,000)

| Method | Overall | Brightness | Darkness | Noise | Blur | Contrast |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CD-DSD** | **0.997** | 0.997 | 0.997 | 1.000 | 0.992 | 1.000 |
| ViM | 0.834 | 0.610 | 0.814 | 1.000 | 0.959 | 0.786 |
| KNN | 0.850 | 0.541 | 0.883 | 1.000 | 0.991 | 0.836 |
| MSP | 0.600 | 0.578 | 0.685 | 0.419 | 0.661 | 0.657 |

### Downstream Clinical Utility (MIMIC-CXR, N=500)

| Signal | Spearman ρ with BCE | p-value |
|---|:---:|:---:|
| U_total | 0.411 | < 0.0001 |
| U_semantic | 0.470 | < 0.0001 |
| U_domain | 0.033 | 0.459 (n.s.) |

Images in the highest uncertainty quartile have **2.3× higher prediction error** than those in the lowest quartile (BCE: 0.348 → 0.788).

---

## Installation

```bash
git clone https://github.com/dawoodrehman44/CD-DSD.git
cd CD-DSD
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+ with CUDA are recommended. The DDPM domain scoring requires a GPU.

---

## Dataset Setup

CD-DSD is trained on [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and evaluated on [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and [NIH ChestX-Ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC). After downloading, update the paths in `cd_dsd/config.py`:

```python
CHEXPERT_IMAGE_ROOT = "/path/to/CheXpert-v1.0/train/"
CHEXPERT_TRAIN_CSV  = "/path/to/chexpert_train.csv"
CHEXPERT_VALID_CSV  = "/path/to/chexpert_valid.csv"

MIMIC_IMAGE_ROOT    = "/path/to/mimic-cxr/jpg"
MIMIC_TRAIN_CSV     = "/path/to/mimic_train.csv"
MIMIC_VALID_CSV     = "/path/to/mimic_valid.csv"

NIH_IMAGE_ROOT      = "/path/to/ChestX-Ray8/images"
NIH_CSV             = "/path/to/Data_Entry_2017_v2020.csv"
```

---

## Pretrained Checkpoints

| Model | Description | Download |
|---|---|---|
| `classifier_best.pt` | DenseNet-121 (8-class CheXpert, MC-Dropout) | [HuggingFace / coming soon] |
| `diffusion_best.pt` | DDPM trained on CheXpert (224×224) | [HuggingFace / coming soon] |
| `factor_classifier.pt` | FactorMLP (4-factor attribution) | [HuggingFace / coming soon] |

Place downloaded checkpoints in `checkpoints/`.

---

## Quick Start

### Run the demo notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

Open the notebook and run the **Quick Setup** cell (Cell 1). This loads all components, fits the baseline suite, and runs all experiments without re-executing the class-definition cells.

### Run inference on a single image

```python
import torch
from cd_dsd import Config, CDDSDDiagnoser, get_eval_transform

cfg       = Config()
diagnoser = CDDSDDiagnoser(cfg)
transform = get_eval_transform(cfg.IMAGE_SIZE, cfg.MEAN, cfg.STD)

from PIL import Image
img = Image.open("path/to/cxr.jpg").convert("RGB")
x   = transform(img).unsqueeze(0).to(diagnoser.device)

results = diagnoser.diagnose_batch(x, domain="external", save_vis=True)
r = results[0]
print(f"U_total:   {r['u_total']:.4f}")
print(f"U_domain:  {r['u_domain']:.4f}")
print(f"U_semantic:{r['u_semantic']:.4f}")
print(f"Top factor:{max(r['factor_attributions'], key=r['factor_attributions'].get)}")
```

---

## Training

### 1. Train the classifier

```python
from cd_dsd import Config, CheXpertDataset, get_train_transform, MCDropoutClassifier
# See notebooks/demo.ipynb — Training section
```

### 2. Train the diffusion model

```python
from cd_dsd import Config, DiffusionModel, UNet
# See notebooks/demo.ipynb — Diffusion Training section
```

### 3. Train the FactorMLP

The FactorMLP is trained automatically on first run of `CDDSDDiagnoser` if no checkpoint exists. It trains on synthetic corruptions of clean CheXpert images, requiring no extra annotation.

---

## Module Structure

```
CD-DSD/
├── cd_dsd/
│   ├── config.py          # All hyperparameters and dataset paths
│   ├── datasets.py        # CheXpert, MIMIC-CXR, NIH dataset classes
│   ├── classifier.py      # MCDropoutClassifier (DenseNet-121)
│   ├── unet.py            # U-Net architecture for the DDPM
│   ├── diffusion.py       # DDPM forward/reverse process, DDIM sampling, SDEdit
│   ├── diagnoser.py       # CDDSDDiagnoser: calibration, scoring, visualisation
│   ├── factor_mlp.py      # FactorMLP, FactorAttributor, corruption utilities
│   ├── baselines.py       # 12 OOD baseline methods (MSP → EigenScore)
│   ├── utils.py           # Logging, CSV saving, directory helpers
│   └── __init__.py        # Public API
├── notebooks/
│   └── demo.ipynb         # End-to-end demo: all 8 experiments
├── assets/
│   └── architecture.png   # Architecture figure
├── requirements.txt
└── README.md
```

---

## Method Summary

CD-DSD combines three components that each address a distinct failure mode of existing OOD detectors on medical images:

| Component | What it detects | Why existing methods miss it |
|---|---|---|
| DDPM denoising error | Structural & photometric shifts | Feature-space methods are blind to pixel-level statistics |
| HFER (FFT blur ratio) | Resolution / blur degradation | Denoising error alone is insensitive to sharpness |
| MC-Dropout entropy | Pathology ambiguity | Domain score alone cannot distinguish scanner vs. disease uncertainty |

The calibration step estimates the source distribution statistics on clean CheXpert images, converting raw scores to z-scores. This makes the threshold interpretation **modality-agnostic** and eliminates the need for threshold tuning on target data.

---

## Baselines Implemented

All 12 baselines are re-implemented in `cd_dsd/baselines.py` using the same frozen DenseNet-121 backbone:

| Method | Year | Venue |
|---|:---:|---|
| MSP | 2017 | ICLR |
| Mahalanobis | 2018 | NeurIPS |
| Energy | 2020 | NeurIPS |
| ReAct | 2021 | NeurIPS |
| ViM | 2022 | CVPR |
| KNN | 2022 | ICML |
| DICE | 2022 | ECCV |
| GEN | 2023 | CVPR |
| ASH | 2023 | ICLR |
| DDA | 2023 | CVPR |
| DiffPath* | 2024 | NeurIPS |
| EigenScore* | 2026 | ICLR |

*Simplified reproductions.*

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rehman2026cddsd,
  title     = {CD-DSD: Counterfactual Diffusion-based Domain Shift Detection
               for Chest X-ray Analysis},
  author    = {Rehman, Dawood and others},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal
               Processing (ICASSP)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) — Stanford ML Group
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) — PhysioNet
- [NIH ChestX-Ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — NIH Clinical Center
- DDPM implementation inspired by [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)
- SDEdit from [Meng et al. (2022)](https://arxiv.org/abs/2108.01073)
- OpenOOD benchmark from [Zhang et al. (2023)](https://arxiv.org/abs/2306.09301)
