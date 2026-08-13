# DeepNucleiNet

**Encoding Spatial Nuclear Patterns for Data-Efficient Necrosis Detection in Computational Neuropathology**

*Dasari Naga Raju, T K Srikanth, Shilpa Rao, Ramesh Kestur, Mahadevan A*

Accepted at **BrainWorks 2026** — MICCAI 2026 Satellite Workshop

---

## Overview

Necrosis detection in human meningioma is clinically significant but computationally underexplored, with no publicly available annotated dataset. Appearance-based representations encode tissue texture without explicitly modeling nuclear spatial distribution — omitting the histological criterion that defines necrosis.

**DeepNucleiNet** is a nuclei-guided dual-stream framework that combines H&E tile features with binarized nuclei-map features to directly encode the histological criterion of nuclear depletion, without requiring additional nuclear annotations.

![Architecture](![Architecture](https://github.com/raajuuu1998/DeepNucleiNet/raw/main/assets/architecture.pdf))

---

## Architecture

DeepNucleiNet consists of two parallel XceptionNet encoders:

- **Histology stream** — processes the RGB H&E tile → 2048-d features
- **Nuclei-map stream** — processes the binarized nuclei mask → 2048-d features
- **Concatenation** → 4096-d combined representation
- **RBF-kernel SVM** (C=10, γ=0.001) → Necrosis / Non-Necrosis

Both encoders are initialized from ImageNet-pretrained weights and fine-tuned for 50 epochs using a fully connected head (Adam, lr=1e-4, batch size=32). FC heads are discarded after fine-tuning; frozen features are passed to the SVM.

Nuclei maps are generated using **HoVerNet** (PanNuke checkpoint) via TIA Toolbox and binarized to binary masks (nuclei = white, background = black).

---

## Results

### Internal Evaluation — Inter-Patient Cross-Validation (n=4 folds, 8 patients)

| Model | F1-score | Balanced Acc | MCC | Sensitivity |
|---|---|---|---|---|
| Handcrafted (SVM) | 0.324 ± 0.224 | 0.445 ± 0.022 | -0.073 ± 0.045 | 0.395 ± 0.079 |
| Nuclei Density + LR | 0.593 ± 0.124 | 0.785 ± 0.064 | 0.428 ± 0.179 | 0.681 ± 0.211 |
| XceptionNet (FC) | 0.674 ± 0.198 | 0.791 ± 0.068 | 0.436 ± 0.190 | 0.800 ± 0.077 |
| Virchow2 + MLP | 0.730 ± 0.109 | 0.737 ± 0.177 | 0.438 ± 0.307 | 0.759 ± 0.053 |
| XceptionNet (H&E) | 0.847 ± 0.069 | 0.893 ± 0.055 | 0.681 ± 0.169 | 0.842 ± 0.097 |
| UNI2-h + MLP | 0.884 ± 0.068 | 0.913 ± 0.037 | 0.742 ± 0.162 | 0.901 ± 0.057 |
| XceptionNet (Nuclei) | 0.904 ± 0.044 | 0.911 ± 0.039 | 0.752 ± 0.189 | 0.901 ± 0.038 |
| **DeepNucleiNet** | **0.942 ± 0.030** | **0.936 ± 0.051** | **0.820 ± 0.183** | **0.945 ± 0.025** |

### Bootstrap Confidence Intervals (2000 tile-level resamples per fold)

| Model | F1-score (95% CI) | MCC (95% CI) |
|---|---|---|
| Handcrafted (SVM) | 0.322 [0.107, 0.540] | -0.073 [-0.111, -0.036] |
| XceptionNet (FC) | 0.674 [0.484, 0.863] | 0.435 [0.281, 0.591] |
| XceptionNet (H&E) | 0.847 [0.789, 0.917] | 0.683 [0.525, 0.804] |
| Virchow2 | 0.731 [0.630, 0.830] | 0.435 [0.128, 0.606] |
| UNI2-h | 0.885 [0.821, 0.948] | 0.741 [0.588, 0.846] |
| XceptionNet (Nuclei) | 0.903 [0.863, 0.949] | 0.753 [0.561, 0.868] |
| **DeepNucleiNet** | **0.942 [0.913, 0.970]** | **0.819 [0.639, 0.921]** |

### Cross-Tumor Generalization (no retraining)

| Dataset | Tumor Type | N | NN | Accuracy | Sensitivity | F1 | MCC |
|---|---|---|---|---|---|---|---|
| TCGA GBM | Grade 4 Glioma | 269 | 155 | 0.927 | 0.959 | 0.943 | 0.841 |
| DeepHisto | Mixed Glioma | 1916 | 2874 | 0.852 | 1.000 | 0.843 | 0.741 |
| TiGER | Breast TNBC | 63 | 94 | 0.701 | 0.651 | 0.635 | 0.382 |
| TCGA LGG | Low-Grade Glioma | 0 | 85,632 | 1.000 | — | — | — |

TCGA LGG produced zero necrosis predictions across 85,632 tiles (specificity = 1.000).

---

## Repository Structure

```
DeepNucleiNet/
├── deepnucleiNet/
│   ├── dataset.py              # Tile loading and fold assignments
│   ├── nuclei_map.py           # HoVerNet nuclei map generation
│   ├── model.py                # Dual-stream XceptionNet + SVM
│   ├── train.py                # Fine-tuning and SVM training
│   ├── evaluate.py             # Inter-patient CV + bootstrap CIs
│   └── external_validation.py # Cross-tumor evaluation
├── assets/
│   └── architecture.pdf        # Architecture diagram
├── requirements.txt
└── README.md
```

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate nuclei maps
```bash
python deepnucleiNet/nuclei_map.py \
    --he_dir data/necrosis \
    --out_dir data/masks/necrosis

python deepnucleiNet/nuclei_map.py \
    --he_dir data/non_necrosis \
    --out_dir data/masks/non_necrosis
```

### 3. Train (one fold)
```bash
python deepnucleiNet/train.py \
    --he_necrosis       data/necrosis \
    --he_non_necrosis   data/non_necrosis \
    --mask_necrosis     data/masks/necrosis \
    --mask_non_necrosis data/masks/non_necrosis \
    --out_dir           checkpoints \
    --fold              1
```

### 4. Evaluate all folds
```bash
python deepnucleiNet/evaluate.py \
    --he_necrosis       data/necrosis \
    --he_non_necrosis   data/non_necrosis \
    --mask_necrosis     data/masks/necrosis \
    --mask_non_necrosis data/masks/non_necrosis \
    --ckpt_dir          checkpoints
```

### 5. External validation
```bash
python deepnucleiNet/external_validation.py \
    --he_dir     external/TCGA_GBM/he \
    --mask_dir   external/TCGA_GBM/masks \
    --labels_csv external/TCGA_GBM/labels.csv \
    --ckpt_dir   checkpoints/fold_1 \
    --dataset    TCGA_GBM
```

---

## Data

This repository releases **code only**. No patient data, tile images, or embeddings are included.

The internal meningioma dataset was obtained from a tertiary neuro-oncology center:
- 10,233 tiles (256×256 at 20× magnification, 0.5 μm/pixel)
- 4,040 necrotic / 6,193 non-necrotic
- 8 patients, strict inter-patient cross-validation

External datasets used for validation (publicly available):
- [TCGA GBM / LGG](https://portal.gdc.cancer.gov/)
- [DeepHisto](https://zenodo.org/record/7941080)
- [TiGER](https://tiger.grand-challenge.org/)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{raju2026deepnucleiNet,
  title     = {DeepNucleiNet: Encoding Spatial Nuclear Patterns for 
               Data-Efficient Necrosis Detection in Computational Neuropathology},
  author    = {Dasari Naga Raju and T K Srikanth and Shilpa Rao and 
               Ramesh Kestur and Mahadevan A},
  booktitle = {BrainWorks Workshop, MICCAI 2026},
  year      = {2026}
}
```

---

## License

Code released under the MIT License.
