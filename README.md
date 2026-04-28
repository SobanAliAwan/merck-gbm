# MERCK Framework

**Modality-Efficient, Resource-Constrained, Knowledge-distilled Glioblastoma Segmentation with Evidential Uncertainty**

[![Target](https://img.shields.io/badge/Target-High--impact%20peer--reviewed%20journal-blue)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![MONAI 1.5](https://img.shields.io/badge/MONAI-1.5-purple)](https://monai.io)

## Overview

MERCK is a novel deep learning framework for robust glioblastoma (GBM)
segmentation under missing MRI modalities. It combines:

1. **Knowledge Distillation** from a full-modality Swin UNETR teacher
2. **Modality-Specific Adapters** with learned prototype embeddings
3. **Evidential Deep Learning** for calibrated per-voxel uncertainty

## Clinical Motivation

In resource-limited hospitals (Pakistan, South Asia, Sub-Saharan Africa),
patients often cannot afford full 4-modality MRI. All state-of-the-art
models assume complete input and fail catastrophically on incomplete data.
MERCK is the first to combine KD + modality adapters + evidential UQ
for this specific clinical problem.

## Datasets

- **Primary:** BraTS 2023 Adult Glioma (1,251 cases)
- **External Validation:** BraTS-Africa (~60 cases)

## Current Results — Baseline 3D U-Net (B1)

| Region | Dice Score |
|--------|------------|
| Whole Tumor (WT) | 0.9132 |
| Tumor Core (TC) | 0.8912 |
| Enhancing Tumor (ET) | 0.8331 |
| **Mean** | **0.8792** |

*Epoch 46/100 — Training in progress*

## Research Progress

- [x] Environment setup
- [x] Data pipeline verified (BraTS 2023 GLI, 1251 cases)
- [x] Baseline 3D U-Net training (epoch 46/100, Dice 0.8792)
- [ ] Baseline training complete (epoch 100)
- [ ] Swin UNETR teacher network
- [ ] MERCK modality adapters
- [ ] Evidential uncertainty head
- [ ] Full MERCK training
- [ ] Paper submission

## Key Technical Notes

- BraTS 2023 uses labels 1,2,3 (not 1,2,4 as in older versions)
- Output: 3-channel sigmoid (WT/TC/ET) not 4-channel softmax
- Training: AdamW lr=1e-4, cosine annealing, mixed precision
- Hardware: Kaggle T4 GPU, ~22 min/epoch

## Author

**Soban Ali Awan**
Institute of Management and Sciences, Peshawar

## Citation

*Paper under preparation — Target: High-impact peer-reviewed journal (2026)*
