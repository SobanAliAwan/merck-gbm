# MERCK Framework

**Modality-Efficient, Resource-Constrained, Knowledge-distilled Glioblastoma Segmentation with Evidential Uncertainty**

[![IEEE Access Target](https://img.shields.io/badge/Target-IEEE%20Access-blue)](https://ieeeaccess.ieee.org/)
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

## Author

**Soban Ali Awan**  
Institute of Management and Sciences, Peshawar  
Research Mentor: Claude (Anthropic)

## Status

- [x] Environment setup
- [x] Data pipeline verified
- [x] Baseline 3D U-Net training
- [ ] Swin UNETR teacher
- [ ] MERCK modality adapters
- [ ] Evidential uncertainty head
- [ ] Full MERCK training
- [ ] Paper submission

## Citation

*Paper under preparation — IEEE Access 2026*
