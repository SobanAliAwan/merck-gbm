"""
EvidentialHead — Replaces sigmoid output with Dirichlet parameterization.

Based on: Sensoy et al. (2018) "Evidential Deep Learning to Quantify
Classification Uncertainty" NeurIPS 2018.

For each voxel and channel k:
    evidence    : e_k = softplus(z_k) >= 0
    Dirichlet   : alpha_k = e_k + 1 >= 1
    Strength    : S = sum(alpha_k)
    Probability : p_k = alpha_k / S
    Uncertainty : u = K / S  in [0, 1]

No learnable parameters — pure mathematical transformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential output head for K-channel binary segmentation.

    Args:
        num_classes: number of output channels (default 3: WT/TC/ET)

    Forward input:
        logits: (B, K, H, W, D)

    Returns:
        probs      : (B, K, H, W, D) — predicted probabilities
        uncertainty: (B, 1, H, W, D) — per-voxel uncertainty [0,1]
        alpha      : (B, K, H, W, D) — Dirichlet parameters (for EDL loss)
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.K = num_classes

    def forward(self, logits):
        evidence    = F.softplus(logits)
        alpha       = evidence + 1.0
        S           = alpha.sum(dim=1, keepdim=True)
        probs       = alpha / S
        uncertainty = self.K / S
        return probs, uncertainty, alpha
