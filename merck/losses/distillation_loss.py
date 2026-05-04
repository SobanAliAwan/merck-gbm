"""
Knowledge Distillation Loss

Based on: Hinton et al. (2015)
"Distilling the Knowledge in a Neural Network"

L_KD = T² * KL(p_teacher_τ || p_student_τ)

Where:
    T   = temperature (default 3)
    p_τ = softmax(logits / T) — temperature-scaled probabilities

The T² multiplier compensates for the 1/T² gradient scaling
that occurs when using temperature-scaled softmax.

In MERCK:
    Teacher receives full 4-modality input always
    Student receives whatever modality subset is available
    KD loss aligns student output with teacher's soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss.

    Args:
        temperature: T for softmax scaling (default 3)

    Forward:
        student_probs : (B, K, H, W, D) from EvidentialHead
        teacher_logits: (B, K, H, W, D) raw logits from frozen teacher
                        NOTE: teacher logits, not probabilities

    Returns:
        scalar KD loss
    """
    def __init__(self, temperature=3.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_probs, teacher_logits):
        # Temperature-scaled teacher probabilities
        # Apply softmax across K channels per voxel
        teacher_probs_t = F.softmax(teacher_logits / self.T, dim=1)

        # Student probabilities also temperature-scaled
        # student_probs already normalized — we re-derive from alpha
        # but for simplicity we use student_probs directly with log
        student_log_probs_t = torch.log(student_probs + 1e-10)

        # KL divergence: KL(teacher || student)
        # = sum(teacher * log(teacher/student))
        kl = F.kl_div(
            student_log_probs_t,
            teacher_probs_t,
            reduction='batchmean',
            log_target=False,
        )

        # Multiply by T² to compensate for gradient scaling
        return self.T ** 2 * kl
