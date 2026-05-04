"""
Evidential Deep Learning Loss

Based on: Sensoy et al. (2018) NeurIPS
"Evidential Deep Learning to Quantify Classification Uncertainty"

For segmentation with K binary channels (WT, TC, ET):

L_EDL = E[CrossEntropy under Dirichlet] + λ_t * KL[Dir(α) || Dir(1)]

Where:
    λ_t = min(1.0, t/warmup_epochs)  — annealing coefficient
    α~  = y + (1 - y) ⊙ α           — alpha with correct class set to 1

The KL term penalizes evidence for incorrect classes,
forcing the model to be uncertain rather than confidently wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha, K):
    """
    KL divergence between Dirichlet(alpha) and Dirichlet(1).
    Pushes evidence for incorrect classes toward zero.

    alpha: (B, K, H, W, D) Dirichlet parameters
    K    : number of classes
    """
    # Sum of alpha
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W, D)

    # KL = log(Gamma(S)/Gamma(K)) - sum(log(Gamma(alpha_k)/Gamma(1)))
    #    + sum((alpha_k - 1) * (digamma(alpha_k) - digamma(S)))
    ones = torch.ones_like(alpha)

    kl = (
        torch.lgamma(S)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + (
            (alpha - ones)
            * (torch.digamma(alpha) - torch.digamma(S))
        ).sum(dim=1, keepdim=True)
    )
    return kl  # (B, 1, H, W, D)


class EDLLoss(nn.Module):
    """
    Evidential Deep Learning Loss for binary segmentation.

    Applied independently per channel (WT, TC, ET).

    Args:
        num_classes   : K = 3
        warmup_epochs : epochs to ramp KL weight from 0 to 1 (default 10)

    Forward:
        alpha  : (B, K, H, W, D) Dirichlet parameters from EvidentialHead
        targets: (B, K, H, W, D) binary ground truth (0 or 1)
        epoch  : current training epoch (for KL annealing)

    Returns:
        scalar loss
    """
    def __init__(self, num_classes=3, warmup_epochs=10):
        super().__init__()
        self.K             = num_classes
        self.warmup_epochs = warmup_epochs

    def forward(self, alpha, targets, epoch):
        # Annealing coefficient: ramp from 0 to 1 over warmup_epochs
        lambda_t = min(1.0, epoch / self.warmup_epochs)

        # α~ = y + (1 - y) * α
        # For correct class: alpha stays, for wrong class: alpha → 1
        alpha_tilde = targets + (1 - targets) * alpha

        # S from α~
        S = alpha_tilde.sum(dim=1, keepdim=True)

        # Expected cross-entropy under Dirichlet
        # E[-log p_k] = digamma(S) - digamma(alpha_k)
        ece = (
            targets * (torch.digamma(S) - torch.digamma(alpha_tilde))
        ).sum(dim=1, keepdim=True)

        # KL divergence term
        kl = kl_divergence_dirichlet(alpha_tilde, self.K)

        # Combined EDL loss
        loss = (ece + lambda_t * kl).mean()

        return loss
