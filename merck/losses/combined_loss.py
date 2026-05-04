"""
MERCK Combined Loss

L_total = L_EDL + λ_Dice * L_Dice + λ_KD * L_KD

Where:
    L_EDL  : Evidential loss (uncertainty calibration)
    L_Dice : Standard Dice loss (segmentation accuracy)
    L_KD   : Knowledge distillation loss (teacher-student alignment)

Default weights from document:
    λ_Dice = 1.0
    λ_KD   = 0.5
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss

from merck.losses.evidential_loss import EDLLoss
from merck.losses.distillation_loss import KDLoss


class MERCKLoss(nn.Module):
    """
    Combined MERCK training loss.

    Args:
        lambda_dice     : weight for Dice loss (default 1.0)
        lambda_kd       : weight for KD loss (default 0.5)
        temperature     : KD temperature (default 3.0)
        warmup_epochs   : EDL KL annealing epochs (default 10)

    Forward:
        alpha          : (B, 3, H, W, D) Dirichlet params from student
        targets        : (B, 3, H, W, D) binary ground truth
        student_probs  : (B, 3, H, W, D) student probabilities
        teacher_logits : (B, 3, H, W, D) teacher raw logits (or None)
        epoch          : current epoch (for EDL annealing)

    Returns:
        dict with keys: total, edl, dice, kd
    """
    def __init__(self,
                 lambda_dice=1.0,
                 lambda_kd=0.5,
                 temperature=3.0,
                 warmup_epochs=10):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_kd   = lambda_kd

        self.edl_loss  = EDLLoss(num_classes=3,
                                  warmup_epochs=warmup_epochs)
        self.dice_loss = DiceLoss(sigmoid=False,
                                   squared_pred=True,
                                   reduction="mean")
        self.kd_loss   = KDLoss(temperature=temperature)

    def forward(self, alpha, targets, student_probs,
                teacher_logits=None, epoch=1):

        # EDL loss — uncertainty calibration
        l_edl = self.edl_loss(alpha, targets, epoch)

        # Dice loss — segmentation accuracy
        l_dice = self.dice_loss(student_probs, targets)

        # KD loss — teacher-student alignment
        if teacher_logits is not None:
            l_kd = self.kd_loss(student_probs, teacher_logits)
        else:
            l_kd = torch.tensor(0.0, device=alpha.device)

        # Combined
        total = l_edl + self.lambda_dice * l_dice + self.lambda_kd * l_kd

        return {
            "total": total,
            "edl":   l_edl,
            "dice":  l_dice,
            "kd":    l_kd,
        }
