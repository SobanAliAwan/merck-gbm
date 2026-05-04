"""
MERCKModel — Full Architecture

Components:
    1. MERCKInput   — modality adapter with learned prototypes
    2. U-Net Backbone — shared across all 15 modality subsets
    3. EvidentialHead — calibrated uncertainty in single forward pass

Total parameters: ~12.87M
Adapter overhead: 0.03% of total

Deviation from document (approved):
    - Prototypes are scalar values (not full 3D volumes)
      Reason: compact, faster to learn, same conceptual purpose
    - out_channels=3 sigmoid → replaced by EvidentialHead
      Reason: BraTS regions overlap, evidential is the core novelty
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from merck.models.merck_input import MERCKInput, MODALITY_SUBSETS, get_subset_id
from merck.models.evidential_head import EvidentialHead


class MERCKModel(nn.Module):
    """
    Full MERCK segmentation model.

    Args:
        backbone_channels: U-Net feature channels
        token_dim        : modality token dimension
        adapter_dim      : adapter intermediate channels
    """
    def __init__(self,
                 backbone_channels=(32, 64, 128, 256, 320),
                 token_dim=32,
                 adapter_dim=32):
        super().__init__()

        self.merck_input = MERCKInput(
            token_dim=token_dim,
            adapter_dim=adapter_dim,
        )
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=backbone_channels,
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
        )
        self.evid_head = EvidentialHead(num_classes=3)

    def forward(self, x, subset_id, present_mask):
        """
        x           : (B, 4, H, W, D) — zeros where modality absent
        subset_id   : int 0-14
        present_mask: list of present modality indices

        Returns:
            probs      : (B, 3, H, W, D)
            uncertainty: (B, 1, H, W, D)
            alpha      : (B, 3, H, W, D)
        """
        x              = self.merck_input(x, subset_id, present_mask)
        logits         = self.backbone(x)
        probs, u, alpha = self.evid_head(logits)
        return probs, u, alpha

    def get_segmentation(self, probs, threshold=0.5):
        return (probs > threshold).float()

    def count_params(self):
        total   = sum(p.numel() for p in self.parameters())
        inp     = sum(p.numel() for p in self.merck_input.parameters())
        back    = sum(p.numel() for p in self.backbone.parameters())
        return {"total": total, "merck_input": inp, "backbone": back}
