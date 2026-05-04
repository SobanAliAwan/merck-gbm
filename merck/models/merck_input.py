"""
MERCKInput — Modality Adapter with Learned Prototypes

Fills missing modalities with learned scalar prototypes,
conditions features on subset identity via modality tokens,
and applies a lightweight 2-layer adapter.

Parameters: ~4,200 (0.03% of total model)
"""

import torch
import torch.nn as nn

# All 15 non-empty subsets of {T1n(0), T1c(1), T2w(2), T2f(3)}
MODALITY_SUBSETS = [
    (0,), (1,), (2,), (3,),
    (0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
    (0,1,2), (0,1,3), (0,2,3), (1,2,3),
    (0,1,2,3),
]

def get_subset_id(present_modalities):
    """Returns subset ID (0-14) for given present modality indices."""
    return MODALITY_SUBSETS.index(tuple(sorted(present_modalities)))


class MERCKInput(nn.Module):
    """
    Modality-Efficient Input Processor.

    Args:
        token_dim  : dimension of modality token (default 32)
        adapter_dim: intermediate channels in adapter (default 32)

    Forward:
        x           : (B, 4, H, W, D) — zeros where modality absent
        subset_id   : int 0-14
        present_mask: list of present modality indices e.g. [0, 2]

    Returns:
        (B, 4, H, W, D) processed tensor ready for U-Net backbone
    """
    def __init__(self, token_dim=32, adapter_dim=32):
        super().__init__()

        # Learned prototypes — one scalar per modality
        self.prototypes = nn.Parameter(torch.zeros(4))

        # Modality tokens — one per subset configuration
        self.modality_tokens = nn.Parameter(
            torch.randn(15, token_dim) * 0.02
        )

        # Token projection to channel weights
        self.token_proj = nn.Linear(token_dim, 4)

        # Lightweight 2-layer adapter
        self.adapter = nn.Sequential(
            nn.Conv3d(4, adapter_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(adapter_dim, 4, kernel_size=1, bias=False),
        )

    def forward(self, x, subset_id, present_mask):
        x = x.clone()

        # Fill missing modalities with learned prototype value
        for i in range(4):
            if i not in present_mask:
                x[:, i] = self.prototypes[i]

        # Token conditioning — scale channels by subset identity
        token     = self.modality_tokens[subset_id]
        channel_w = torch.sigmoid(self.token_proj(token))
        x         = x * channel_w.view(1, 4, 1, 1, 1)

        # Adapter
        return self.adapter(x)
