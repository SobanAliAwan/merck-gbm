"""
MERCK Training Script — Full Pipeline

Training procedure:
    1. At each batch, sample modality subset via dropout schedule:
       - 40%: full 4-modality  (subset_id=14)
       - 30%: 3 modalities     (random 3-modality subset)
       - 20%: 2 modalities     (random 2-modality subset)
       - 10%: 1 modality       (random single modality)
    2. Teacher receives full 4-modality input always (frozen)
    3. Student receives sampled subset
    4. Loss = L_EDL + λ_Dice * L_Dice + λ_KD * L_KD
    5. Save permanently to Kaggle dataset after every epoch

Warm-start: student backbone initialized from baseline U-Net weights
Teacher   : frozen Swin UNETR fine-tuned on BraTS 2023

Usage (on Kaggle):
    Run cells in order — setup, load checkpoints, train
"""

import os, time, warnings, shutil, json, subprocess, random
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import synapseclient
import zipfile

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, Spacingd, EnsureTyped, Compose,
    RandSpatialCropd, RandFlipd, RandScaleIntensityd,
    RandShiftIntensityd, MapTransform,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

set_determinism(seed=42)

# ── Modality subset definitions ───────────────────────────
MODALITY_SUBSETS = [
    (0,), (1,), (2,), (3,),
    (0,1), (0,2), (0,3), (1,2), (1,3), (2,3),
    (0,1,2), (0,1,3), (0,2,3), (1,2,3),
    (0,1,2,3),
]

SUBSETS_BY_SIZE = {
    1: [(0,), (1,), (2,), (3,)],
    2: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
    3: [(0,1,2), (0,1,3), (0,2,3), (1,2,3)],
    4: [(0,1,2,3)],
}

def sample_modality_subset():
    """
    Sample modality subset according to training schedule:
    40% full, 30% 3-modal, 20% 2-modal, 10% 1-modal
    """
    r = random.random()
    if r < 0.40:
        size = 4
    elif r < 0.70:
        size = 3
    elif r < 0.90:
        size = 2
    else:
        size = 1
    subset = random.choice(SUBSETS_BY_SIZE[size])
    subset_id = MODALITY_SUBSETS.index(subset)
    return subset, subset_id

def apply_modality_dropout(img_full, subset):
    """
    Given full 4-channel image, zero out missing modalities.
    img_full: (B, 4, H, W, D)
    subset  : tuple of present modality indices
    Returns : masked image + present_mask list
    """
    img = img_full.clone()
    present_mask = list(subset)
    for i in range(4):
        if i not in present_mask:
            img[:, i] = 0.0
    return img, present_mask


# ── Label converter ────────────────────────────────────────
class ConvertBraTS2023Labels(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            wt = ((seg==1)|(seg==2)|(seg==3)).float()
            tc = ((seg==1)|(seg==3)).float()
            et = (seg==3).float()
            d[key] = torch.cat([wt, tc, et], dim=0)
        return d


# ── MERCKInput ─────────────────────────────────────────────
class MERCKInput(nn.Module):
    def __init__(self, token_dim=32, adapter_dim=32):
        super().__init__()
        self.prototypes     = nn.Parameter(torch.zeros(4))
        self.modality_tokens = nn.Parameter(
            torch.randn(15, token_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, 4)
        self.adapter = nn.Sequential(
            nn.Conv3d(4, adapter_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm3d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(adapter_dim, 4, kernel_size=1, bias=False),
        )

    def forward(self, x, subset_id, present_mask):
        x = x.clone()
        for i in range(4):
            if i not in present_mask:
                x[:, i] = self.prototypes[i]
        token     = self.modality_tokens[subset_id]
        channel_w = torch.sigmoid(self.token_proj(token))
        x         = x * channel_w.view(1, 4, 1, 1, 1)
        return self.adapter(x)


# ── EvidentialHead ─────────────────────────────────────────
class EvidentialHead(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.K = num_classes

    def forward(self, logits):
        import torch.nn.functional as F
        evidence    = F.softplus(logits)
        alpha       = evidence + 1.0
        S           = alpha.sum(dim=1, keepdim=True)
        probs       = alpha / S
        uncertainty = self.K / S
        return probs, uncertainty, alpha


# ── MERCKModel ─────────────────────────────────────────────
class MERCKModel(nn.Module):
    def __init__(self, backbone_channels=(32,64,128,256,320),
                 token_dim=32, adapter_dim=32):
        super().__init__()
        self.merck_input = MERCKInput(token_dim, adapter_dim)
        self.backbone    = UNet(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=backbone_channels, strides=(2,2,2,2),
            num_res_units=2, dropout=0.1,
        )
        self.evid_head = EvidentialHead(num_classes=3)

    def forward(self, x, subset_id, present_mask):
        x               = self.merck_input(x, subset_id, present_mask)
        logits          = self.backbone(x)
        probs, u, alpha = self.evid_head(logits)
        return probs, u, alpha


# ── EDL Loss ───────────────────────────────────────────────
def edl_loss(alpha, targets, epoch, warmup_epochs=10):
    import torch.nn.functional as F
    lambda_t    = min(1.0, epoch / warmup_epochs)
    alpha_tilde = targets + (1 - targets) * alpha
    S           = alpha_tilde.sum(dim=1, keepdim=True)
    ece = (targets * (
        torch.digamma(S) - torch.digamma(alpha_tilde)
    )).sum(dim=1, keepdim=True)
    ones = torch.ones_like(alpha_tilde)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(torch.tensor(3.0, device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        + ((alpha_tilde - ones) * (
            torch.digamma(alpha_tilde) - torch.digamma(S)
        )).sum(dim=1, keepdim=True)
    )
    return (ece + lambda_t * kl).mean()


# ── KD Loss ────────────────────────────────────────────────
def kd_loss(student_probs, teacher_logits, T=3.0):
    import torch.nn.functional as F
    teacher_p = F.softmax(teacher_logits / T, dim=1)
    student_lp = torch.log(student_probs + 1e-10)
    kl = F.kl_div(student_lp, teacher_p,
                  reduction='batchmean', log_target=False)
    return T ** 2 * kl


# ── Dice Loss ──────────────────────────────────────────────
def dice_loss_fn(probs, targets, smooth=1e-5):
    intersection = (probs * targets).sum(dim=(2,3,4))
    union        = probs.sum(dim=(2,3,4)) + targets.sum(dim=(2,3,4))
    dice         = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ── Validate ───────────────────────────────────────────────
def validate_merck(model, loader, subset_id=14,
                   present_mask=None, device="cuda"):
    """Validate on full 4-modality by default (subset_id=14)."""
    if present_mask is None:
        present_mask = [0, 1, 2, 3]
    model.eval()
    wt_scores, tc_scores, et_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            img = torch.cat([batch["t1n"], batch["t1c"],
                             batch["t2w"], batch["t2f"]],
                            dim=1).to(device)
            seg = batch["seg"].to(device)

            with torch.amp.autocast("cuda"):
                def _pred(x):
                    p, _, _ = model(x, subset_id, present_mask)
                    return p
                out = sliding_window_inference(
                    img, (128,128,128), 2, _pred, overlap=0.5)

            pred = (out > 0.5).float()
            s = 1e-5
            for ch, store in zip(
                [0,1,2], [wt_scores, tc_scores, et_scores]
            ):
                p = pred[:,ch]; t = seg[:,ch]
                store.append(((2*(p*t).sum()+s) /
                               (p.sum()+t.sum()+s)).item())

    wt = sum(wt_scores)/len(wt_scores)
    tc = sum(tc_scores)/len(tc_scores)
    et = sum(et_scores)/len(et_scores)
    return {"WT": wt, "TC": tc, "ET": et, "mean": (wt+tc+et)/3}


# ── Permanent save ─────────────────────────────────────────
def save_merck_checkpoint(epoch, model, optimizer, scheduler,
                           best_dice, train_losses,
                           val_wt, val_tc, val_et,
                           ckpt_root, upload_dir,
                           kaggle_user, dataset_id):
    data = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_mean_dice":  best_dice,
        "train_losses":    train_losses,
        "val_wt":          val_wt,
        "val_tc":          val_tc,
        "val_et":          val_et,
    }
    torch.save(data, ckpt_root)
    torch.save(data, f"{upload_dir}/merck_best.pth")

    r = subprocess.run(
        f"kaggle datasets version -p {upload_dir} "
        f"-m 'merck_ep{epoch}_dice{best_dice:.4f}' --dir-mode zip",
        shell=True, capture_output=True, text=True
    )
    if "error" in r.stdout.lower() or "error" in r.stderr.lower():
        subprocess.run(
            f"kaggle datasets create -p {upload_dir} --dir-mode zip",
            shell=True, capture_output=True, text=True
        )
    print(f"  *** MERCK SAVED — epoch {epoch} | Dice {best_dice:.4f} ***")


# ══════════════════════════════════════════════════════════
# KAGGLE TRAINING LOOP
# Run this section on Kaggle after setting up data + models
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    DEVICE        = "cuda"
    MAX_EPOCHS    = 200
    VAL_INTERVAL  = 2
    LAMBDA_DICE   = 1.0
    LAMBDA_KD     = 0.5
    KD_TEMP       = 3.0
    LR            = 1e-4

    # Paths — set these on Kaggle
    BASELINE_CKPT = "/kaggle/input/merck-checkpoint/baseline_best.pth"
    TEACHER_CKPT  = "/kaggle/input/merck-teacher-checkpoint/teacher_best.pth"
    CKPT_ROOT     = "/kaggle/working/merck_best.pth"
    UPLOAD_DIR    = "/tmp/merck_upload"
    KAGGLE_USER   = "sobanaliawan"
    DATASET_ID    = f"{KAGGLE_USER}/merck-student-checkpoint"

    print("MERCK training script loaded.")
    print("Import this file on Kaggle and call the functions above.")
