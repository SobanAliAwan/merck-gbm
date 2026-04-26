"""
Baseline 3D U-Net training script.
Trains on full 4-modality BraTS 2023 input.
This is Baseline B1 in the MERCK paper.

Usage:
    python scripts/train_baseline.py \
        --data_root /path/to/BraTS2023/TrainingData \
        --checkpoint_dir ./checkpoints \
        --max_epochs 100

Deviation from design document:
    - out_channels=3 (not 4) using sigmoid activation per channel.
      Reason: BraTS regions (WT/TC/ET) are overlapping composites,
      not mutually exclusive classes. Sigmoid per channel is correct;
      softmax assumes mutual exclusivity which is mathematically wrong
      for overlapping BraTS regions. All top BraTS challenge papers
      use this approach.
    - BraTS 2023 uses labels 1,2,3 (not 1,2,4 as in older versions).
      Verified directly from data.
"""

import os
import time
import argparse
import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.utils import set_determinism

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merck.data.transforms import get_train_transforms, get_val_transforms
from merck.data.brats_dataset import build_data_dicts, split_data, get_dataloaders
from merck.evaluation.metrics import validate


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline 3D U-Net")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(seed=args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    all_cases = build_data_dicts(args.data_root)
    train_dicts, val_dicts, test_dicts = split_data(all_cases)
    train_loader, val_loader = get_dataloaders(
        train_dicts, val_dicts,
        get_train_transforms(), get_val_transforms()
    )
    print(f"Train: {len(train_dicts)} | Val: {len(val_dicts)} | Test: {len(test_dicts)}")

    # Model
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    loss_fn   = DiceCELoss(to_onehot_y=False, sigmoid=True,
                            squared_pred=True, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-5)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )

    # Load checkpoint if exists
    ckpt_path      = os.path.join(args.checkpoint_dir, "baseline_best.pth")
    start_epoch    = 1
    best_mean_dice = 0.0
    train_losses   = []
    val_wt, val_tc, val_et = [], [], []

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch    = ck["epoch"] + 1
        best_mean_dice = ck["best_mean_dice"]
        train_losses   = ck.get("train_losses", [])
        val_wt         = ck.get("val_wt", [])
        val_tc         = ck.get("val_tc", [])
        val_et         = ck.get("val_et", [])
        print(f"Resumed from epoch {ck['epoch']} | "
              f"Best Dice: {best_mean_dice:.4f}")

    print("=" * 60)
    print(f"Training epochs {start_epoch} to {args.max_epochs}")
    print("=" * 60)

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0

        for batch in train_loader:
            img = torch.cat(
                [batch["t1n"], batch["t1c"],
                 batch["t2w"], batch["t2f"]], dim=1
            ).to(device)
            seg = batch["seg"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out  = model(img)
                loss = loss_fn(out, seg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()

        scheduler.step()
        avg_loss = ep_loss / len(train_loader)
        train_losses.append(avg_loss)
        elapsed = (time.time() - t0) / 60

        if epoch % args.val_interval == 0:
            sc = validate(model, val_loader, device=str(device))
            val_wt.append(sc["WT"])
            val_tc.append(sc["TC"])
            val_et.append(sc["ET"])

            print(
                f"Ep {epoch:3d}/{args.max_epochs} | "
                f"Loss {avg_loss:.4f} | "
                f"WT {sc['WT']:.4f} | "
                f"TC {sc['TC']:.4f} | "
                f"ET {sc['ET']:.4f} | "
                f"Mean {sc['mean']:.4f} | "
                f"{elapsed:.1f}min"
            )

            if sc["mean"] > best_mean_dice:
                best_mean_dice = sc["mean"]
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_mean_dice":  best_mean_dice,
                    "train_losses":    train_losses,
                    "val_wt":          val_wt,
                    "val_tc":          val_tc,
                    "val_et":          val_et,
                }, ckpt_path)
                print(f"  *** BEST saved: {best_mean_dice:.4f} ***")
        else:
            print(f"Ep {epoch:3d}/{args.max_epochs} | "
                  f"Loss {avg_loss:.4f} | {elapsed:.1f}min")

    print("=" * 60)
    print(f"DONE. Best mean Dice: {best_mean_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
