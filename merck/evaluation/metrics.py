import torch
from monai.inferers import sliding_window_inference


def dice_score(pred, target, smooth=1e-5):
    """
    Computes Dice score for a single channel.
    pred and target: binary tensors of same shape.
    """
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )


def validate(model, loader, device="cuda"):
    """
    Runs sliding window inference on all validation cases.
    Returns dict with mean WT, TC, ET Dice scores.

    Uses manual per-channel Dice computation to handle
    cases where ET voxels may be zero (avoids MONAI
    DiceMetric shape mismatch bug with get_not_nans=True).
    """
    model.eval()
    wt_scores, tc_scores, et_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            img = torch.cat(
                [batch["t1n"], batch["t1c"],
                 batch["t2w"], batch["t2f"]], dim=1
            ).to(device)
            seg = batch["seg"].to(device)

            with torch.amp.autocast("cuda"):
                out = sliding_window_inference(
                    img, (128, 128, 128), 2,
                    model, overlap=0.5
                )

            pred = (torch.sigmoid(out) > 0.5).float()

            for ch, store in zip(
                [0, 1, 2],
                [wt_scores, tc_scores, et_scores]
            ):
                store.append(
                    dice_score(pred[:, ch], seg[:, ch]).item()
                )

    wt   = sum(wt_scores) / len(wt_scores)
    tc   = sum(tc_scores) / len(tc_scores)
    et   = sum(et_scores) / len(et_scores)
    mean = (wt + tc + et) / 3

    return {"WT": wt, "TC": tc, "ET": et, "mean": mean}
