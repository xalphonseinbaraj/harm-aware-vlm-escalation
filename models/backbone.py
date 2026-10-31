"""
Lightweight U-Net segmentation backbone with an auxiliary confidence head.

Design choice: the confidence head is trained to predict the mask's actual
IoU against ground truth (outcome supervision) -- NOT derived from softmax
entropy or a training-free heuristic. This mirrors the same design principle
as CAGE's retrieval gate and the unified-segdet-fusion project's confidence
head: learn confidence from real outcomes, not a proxy for them.

Kept small deliberately (few channels, few down/up blocks) to fit comfortably
on a 6-8GB laptop GPU alongside room for the MLLM API calls and eval loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PolypUNet(nn.Module):
    """Small U-Net: 4 down / 4 up stages, base_ch=32 by default.
    Outputs:
        mask_logits: (B, 1, H, W) -- raw logits, apply sigmoid for probability
        confidence:  (B,)         -- predicted IoU in [0, 1], via a small head
                                     on the bottleneck features
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.mask_head = nn.Conv2d(c, 1, kernel_size=1)

        # confidence head: global-pool the bottleneck, predict scalar IoU
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # IoU is in [0, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        mask_logits = self.mask_head(d1)
        confidence = self.confidence_head(b).squeeze(-1)  # (B,)

        return mask_logits, confidence


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def compute_iou(logits, targets, threshold=0.5, eps=1e-6):
    """Per-sample IoU, used both as an eval metric and as the confidence
    head's training target (outcome supervision)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)  # (B,)


class BackboneLoss(nn.Module):
    """Combined loss: BCE + Dice for the mask, MSE for the confidence head
    against the *actual* per-sample IoU (stop-gradient on the IoU target --
    it's a supervision signal, not something to backprop through the mask
    branch a second time)."""

    def __init__(self, confidence_weight: float = 0.5):
        super().__init__()
        self.confidence_weight = confidence_weight

    def forward(self, mask_logits, confidence_pred, mask_target):
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
        dice = dice_loss(mask_logits, mask_target)
        seg_loss = bce + dice

        with torch.no_grad():
            iou_target = compute_iou(mask_logits, mask_target)
        conf_loss = F.mse_loss(confidence_pred, iou_target)

        total = seg_loss + self.confidence_weight * conf_loss
        return {
            "total": total,
            "seg_loss": seg_loss,
            "bce": bce,
            "dice": dice,
            "conf_loss": conf_loss,
        }
