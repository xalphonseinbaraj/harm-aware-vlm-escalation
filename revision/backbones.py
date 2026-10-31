"""
Uniform wrapper so the SAME gate / escalation pipeline can sit on top of
(a) the paper's PolypUNet (with its outcome-trained confidence head), or
(b) any baseline trained by train_baselines.py (segmentation_models_pytorch), which has
    no confidence head -- for those, feature (i) of the gate is replaced by the mean
    max-class probability over the predicted foreground (a standard training-free proxy).
This lets the revision show that the escalation policy is backbone-agnostic (Reviewer 1 c2/c3).
"""
import json
from pathlib import Path

import torch

from common import ROOT  # noqa: F401
from models.backbone import PolypUNet


class Backbone(torch.nn.Module):
    def __init__(self, net, has_conf: bool, name: str):
        super().__init__()
        self.net, self.has_conf, self.name = net, has_conf, name

    def forward(self, x):
        out = self.net(x)
        if self.has_conf:
            return out  # (logits, conf)
        logits = out
        if logits.shape[-2:] != x.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        p = torch.sigmoid(logits)
        fg = (p > 0.5).float()
        conf = (torch.maximum(p, 1 - p) * fg).sum(dim=(1, 2, 3)) / fg.sum(dim=(1, 2, 3)).clamp(min=1)
        return logits, conf


def build_smp(arch: str, encoder: str, pretrained: bool = True):
    import segmentation_models_pytorch as smp
    w = "imagenet" if pretrained else None
    A = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3p": smp.DeepLabV3Plus,
         "fpn": smp.FPN, "segformer": smp.Segformer, "manet": smp.MAnet}[arch]
    return A(encoder_name=encoder, encoder_weights=w, in_channels=3, classes=1)


def load_backbone(ckpt_path: str, device: str, base_ch: int = 32) -> Backbone:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("smp") if isinstance(ckpt, dict) else None
    if meta:
        net = Normalized(build_smp(meta["arch"], meta["encoder"], pretrained=False))
        net.load_state_dict(ckpt["model"])
        bb = Backbone(net, has_conf=False, name=f"{meta['arch']}-{meta['encoder']}")
    else:
        net = PolypUNet(base_ch=base_ch)
        net.load_state_dict(ckpt["model"])
        bb = Backbone(net, has_conf=True, name="PolypUNet")
    return bb.to(device).eval()


# ImageNet normalisation is applied INSIDE the smp wrapper used in training, see train_baselines.py
class Normalized(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.m) / self.s)

    def load_state_dict(self, sd, strict=True):  # accept checkpoints saved from the inner net
        return self.net.load_state_dict(sd, strict=strict)

    def state_dict(self, *a, **k):
        return self.net.state_dict(*a, **k)
