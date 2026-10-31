"""
Train recent/strong baselines on EXACTLY the same split (700/150/150, seed 42), resolution (384x384),
augmentation (strong) and selection rule (best validation IoU) as our backbone -- a matched comparison
(Reviewer 1 c6) and the source of the qualitative comparison figure (Reviewer 1 c7).

Requires:  pip install segmentation-models-pytorch timm
ImageNet-pretrained encoder weights are downloaded automatically on first use.

Usage:
  python revision\\train_baselines.py --model unet_r34            --out ckpt_rev\\base_unet_r34
  python revision\\train_baselines.py --model unetpp_r34          --out ckpt_rev\\base_unetpp_r34
  python revision\\train_baselines.py --model deeplabv3p_r50      --out ckpt_rev\\base_deeplabv3p_r50
  python revision\\train_baselines.py --model segformer_mitb2     --out ckpt_rev\\base_segformer_mitb2 --batch 4
  python revision\\train_baselines.py --model fpn_pvtv2b2         --out ckpt_rev\\base_fpn_pvtv2b2   --batch 4   (optional)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones import Normalized, build_smp
from common import ROOT, TRANSFORMS, eval_transform, set_seed
from data.kvasir_loader import KvasirSegDataset
from models.backbone import compute_iou, dice_loss

MODELS = {
    "unet_r34":        ("unet", "resnet34"),
    "unetpp_r34":      ("unetpp", "resnet34"),
    "deeplabv3p_r50":  ("deeplabv3p", "resnet50"),
    "segformer_mitb2": ("segformer", "mit_b2"),
    "fpn_pvtv2b2":     ("fpn", "tu-pvt_v2_b2"),   # PVTv2-B2 encoder = the encoder family of Polyp-PVT / PVT-CASCADE / EMCAD
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="offline test only")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data"]["root"]); size = cfg["data"]["image_size"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arch, enc = MODELS[args.model]

    g = torch.Generator(); g.manual_seed(args.seed)
    tr = KvasirSegDataset(str(root), str(root / "splits/train.txt"), TRANSFORMS["strong"], size)
    va = KvasirSegDataset(str(root), str(root / "splits/val.txt"), eval_transform, size)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, generator=g, drop_last=True)
    vl = DataLoader(va, batch_size=4, shuffle=False, num_workers=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Normalized(build_smp(arch, enc, pretrained=not args.no_pretrained)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    best, best_ep, t0, log = -1, -1, time.time(), open(out / "log.jsonl", "w")
    for ep in range(args.epochs):
        net.train()
        for b in tqdm(tl, desc=f"{args.model} ep{ep}", leave=False):
            x = b["image"].permute(0, 3, 1, 2).float().to(dev); y = b["mask"].unsqueeze(1).float().to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                lo = net(x)
                if lo.shape[-2:] != y.shape[-2:]:
                    lo = Fn.interpolate(lo, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = Fn.binary_cross_entropy_with_logits(lo, y) + dice_loss(lo, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        net.eval(); ious = []
        with torch.no_grad():
            for b in vl:
                x = b["image"].permute(0, 3, 1, 2).float().to(dev); y = b["mask"].unsqueeze(1).float().to(dev)
                lo = net(x)
                if lo.shape[-2:] != y.shape[-2:]:
                    lo = Fn.interpolate(lo, size=y.shape[-2:], mode="bilinear", align_corners=False)
                ious += compute_iou(lo, y).cpu().tolist()
        v = float(np.mean(ious)); log.write(json.dumps({"epoch": ep, "val_iou": v}) + "\n"); log.flush()
        print(f"{args.model} ep{ep:02d} val_IoU={v:.4f}")
        if v > best:
            best, best_ep = v, ep
            torch.save({"model": net.state_dict(), "smp": {"arch": arch, "encoder": enc}, "epoch": ep, "val_iou": v},
                       out / "best.pt")
    json.dump({"model": args.model, "best_val_iou": best, "best_epoch": best_ep,
               "train_minutes": (time.time() - t0) / 60}, open(out / "summary.json", "w"), indent=2)
    print(f"DONE {args.model}: best val IoU {best:.4f} @ ep {best_ep}")


if __name__ == "__main__":
    main()
