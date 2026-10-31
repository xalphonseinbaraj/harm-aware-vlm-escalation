"""
Backbone training for the revision (Reviewer 1 c5: augmentation; Reviewer 3 Method 2: lambda sensitivity;
Reviewer 3 Abstract 2 / Results: more seeds).

Same PolypUNet + BackboneLoss as the submitted paper; adds:
  --aug {light,strong}   light = original (hflip + brightness), strong = common.strong_train_transform
  --lambda-conf          confidence-loss weight lambda in Eq. (1)  (paper default 0.5)
  --seed                 training seed (data split stays FIXED, seed 42, 700/150/150)
  best.pt                checkpoint with the best validation IoU is saved automatically

Usage (from E:\\polyp_llm_seg):
  python revision\\train_backbone_v2.py --seed 0 --aug strong --out ckpt_rev\\strong_s0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ROOT, TRANSFORMS, eval_transform, set_seed
from data.kvasir_loader import KvasirSegDataset
from models.backbone import BackboneLoss, PolypUNet, compute_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", choices=["light", "strong"], default="strong")
    ap.add_argument("--lambda-conf", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg["data"]["root"]); splits = root / "splits"
    size = cfg["data"]["image_size"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)

    g = torch.Generator(); g.manual_seed(args.seed)
    tr = KvasirSegDataset(str(root), str(splits / "train.txt"), TRANSFORMS[args.aug], size)
    va = KvasirSegDataset(str(root), str(splits / "val.txt"), eval_transform, size)
    tl = DataLoader(tr, batch_size=cfg["backbone"]["batch_size"], shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(va, batch_size=8, shuffle=False, num_workers=0)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  train={len(tr)} val={len(va)}  aug={args.aug} lambda={args.lambda_conf} seed={args.seed}")
    model = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(dev)
    crit = BackboneLoss(confidence_weight=args.lambda_conf)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, best_ep, t0 = -1.0, -1, time.time()
    log = open(out / "log.jsonl", "w")
    for ep in range(args.epochs):
        model.train(); tot, nb = 0.0, 0
        for b in tqdm(tl, desc=f"ep{ep}", leave=False):
            x = b["image"].permute(0, 3, 1, 2).float().to(dev); y = b["mask"].unsqueeze(1).float().to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                ml, c = model(x); L = crit(ml, c, y)
            if not torch.isfinite(L["total"]):
                continue
            scaler.scale(L["total"]).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); tot += L["total"].item(); nb += 1
        model.eval(); ious, confs = [], []
        with torch.no_grad():
            for b in vl:
                x = b["image"].permute(0, 3, 1, 2).float().to(dev); y = b["mask"].unsqueeze(1).float().to(dev)
                ml, c = model(x); ious += compute_iou(ml, y).cpu().tolist(); confs += c.float().cpu().tolist()
        viou = float(np.mean(ious)); r = float(np.corrcoef(confs, ious)[0, 1])
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_iou": viou, "val_conf_iou_pearson": r,
               "minutes": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"ep{ep:02d} loss={rec['train_loss']:.4f} val_IoU={viou:.4f} r(conf,IoU)={r:.3f}")
        if viou > best:
            best, best_ep = viou, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_iou": viou, "args": vars(args)}, out / "best.pt")
    json.dump({"best_val_iou": best, "best_epoch": best_ep, "train_minutes": (time.time() - t0) / 60},
              open(out / "summary.json", "w"), indent=2)
    print(f"DONE best val IoU {best:.4f} at epoch {best_ep}")


if __name__ == "__main__":
    main()
