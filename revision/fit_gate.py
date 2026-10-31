"""
Fit the outcome-supervised confidence gate (Section 3.3) on the VALIDATION split for a given backbone,
and report what Reviewer 3 asked for on the validation split: escalation rate, precision / recall / F1 / AUROC
against the true difficulty label (IoU < 0.7), and the threshold sweep.

Usage:
  python revision\\fit_gate.py --backbone ckpt_rev\\strong_s0\\best.pt --out ckpt_rev\\strong_s0\\gate.joblib
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from backbones import load_backbone
from common import ROOT, eval_transform
from data.kvasir_loader import KvasirSegDataset
from models.backbone import compute_iou
from models.confidence_gate import ConfidenceGate, build_refinement_labels, extract_gate_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.7)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = load_backbone(args.backbone, dev, cfg["backbone"]["base_channels"])
    root = cfg["data"]["root"]
    ds = KvasirSegDataset(root, f"{root}/splits/val.txt", eval_transform, cfg["data"]["image_size"])
    F, I = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            x = torch.from_numpy(b["image"]).permute(2, 0, 1)[None].float().to(dev)
            y = torch.from_numpy(b["mask"])[None, None].float().to(dev)
            ml, c = bb(x)
            F.append(extract_gate_features(ml.float(), c.float())); I.append(compute_iou(ml, y).cpu().numpy())
    F, I = np.concatenate(F), np.concatenate(I)
    y = build_refinement_labels(I, args.iou_threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gate = ConfidenceGate(); gate.fit(F, y); gate.save(args.out)
    p = gate.predict_proba(F)
    rep = {"n_val": int(len(y)), "hard_rate": float(y.mean()), "auroc_gate": float(roc_auc_score(y, p)),
           "auroc_conf_only": float(roc_auc_score(y, -F[:, 0])), "coef": gate.clf.coef_.tolist(), "sweep": []}
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        f = p >= t; tp = (f & (y == 1)).sum()
        prec = tp / max(f.sum(), 1); rec = tp / max(y.sum(), 1)
        rep["sweep"].append({"theta": t, "rate": float(f.mean()), "precision": float(prec), "recall": float(rec)})
    json.dump(rep, open(Path(args.out).with_suffix(".val_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
