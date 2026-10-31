"""
Qualitative comparison figure (Reviewer 1, comment 7): rows = test images, columns =
Image | Ground truth | each baseline | Ours (backbone) | Ours (after gated escalation).
Prediction = filled overlay, ground-truth boundary = green line; per-panel IoU printed in the corner.

Prerequisites: run eval_all.py with --save-masks for every model, e.g.
  python revision\\eval_all.py --backbone ckpt_rev\\base_unet_r34\\best.pt --out results_rev\\base_unet_r34.jsonl --save-masks masks_rev\\base_unet_r34
  python revision\\eval_all.py --backbone ckpt_rev\\strong_s0\\best.pt --gate ckpt_rev\\strong_s0\\gate.joblib --vlm qwen7b_p3 ^
         --out results_rev\\strong_s0__qwen7b_p3.jsonl --save-masks masks_rev\\strong_s0__qwen7b_p3

Usage:
  python revision\\make_qual_figure.py --ours results_rev\\strong_s0__qwen7b_p3.jsonl --ours-masks masks_rev\\strong_s0__qwen7b_p3 ^
     --baselines unet_r34 unetpp_r34 deeplabv3p_r50 segformer_mitb2 fpn_pvtv2b2 --out results_rev\\analysis\\figures\\fig_qualitative.png
  (add --ids id1 id2 ... to choose images manually)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from common import ROOT, eval_transform, seg_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"unet_r34": "U-Net (R34)", "unetpp_r34": "UNet++ (R34)", "deeplabv3p_r50": "DeepLabV3+ (R50)",
         "segformer_mitb2": "SegFormer (MiT-B2)", "fpn_pvtv2b2": "FPN (PVTv2-B2)"}


def unpack(path, key, shape):
    return np.unpackbits(np.load(path)[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def overlay(img, pred, gt, color=(0.18, 0.53, 0.87)):
    out = img.copy()
    out[pred] = 0.55 * out[pred] + 0.45 * np.array(color)
    edge = gt & ~ndimage.binary_erosion(gt, iterations=2)
    out[edge] = (0.1, 0.9, 0.2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--ours", required=True); ap.add_argument("--ours-masks", required=True)
    ap.add_argument("--baselines", nargs="*", default=list(NAMES))
    ap.add_argument("--masks-root", default="masks_rev")
    ap.add_argument("--theta", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); root = Path(cfg["data"]["root"]); size = cfg["data"]["image_size"]
    R = {(r := json.loads(l))["image_id"]: r for l in open(a.ours) if "vlm_error" not in l}

    if a.ids:
        ids = a.ids
    else:  # automatic, transparent selection: easy / medium / hard (by OUR backbone IoU) + escalated & accepted edit
        srt = sorted(R.values(), key=lambda r: r["bb"]["iou"])
        n = len(srt); ids = [srt[int(n * q)]["image_id"] for q in (0.9, 0.5, 0.15)]
        esc = [r for r in srt if r["gate_prob"] >= a.theta and r.get("pixels_changed", 0) > 0
               and (r["F"] is None or r["F"] >= a.tau) and r["image_id"] not in ids]
        if esc:
            ids.append(max(esc, key=lambda r: r["prop"]["iou"] - r["bb"]["iou"])["image_id"])
    cols = ["Image", "Ground truth"] + [NAMES.get(b, b) for b in a.baselines] + ["Ours: backbone", "Ours: gated escalation"]
    fig, axs = plt.subplots(len(ids), len(cols), figsize=(1.9 * len(cols), 1.95 * len(ids)))
    axs = np.atleast_2d(axs)
    for i, iid in enumerate(ids):
        img = np.asarray(Image.open(root / "images" / f"{iid}.jpg").convert("RGB"))
        gt = (np.asarray(Image.open(root / "masks" / f"{iid}.jpg").convert("L")) > 127).astype(np.float32)
        img, gt = eval_transform(img, gt, size); gt = gt > 0.5; sh = gt.shape
        panels = [(img, None), (overlay(img, gt, gt, (0.1, 0.8, 0.2)), None)]
        for b in a.baselines:
            p = unpack(Path(a.masks_root) / f"base_{b}" / f"{iid}.npz", "bb", sh)
            panels.append((overlay(img, p, gt), seg_metrics(p, gt)["iou"]))
        ob = unpack(Path(a.ours_masks) / f"{iid}.npz", "bb", sh)
        panels.append((overlay(img, ob, gt, (1.0, 0.85, 0.0)), seg_metrics(ob, gt)["iou"]))
        r = R[iid]; esc = r["gate_prob"] >= a.theta; acc = r["F"] is None or r["F"] >= a.tau
        fin = unpack(Path(a.ours_masks) / f"{iid}.npz", "prop", sh) if (esc and acc) else ob
        tag = ("escalated: " + r["adj"] + (" ✓" if acc else " ✗ rejected")) if esc else "not escalated"
        panels.append((overlay(img, fin, gt, (1.0, 0.85, 0.0)), seg_metrics(fin, gt)["iou"]))
        for j, (pimg, iou) in enumerate(panels):
            ax = axs[i, j]; ax.imshow(np.clip(pimg, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if iou is not None:
                ax.text(4, 20, f"IoU {iou:.2f}", color="white", fontsize=7, bbox=dict(fc="black", alpha=.6, pad=1, lw=0))
            if i == 0:
                ax.set_title(cols[j], fontsize=7.5)
        axs[i, -1].text(4, size - 10, tag, color="white", fontsize=6, bbox=dict(fc="black", alpha=.6, pad=1, lw=0))
    plt.tight_layout(pad=0.3); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=300); print("saved", a.out, "ids:", ids)


if __name__ == "__main__":
    main()
