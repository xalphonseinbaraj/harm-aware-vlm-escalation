"""Operator ceiling (revision R1, Section 4.5): apply all four edit operations to every test image and keep the best one, i.e. what a
perfect VLM could reach with the edit operator of Section 3.4. Uses the masks saved with --save-masks. Run from the project root:
    python revision\\operator_ceiling.py
"""
import json
import numpy as np
from PIL import Image
from scipy import ndimage


def adj(mask, a):
    it = int(np.clip(round(0.03 * np.sqrt(max(float(mask.sum()), 1.0))), 1, 10))
    if a == "expand":
        return ndimage.binary_dilation(mask, iterations=it)
    if a == "shrink":
        return ndimage.binary_erosion(mask, iterations=it)
    if a == "shift_boundary":
        h = max(1, it // 2)
        return ndimage.binary_dilation(ndimage.binary_erosion(mask, iterations=h), iterations=h)
    return mask.copy()


def iou(p, g):
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


ids = [l.strip() for l in open("data/Kvasir-SEG/splits/test.txt") if l.strip()]
G = {i: np.array(Image.fromarray(((np.asarray(Image.open(f"data/Kvasir-SEG/masks/{i}.jpg").convert("L")) > 127) * 255)
                                 .astype(np.uint8)).resize((384, 384), Image.NEAREST)) > 127 for i in ids}
for s in range(3):
    R = {r["image_id"]: r for r in map(json.loads, open(f"results_rev/strong_s{s}__medgemma4b_p1.jsonl"))}
    nv, best, bg = [], [], []
    for i in ids:
        bb = np.unpackbits(np.load(f"masks_rev/strong_s{s}__medgemma4b_p1/{i}.npz")["bb"])[:384 * 384].reshape(384, 384).astype(bool)
        b = iou(bb, G[i]); e = max(iou(adj(bb, a), G[i]) for a in ["no_change", "expand", "shrink", "shift_boundary"])
        nv.append(b); best.append(e); bg.append(e if R[i]["gate_prob"] >= 0.5 else b)
    print(f"seed {s}: backbone {np.mean(nv):.4f}  best of four edits, every image {np.mean(best):.4f} (+{np.mean(best) - np.mean(nv):.4f})"
          f"  gated images only {np.mean(bg):.4f} (+{np.mean(bg) - np.mean(nv):.4f})")
