"""Prepare PolypGen single frames for evaluation (revision R1, Section 4.1). Run from the repository root after downloading
PolypGen from its authors (Synapse syn45200214) and unzipping it:
    python revision/prepare_polypgen.py <path to PolypGen2021_MultiCenterData_v3 folder that contains data_C1 ... data_C6>
Writes data/PolypGen/C1..C6/{images,masks} (masks binarized as PNG), skipping frames whose mask is empty or missing, and
data/PolypGen/prepare_report.json. Then checks every image against Kvasir-SEG and the PraNet test release with a 16x16 average
hash and writes data/PolypGen/dedup_report.json (closest match per image); pairs within 10 of 256 bits should be inspected."""
import glob
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

P = sys.argv[1]; O = "data/PolypGen"; rep = {}
for c in range(1, 7):
    os.makedirs(f"{O}/C{c}/images", exist_ok=True); os.makedirs(f"{O}/C{c}/masks", exist_ok=True)
    kept = empty = nomask = 0
    for f in sorted(glob.glob(f"{P}/data_C{c}/images_C{c}/*.jpg")):
        st = os.path.splitext(os.path.basename(f))[0]; m = f"{P}/data_C{c}/masks_C{c}/{st}_mask.jpg"
        if not os.path.exists(m):
            nomask += 1; continue
        mk = np.asarray(Image.open(m).convert("L")) > 127
        if mk.sum() == 0:
            empty += 1; continue
        shutil.copyfile(f, f"{O}/C{c}/images/{st}.jpg")
        Image.fromarray((mk * 255).astype(np.uint8)).save(f"{O}/C{c}/masks/{st}.png"); kept += 1
    rep[f"C{c}"] = {"kept": kept, "empty_mask": empty, "no_mask": nomask}; print(f"C{c}", rep[f"C{c}"])
json.dump(rep, open(f"{O}/prepare_report.json", "w"), indent=1)


def h(p):
    a = np.asarray(Image.open(p).convert("L").resize((16, 16), Image.BILINEAR), dtype=float); return (a > a.mean()).flatten()


refs = {}
for name, pat in [("Kvasir-SEG", "data/Kvasir-SEG/images/*"), ("CVC-ClinicDB", "data/TestDataset/CVC-ClinicDB/images/*"),
                  ("CVC-ColonDB", "data/TestDataset/CVC-ColonDB/images/*"), ("ETIS", "data/TestDataset/ETIS-LaribPolypDB/images/*")]:
    fs = sorted(glob.glob(pat))
    if fs:
        refs[name] = (fs, np.array([h(f) for f in fs]))
out = {}
for c in range(1, 7):
    rows = []
    for f in sorted(glob.glob(f"{O}/C{c}/images/*.jpg")):
        v = h(f); best = (999, None, None)
        for name, (rf, H) in refs.items():
            d = (H != v).sum(1); i = int(d.argmin())
            if d[i] < best[0]:
                best = (int(d[i]), name, os.path.basename(rf[i]))
        rows.append((os.path.basename(f),) + best)
    out[f"C{c}"] = rows; ds = [r[1] for r in rows]
    print(f"C{c}: closest distance {min(ds)}; pairs within 10 bits: {sum(x <= 10 for x in ds)}")
json.dump(out, open(f"{O}/dedup_report.json", "w"))
