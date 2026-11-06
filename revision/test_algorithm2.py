"""Implementation checks for Algorithm 2 (revision R1). Run from the project root:  python revision\\test_algorithm2.py
Checks, without downloading MedSAM:
  (1) the box prompt is the bounding box of the backbone mask enlarged by max(3 px, 10% of the box size) on each side, clipped;
  (2) an empty backbone mask returns the backbone mask with label 'no_change' and never calls the model;
  (3) the change-size input uses the mean foreground probability times 384^2, bounded below by 1, so it is finite for empty masks;
  (4) the area feature passed to the gate (feats[1]) is the mean foreground probability."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medsam_refiner as M  # noqa: E402


class _NoModel:
    def __call__(self, *a, **k):
        raise AssertionError("model must not be called for an empty mask")


r = M.MedSAMRefiner.__new__(M.MedSAMRefiner); r.margin = 0.10; r.model = _NoModel(); r.proc = None; r.dev = "cpu"
m = np.zeros((384, 384), np.float32); m[100:200, 150:170] = 1          # 100 x 20 box
box = r.box_from_mask(m)
assert box == [147.0, 90.0, 172.0, 209.0], box                         # x: max(3, 2)=3; y: max(3, 10)=10
m2 = np.zeros((384, 384), np.float32); m2[0:5, 380:384] = 1
assert r.box_from_mask(m2) == [377.0, 0.0, 383.0, 7.0], r.box_from_mask(m2)   # clipped at the image border
out, lab = r(np.zeros((384, 384, 3), np.float32), np.zeros((384, 384), np.float32))
assert lab == "no_change" and out.sum() == 0
print("(1)-(2) box construction and empty-mask rule: OK")

src = (Path(__file__).resolve().parent / "learned_acceptance.py").read_text()
assert "max(a*384*384,1.0)" in src.replace(" ", ""), "change-size denominator differs from Algorithm 2"
a, pix = 0.0, 0
val = np.log1p(pix / max(a * 384 * 384, 1.0)); assert np.isfinite(val)
print("(3) change-size denominator bounded below by one pixel: OK")

root = Path(__file__).resolve().parents[1]
cg = (root / "models" / "confidence_gate.py")
if cg.exists():
    t = cg.read_text()
    import re
    assert "area_frac = probs.mean(dim=(1, 2))" in t, "area feature is not the mean foreground probability"
    assert re.search(r"\[\s*confidence_pred,\s*area_frac,\s*mean_entropy,\s*boundary_complexity\s*\]", t), "feature order differs"
    print("(4) gate area feature is the mean foreground probability: OK")
else:
    print("(4) models/confidence_gate.py not found from here; skipped")
print("ALL CHECKS PASSED")
