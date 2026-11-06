"""Agreement and calibration of the two acceptance-rule fitting populations (revision R1, Section 4.9).
Rule A: fitted on all validation rows (proposed). Rule B: fitted on gated validation rows only (inference population).
On every evaluation set, among gated images: accept/reject agreement, Cohen's kappa, Brier score and expected calibration error
(10 equal-width bins) for predicting 'MedSAM mask better than backbone'. Run from the project root:
    python revision\\acceptance_agreement.py
Writes results_ext/acceptance_agreement.json."""
import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def L(f):
    return {r["image_id"]: r for r in map(json.loads, open(f)) if "vlm_error" not in r}


def X(r):
    c, a, e, b = r["feats"]
    F = r.get("F"); F = 0.0 if (F is None or F != F) else F
    return [F, np.log1p(r.get("pixels_changed", 0) / max(a * 384 * 384, 1.0)), c, a, e, b, r["gate_prob"]]


def fit(bk, gated):
    Xs, y = [], []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            if gated and r["gate_prob"] < 0.5:
                continue
            Xs.append(X(r)); y.append(r["prop"]["iou"] > r["bb"]["iou"] + 1e-9)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.array(Xs), np.array(y))


def ece(p, y, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def files(bk, ds):
    if ds == "Kvasir-SEG test":
        return [f"results_rev/{bk}_s{s}__medsam_box.jsonl" for s in range(3)]
    if ds == "PolypGen":
        return [f"results_ext/PolypGen-C{c}_{bk}_s{s}__medsam_box.jsonl" for s in range(3) for c in range(1, 7)]
    return [f"results_ext/{ds}_{bk}_s{s}__medsam_box.jsonl" for s in range(3)]


out = {}
for bk in ["strong", "segformer"]:
    A, B = fit(bk, False), fit(bk, True)
    for ds in ["Kvasir-SEG test", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]:
        rows = [r for f in files(bk, ds) for r in L(f).values() if r["gate_prob"] >= 0.5]
        Z = np.array([X(r) for r in rows]); y = np.array([r["prop"]["iou"] > r["bb"]["iou"] + 1e-9 for r in rows])
        pa, pb = A.predict_proba(Z)[:, 1], B.predict_proba(Z)[:, 1]
        da, db = pa >= 0.5, pb >= 0.5
        res = {"gated_rows": len(rows), "agreement": float((da == db).mean()), "kappa": float(cohen_kappa_score(da, db)),
               "accept_rate_all": float(da.mean()), "accept_rate_gated": float(db.mean()),
               "brier_all": float(brier_score_loss(y, pa)), "brier_gated": float(brier_score_loss(y, pb)),
               "ece_all": float(ece(pa, y)), "ece_gated": float(ece(pb, y)), "helpful_rate": float(y.mean())}
        out[f"{bk}|{ds}"] = res
        print(f"{bk:9} {ds:18} n={len(rows):5} agree {100*res['agreement']:.1f}% kappa {res['kappa']:.2f}  accept {100*res['accept_rate_all']:.0f}% vs "
              f"{100*res['accept_rate_gated']:.0f}%  Brier {res['brier_all']:.3f} vs {res['brier_gated']:.3f}  ECE {res['ece_all']:.3f} vs {res['ece_gated']:.3f}  "
              f"helpful {100*res['helpful_rate']:.0f}%", flush=True)
json.dump(out, open("results_ext/acceptance_agreement.json", "w"), indent=1)
