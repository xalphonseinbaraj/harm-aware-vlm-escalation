"""Image-grouped cross-validation of the learned acceptance rule on the Kvasir-SEG validation split (revision R1, Section 4.9).
The 450 training rows come from 150 validation images x 3 seeds. Five-fold GroupKFold keeps all three rows of an image in the
same fold, so out-of-fold predictions never use another prediction of the same image. Reports out-of-fold AUROC for
'MedSAM mask better than backbone' and the out-of-fold IoU gain of gated escalation with learned acceptance, next to the
in-sample values, repeated over 20 random fold assignments. Run from the project root:  python revision\\acceptance_grouped_cv.py"""
import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def L(f):
    return {r["image_id"]: r for r in map(json.loads, open(f)) if "vlm_error" not in r}


def X(r):
    c, a, e, b = r["feats"]
    F = r.get("F"); F = 0.0 if (F is None or F != F) else F
    return [F, np.log1p(r.get("pixels_changed", 0) / max(a * 384 * 384, 1.0)), c, a, e, b, r["gate_prob"]]


out = {}
for bk in ["strong", "segformer"]:
    rows = []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            rows.append((r["image_id"], X(r), r["prop"]["iou"], r["bb"]["iou"], r["gate_prob"] >= 0.5))
    ids = np.array([r[0] for r in rows]); Z = np.array([r[1] for r in rows]); pr = np.array([r[2] for r in rows])
    bb = np.array([r[3] for r in rows]); esc = np.array([r[4] for r in rows]); y = pr > bb + 1e-9
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p_in = mk().fit(Z, y).predict_proba(Z)[:, 1]
    auc_in = roc_auc_score(y, p_in); gain_in = (np.where(esc & (p_in >= 0.5), pr, bb) - bb).mean()
    aucs, gains = [], []
    uniq = np.unique(ids)
    for rep in range(20):
        rng = np.random.default_rng(rep); perm = {u: k for k, u in enumerate(rng.permutation(uniq))}
        groups = np.array([perm[i] for i in ids]); p = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Z, y, groups):
            p[te] = mk().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        aucs.append(roc_auc_score(y, p)); gains.append((np.where(esc & (p >= 0.5), pr, bb) - bb).mean())
    out[bk] = {"rows": len(y), "images": len(uniq), "auroc_in_sample": float(auc_in), "auroc_grouped_cv": [float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs))],
               "gain_in_sample": float(gain_in), "gain_grouped_cv": [float(np.mean(gains)), float(np.min(gains)), float(np.max(gains))]}
    print(bk, json.dumps(out[bk]))
json.dump(out, open("results_val/acceptance_grouped_cv.json", "w"), indent=1)
