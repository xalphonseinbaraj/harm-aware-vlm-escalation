"""Training population of the acceptance rule (revision R1, Section 4.9, Algorithm 2). Run from the project root:
    python revision\\acceptance_population.py
The proposed rule is fitted on all validation images (MedSAM is run on every image, so a candidate exists for each), with the
gate probability as one of its inputs, and is applied at inference only to gated images. This script compares it with a rule
fitted only on the gated validation images (gate probability >= 0.5), i.e. on exactly the population seen at inference.
Writes results_ext/acceptance_population.json."""
import glob
import json

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)


def L(f):
    return {r["image_id"]: r for r in map(json.loads, open(f)) if "vlm_error" not in r}


def X(r):
    c, a, e, b = r["feats"]
    F = r.get("F"); F = 0.0 if (F is None or F != F) else F
    return [F, np.log1p(r.get("pixels_changed", 0) / max(a * 384 * 384, 1.0)), c, a, e, b, r["gate_prob"]]


def fit(bk, gated_only):
    Xs, y = [], []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            if gated_only and r["gate_prob"] < 0.5:
                continue
            Xs.append(X(r)); y.append(r["prop"]["iou"] > r["bb"]["iou"] + 1e-9)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.array(Xs), np.array(y)), len(y)


def sets(bk, ds):
    if ds == "Kvasir-SEG test":
        return [[f"results_rev/{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]
    if ds == "PolypGen":
        return [[f"results_ext/PolypGen-C{c}_{bk}_s{s}__medsam_box.jsonl" for c in range(1, 7)] for s in range(3)]
    return [[f"results_ext/{ds}_{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]


out = {}
for bk in ["strong", "segformer"]:
    m_all, n_all = fit(bk, False); m_g, n_g = fit(bk, True)
    out[f"{bk}|rows"] = {"all": n_all, "gated": n_g}
    print(f"{bk}: training rows all={n_all}, gated only={n_g}")
    for ds in ["Kvasir-SEG test", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]:
        runs = []
        for fs in sets(bk, ds):
            R = {}
            for f in fs:
                R.update(L(f))
            runs.append(R)
        ids = sorted(set.intersection(*[set(R) for R in runs]))
        res = {}
        for name, m in [("all", m_all), ("gated", m_g)]:
            G, N = [], []
            for R in runs:
                bb = np.array([R[i]["bb"]["iou"] for i in ids]); pr = np.array([R[i]["prop"]["iou"] for i in ids])
                esc = np.array([R[i]["gate_prob"] >= 0.5 for i in ids])
                a = esc & (m.predict_proba(np.array([X(R[i]) for i in ids]))[:, 1] >= 0.5)
                G.append(np.where(a, pr, bb)); N.append(bb)
            G, N = np.stack(G), np.stack(N); d = G - N; n = len(ids)
            ci = np.percentile([d[:, ix].mean() for ix in (rng.integers(0, n, n) for _ in range(2000))], [2.5, 97.5])
            res[name] = {"gain": float(d.mean()), "ci": [float(x) for x in ci]}
        out[f"{bk}|{ds}"] = res
        print(f"  {ds:20} all-images rule {res['all']['gain']:+.4f} [{res['all']['ci'][0]:+.4f}, {res['all']['ci'][1]:+.4f}]   "
              f"gated-only rule {res['gated']['gain']:+.4f} [{res['gated']['ci'][0]:+.4f}, {res['gated']['ci'][1]:+.4f}]", flush=True)
json.dump(out, open("results_ext/acceptance_population.json", "w"), indent=1)
