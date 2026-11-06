"""Learned acceptance vs. simpler acceptance rules, and fitting sensitivity (revision R1, Section 4.9, Supplementary Table S6).
All rules are fitted on the Kvasir-SEG VALIDATION split only and applied unchanged. Run from the project root:
    python revision\\acceptance_baselines.py
Rules (each accepts MedSAM's mask only for gated images, gate probability >= 0.5):
  full        logistic regression on the seven inputs of Section 4.9 (the proposed rule), three seeds pooled
  full_seed0  the same, fitted on the seed-0 validation run only (no pooling of repeated images)
  full_bal    the same, with balanced class weights
  change      accept if the log relative size of the change is below a threshold
  faith       accept if the faithfulness score F is above a threshold
  conf        accept if the backbone confidence is below a threshold
              (single-feature thresholds chosen on the validation split to maximise the mean IoU gain, grid of 99 quantiles)
  deletion    the original deletion test (F >= 0.5)
Writes results_ext/acceptance_baselines.json."""
import glob
import json

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
COLS = {"full": [0, 1, 2, 3, 4, 5, 6], "change": [1], "faith": [0], "conf": [2]}


def L(f):
    return {r["image_id"]: r for r in map(json.loads, open(f)) if "vlm_error" not in r}


def X(r):
    c, a, e, b = r["feats"]
    F = r.get("F"); F = 0.0 if (F is None or F != F) else F
    return [F, np.log1p(r.get("pixels_changed", 0) / max(a * 384 * 384, 1.0)), c, a, e, b, r["gate_prob"]]


def acc(r):
    return (r.get("F") is None) or (r["F"] != r["F"]) or r["F"] >= 0.5


def fit(files, cols, cw=None):
    Xs, y = [], []
    for f in files:
        for r in L(f).values():
            Xs.append(np.array(X(r))[cols]); y.append(r["prop"]["iou"] > r["bb"]["iou"] + 1e-9)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=cw)).fit(np.array(Xs), np.array(y))


class Thr:
    """Single-feature acceptance threshold chosen on validation (maximises mean IoU gain over gated images)."""
    def __init__(self, files, col, below):
        v, g = [], []
        for f in files:
            for r in L(f).values():
                if r["gate_prob"] >= 0.5:
                    v.append(X(r)[col]); g.append(r["prop"]["iou"] - r["bb"]["iou"])
        v, g = np.array(v), np.array(g); best = (0.0, np.inf if not below else -np.inf)
        for t in np.quantile(v, np.linspace(0.01, 0.99, 99)):
            m = v <= t if below else v >= t
            if g[m].sum() / len(v) > best[0]:
                best = (g[m].sum() / len(v), t)
        self.t, self.below, self.col = best[1], below, col

    def accept(self, rows):
        v = np.array([X(r)[self.col] for r in rows])
        return v <= self.t if self.below else v >= self.t


def datasets(bk):
    d = {"Kvasir-SEG test": [[f"results_rev/{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]}
    for n in ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB"]:
        d[n] = [[f"results_ext/{n}_{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]
    d["PolypGen (6 centres)"] = [[f"results_ext/PolypGen-C{c}_{bk}_s{s}__medsam_box.jsonl" for c in range(1, 7)] for s in range(3)]
    return d


out = {}
for bk in ["strong", "segformer"]:
    vf = sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl"))
    models = {"full": (fit(vf, COLS["full"]), COLS["full"]), "full_seed0": (fit(vf[:1], COLS["full"]), COLS["full"]),
              "full_bal": (fit(vf, COLS["full"], "balanced"), COLS["full"]),
              "change": (Thr(vf, 1, True), None), "faith": (Thr(vf, 0, False), None), "conf": (Thr(vf, 2, True), None)}
    for ds, seeds in datasets(bk).items():
        if not all(glob.glob(f) for fs in seeds for f in fs):
            continue
        runs = []
        for fs in seeds:
            R = {}
            for f in fs:
                R.update(L(f))
            runs.append(R)
        ids = sorted(set.intersection(*[set(R) for R in runs]))
        res = {}
        for name in list(models) + ["deletion"]:
            G, N = [], []
            for R in runs:
                bb = np.array([R[i]["bb"]["iou"] for i in ids]); pr = np.array([R[i]["prop"]["iou"] for i in ids])
                esc = np.array([R[i]["gate_prob"] >= 0.5 for i in ids])
                if name == "deletion":
                    a = esc & np.array([acc(R[i]) for i in ids])
                elif isinstance(models[name][0], Thr):
                    a = esc & models[name][0].accept([R[i] for i in ids])
                else:
                    m, cols = models[name]
                    a = esc & (m.predict_proba(np.array([np.array(X(R[i]))[cols] for i in ids]))[:, 1] >= 0.5)
                G.append(np.where(a, pr, bb)); N.append(bb)
            G, N = np.stack(G), np.stack(N); d = G - N; n = len(ids)
            ci = np.percentile([d[:, ix].mean() for ix in (rng.integers(0, n, n) for _ in range(2000))], [2.5, 97.5])
            try:
                p = wilcoxon(G.mean(0), N.mean(0)).pvalue
            except ValueError:
                p = float("nan")
            res[name] = {"gain": float(d.mean()), "ci": [float(x) for x in ci], "p": float(p)}
        out[f"{bk}|{ds}"] = res
        print(f"{bk:9} {ds:22} " + "  ".join(f"{k} {v['gain']:+.4f}[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for k, v in res.items()), flush=True)
json.dump(out, open("results_ext/acceptance_baselines.json", "w"), indent=1)
