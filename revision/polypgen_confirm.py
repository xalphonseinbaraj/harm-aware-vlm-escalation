"""Retrospective external evaluation on PolypGen (revision R1; internal analysis plan in revision/PHASE10_PREREGISTRATION.txt).
The learned acceptance rule is fitted on the Kvasir-SEG VALIDATION split only (results_val/*__medsam_box.jsonl, seeds pooled),
exactly as in revision/learned_acceptance.py, and applied unchanged to PolypGen centres C1-C6.
Primary outcome: strong PolypUNet, three seeds, all centres pooled. Run from the project root:  python revision\\polypgen_confirm.py
Writes results_ext/polypgen_confirm.json."""
import glob
import json

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CENTRES = ["C1", "C2", "C3", "C4", "C5", "C6"]
rng = np.random.default_rng(0)


def L(f):
    return {r["image_id"]: r for r in map(json.loads, open(f)) if "vlm_error" not in r}


def X(r):
    c, a, e, b = r["feats"]
    F = r.get("F"); F = 0.0 if (F is None or F != F) else F
    rel = r.get("pixels_changed", 0) / max(a * 384 * 384, 1.0)
    return [F, np.log1p(rel), c, a, e, b, r["gate_prob"]]


def acc(r):
    return (r.get("F") is None) or (r["F"] != r["F"]) or r["F"] >= 0.5


def fit(bk):
    Xs, y = [], []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            Xs.append(X(r)); y.append(r["prop"]["iou"] > r["bb"]["iou"] + 1e-9)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.array(Xs), np.array(y))


def seeds(bk):
    return ["strong_s0", "strong_s1", "strong_s2"] if bk == "strong" else ["segformer_s0", "segformer_s1", "segformer_s2"]


def evaluate(bk, centres, m):
    per = []
    for sd in seeds(bk):
        R = {}
        for c in centres:
            f = f"results_ext/PolypGen-{c}_{sd}__medsam_box.jsonl"
            if not glob.glob(f):
                return None
            R.update(L(f))
        per.append(R)
    ids = sorted(set.intersection(*[set(R) for R in per]))
    S = {k: [] for k in ["never", "always", "gateV", "learned", "gate_perfect"]}; accepted = []
    for R in per:
        bb = np.array([R[i]["bb"]["iou"] for i in ids]); pr = np.array([R[i]["prop"]["iou"] for i in ids])
        gp = np.array([R[i]["gate_prob"] for i in ids]); ac = np.array([acc(R[i]) for i in ids]); esc = gp >= 0.5
        p = m.predict_proba(np.array([X(R[i]) for i in ids]))[:, 1]; a = esc & (p >= 0.5)
        S["never"].append(bb); S["always"].append(pr); S["gateV"].append(np.where(esc & ac, pr, bb))
        S["learned"].append(np.where(a, pr, bb)); S["gate_perfect"].append(np.where(esc, np.maximum(pr, bb), bb)); accepted.append(a.mean())
    S = {k: np.stack(v) for k, v in S.items()}; n = len(ids)
    gain = S["learned"] - S["never"]
    boot = [gain[:, ix].mean() for ix in (rng.integers(0, n, n) for _ in range(2000))]
    try:
        p = wilcoxon(S["learned"].mean(0), S["never"].mean(0)).pvalue
    except ValueError:
        p = float("nan")
    out = {k: float(v.mean()) for k, v in S.items()}
    out.update(n=n, gain=float(gain.mean()), ci=[float(x) for x in np.percentile(boot, [2.5, 97.5])], p=float(p),
               accepted=float(np.mean(accepted)), ceiling=float(S["gate_perfect"].mean() - S["never"].mean()))
    return out


res = {}
for bk in ["strong", "segformer"]:
    m = fit(bk)
    for name, cs in [("pooled", CENTRES)] + [(c, [c]) for c in CENTRES]:
        r = evaluate(bk, cs, m)
        if r is None:
            continue
        res[f"{bk}|{name}"] = r
        tag = "PRIMARY " if (bk == "strong" and name == "pooled") else ""
        print(f"{tag}{bk:9} {name:7} n={r['n']:4}  never {r['never']:.3f}  always {r['always']:.3f}  gate+V {r['gateV']:.3f}  "
              f"gate+learned {r['learned']:.3f}  gain {r['gain']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  p={r['p']:.2g}  "
              f"accepted {100 * r['accepted']:.0f}%", flush=True)
if "strong|pooled" in res:
    r = res["strong|pooled"]
    print("\nPrimary-analysis success criterion (gain > 0 and 95% CI excluding zero):", "MET" if (r["gain"] > 0 and r["ci"][0] > 0) else "NOT MET")
json.dump(res, open("results_ext/polypgen_confirm.json", "w"), indent=1)
