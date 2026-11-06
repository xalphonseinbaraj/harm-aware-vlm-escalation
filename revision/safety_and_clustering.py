"""Revision R1, response to strict review of v15. Run from the project root:  python revision\\safety_and_clustering.py
(1) Paired difference between the full learned acceptance rule and the change-size-only rule (both fitted on validation only).
(2) Per-image gain/loss distribution, severe failures and boundary metrics for gated MedSAM escalation with learned acceptance.
(3) Cluster-aware uncertainty on PolypGen: two-stage bootstrap (centres, then images within centres), leave-one-centre-out,
    and for centre C4 a bootstrap over video-sequence prefixes.
Writes results_ext/safety_and_clustering.json."""
import glob
import json
import re

import numpy as np
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


def fit_full(bk):
    Xs, y = [], []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            Xs.append(X(r)); y.append(r["prop"]["iou"] > r["bb"]["iou"] + 1e-9)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.array(Xs), np.array(y))


def fit_change(bk):
    v, g = [], []
    for f in sorted(glob.glob(f"results_val/{bk}_s?__medsam_box.jsonl")):
        for r in L(f).values():
            if r["gate_prob"] >= 0.5:
                v.append(X(r)[1]); g.append(r["prop"]["iou"] - r["bb"]["iou"])
    v, g = np.array(v), np.array(g); best = (0.0, -np.inf)
    for t in np.quantile(v, np.linspace(0.01, 0.99, 99)):
        if g[v <= t].sum() / len(v) > best[0]:
            best = (g[v <= t].sum() / len(v), t)
    return best[1]


def load(bk, ds):
    if ds == "Kvasir-SEG test":
        sets = [[f"results_rev/{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]
    elif ds == "PolypGen":
        sets = [[f"results_ext/PolypGen-C{c}_{bk}_s{s}__medsam_box.jsonl" for c in range(1, 7)] for s in range(3)]
    else:
        sets = [[f"results_ext/{ds}_{bk}_s{s}__medsam_box.jsonl"] for s in range(3)]
    runs = []
    for fs in sets:
        R = {}
        for f in fs:
            R.update(L(f))
        runs.append(R)
    ids = sorted(set.intersection(*[set(R) for R in runs]))
    return runs, ids


def metric(R, ids, key, which):
    return np.array([R[i][which].get(key, np.nan) if R[i][which].get(key) is not None else np.nan for i in ids], dtype=float)


out = {}
for bk in ["strong", "segformer"]:
    m = fit_full(bk); tch = fit_change(bk)
    for ds in ["Kvasir-SEG test", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "PolypGen"]:
        runs, ids = load(bk, ds)
        full, chg, never, bf_n, bf_f, hd_n, hd_f = [], [], [], [], [], [], []
        for R in runs:
            bb = metric(R, ids, "iou", "bb"); pr = metric(R, ids, "iou", "prop")
            esc = np.array([R[i]["gate_prob"] >= 0.5 for i in ids])
            Z = np.array([X(R[i]) for i in ids])
            af = esc & (m.predict_proba(Z)[:, 1] >= 0.5); ac = esc & (Z[:, 1] <= tch)
            full.append(np.where(af, pr, bb)); chg.append(np.where(ac, pr, bb)); never.append(bb)
            bf_n.append(metric(R, ids, "bf1", "bb")); bf_f.append(np.where(af, metric(R, ids, "bf1", "prop"), metric(R, ids, "bf1", "bb")))
            hd_n.append(metric(R, ids, "hd95", "bb")); hd_f.append(np.where(af, metric(R, ids, "hd95", "prop"), metric(R, ids, "hd95", "bb")))
        full, chg, never = map(np.stack, (full, chg, never)); n = len(ids)
        dfc = full - chg
        ci_fc = np.percentile([dfc[:, ix].mean() for ix in (rng.integers(0, n, n) for _ in range(2000))], [2.5, 97.5])
        g = (full - never).mean(0)                     # per-image gain, averaged over seeds
        res = {"n": n, "full_minus_change": float(dfc.mean()), "full_minus_change_ci": [float(x) for x in ci_fc],
               "improved": float((g > 1e-6).mean()), "unchanged": float((np.abs(g) <= 1e-6).mean()), "worsened": float((g < -1e-6).mean()),
               "worse_gt_0.05": float((g < -0.05).mean()), "worse_gt_0.10": float((g < -0.10).mean()),
               "better_gt_0.05": float((g > 0.05).mean()), "better_gt_0.10": float((g > 0.10).mean()),
               "p05_gain": float(np.percentile(g, 5)), "p95_gain": float(np.percentile(g, 95)),
               "fail05_never": float((never < 0.5).mean()), "fail05_full": float((full < 0.5).mean()),
               "fail01_never": float((never < 0.1).mean()), "fail01_full": float((full < 0.1).mean()),
               "bf1_never": float(np.nanmean(bf_n)), "bf1_full": float(np.nanmean(bf_f)),
               "hd95_never": float(np.nanmean(hd_n)), "hd95_full": float(np.nanmean(hd_f))}
        if ds == "PolypGen":
            cen = np.array([i.split("_")[0] for i in ids]); C = sorted(set(cen)); G = full - never
            two = []
            for _ in range(2000):
                pick = rng.choice(C, len(C)); vals = []
                for c in pick:
                    ix = np.where(cen == c)[0]; ix = rng.choice(ix, len(ix)); vals.append(G[:, ix])
                two.append(np.concatenate(vals, 1).mean())
            res["two_stage_ci"] = [float(x) for x in np.percentile(two, [2.5, 97.5])]
            res["centre_gains"] = {c: float(G[:, cen == c].mean()) for c in C}
            res["leave_one_centre_out"] = {c: float(G[:, cen != c].mean()) for c in C}
            c4 = np.where(cen == "C4")[0]
            seq = np.array([re.match(r"C4_(\d+)_", ids[i]).group(1) for i in c4]); S = sorted(set(seq)); g4 = G[:, c4]
            b4 = [np.concatenate([g4[:, seq == s] for s in rng.choice(S, len(S))], 1).mean() for _ in range(2000)]
            res["C4_sequences"] = len(S); res["C4_gain"] = float(g4.mean()); res["C4_sequence_ci"] = [float(x) for x in np.percentile(b4, [2.5, 97.5])]
        out[f"{bk}|{ds}"] = res
        print(bk, ds, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}), flush=True)
        if ds == "PolypGen":
            print("   centre gains", {k: round(v, 4) for k, v in res["centre_gains"].items()}, " LOCO", {k: round(v, 4) for k, v in res["leave_one_centre_out"].items()})
json.dump(out, open("results_ext/safety_and_clustering.json", "w"), indent=1)
