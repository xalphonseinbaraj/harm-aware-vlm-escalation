"""
Offline analysis of eval_all.py logs -> every table / figure needed for the revision.

Input : results_rev\\<backbone-tag>__<vlm>.jsonl   (e.g. strong_s0__qwen7b_p3.jsonl)
        results_rev\\base_<model>.jsonl            (backbone-only baselines, optional)
        results_rev\\efficiency.json               (optional, from efficiency.py)
Output: results_rev\\analysis\\tables.md, summary.json, figures\\*.png

Policies evaluated EXACTLY on the same VLM outputs (VLM queried once per image, temperature 0):
  never         backbone only                         (paper: never_refine)
  always        VLM edit applied to every image        (paper: always_refine)
  always+V      every image escalated, verified        (always_refine with verification)
  gate          learned gate (theta), no verification  (paper: no_verify)
  gate+V        learned gate + verification            (paper: full, PROPOSED)
  conf-only     escalate the k lowest-confidence cases (k = #gate escalations)  -> tests the extra gate features
  random        escalate k random cases, 10,000 draws                           -> Reviewer 3 Results 1
  oracle-diff   escalate iff backbone IoU < 0.7 (difficulty oracle)
  oracle-gain   escalate iff the VLM edit improves IoU (upper bound)

Usage:  python revision\\analyze.py --dir results_rev --theta 0.5 --tau 0.5
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ["iou", "dice", "bf1", "hd95"]
TEMPLATE = "closely hugs the visible edge"   # sentence from the few-shot prompt P3 (copying detector)


def load(path):
    R = [json.loads(l) for l in open(path)]
    R = [r for r in R if "vlm_error" not in r]
    return sorted(R, key=lambda r: r["image_id"])


def arr(R, key, m):
    return np.array([r[key][m] for r in R], dtype=float)


def policy_metrics(R, esc, accept=None):
    """esc: bool array (escalated); accept: bool array (verification accepted) or None (no verification)."""
    use = esc if accept is None else esc & accept
    return {m: np.where(use, arr(R, "prop", m), arr(R, "bb", m)) for m in METRICS}


def wil(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def analyze_run(R, theta, tau, rng):
    n = len(R)
    bb = arr(R, "bb", "iou"); conf = np.array([r["conf"] for r in R])
    gp = np.array([r["gate_prob"] for r in R]); g = gp >= theta
    hard = bb < 0.7
    acc = np.array([(r["F"] is None) or (r["F"] >= tau) for r in R])
    k = int(g.sum())
    cf = np.zeros(n, bool); cf[np.argsort(conf)[:k]] = True
    gain = arr(R, "prop", "iou") > bb + 1e-9
    P = {"never": policy_metrics(R, np.zeros(n, bool)),
         "always": policy_metrics(R, np.ones(n, bool)),
         "always+V": policy_metrics(R, np.ones(n, bool), acc),
         "gate": policy_metrics(R, g),
         "gate+V": policy_metrics(R, g, acc),
         "conf-only": policy_metrics(R, cf),
         "oracle-diff": policy_metrics(R, hard),
         "oracle-gain": policy_metrics(R, gain)}
    calls = {"never": 0, "always": n, "always+V": n, "gate": k, "gate+V": k, "conf-only": k,
             "oracle-diff": int(hard.sum()), "oracle-gain": int(gain.sum())}
    # random gate with the same escalation count
    rand_iou, rand_pr = [], []
    for _ in range(10000):
        f = np.zeros(n, bool); f[rng.choice(n, k, replace=False)] = True
        rand_iou.append(np.where(f, arr(R, "prop", "iou"), bb).mean())
        tp = (f & hard).sum(); rand_pr.append((tp / max(k, 1), tp / max(hard.sum(), 1)))
    rand_iou = np.array(rand_iou); rand_pr = np.array(rand_pr)

    def prf(f):
        tp = (f & hard).sum(); p = tp / max(f.sum(), 1); r_ = tp / max(hard.sum(), 1)
        return {"precision": float(p), "recall": float(r_), "f1": float(2 * p * r_ / (p + r_)) if p + r_ else 0.0}

    # timings (ms)
    tb = np.mean([r["t_backbone_ms"] for r in R]); tg = np.mean([r["t_gate_ms"] for r in R])
    tv = np.mean([r.get("t_vlm_s", np.nan) for r in R]) * 1000
    to = np.mean([r.get("t_overlay_ms", 0) for r in R]); tver = np.nanmean([r.get("t_verify_ms", np.nan) for r in R])
    lat = {p: tb + tg + calls[p] / n * (to + tv + (tver if p.endswith("+V") else 0.0)) for p in calls}
    lat["never"] = tb

    d_always = arr(R, "prop", "iou") - bb
    edits = np.array([r["pixels_changed"] > 0 for r in R])
    Fv = np.array([np.nan if r["F"] is None else r["F"] for r in R])
    m = edits & ~np.isnan(Fv)
    pb = stats.pointbiserialr((d_always[m] > 0).astype(int), Fv[m]) if m.sum() > 5 and len(set(d_always[m] > 0)) > 1 else None

    out = {
        "n": n, "escalation_rate": k / n, "hard_rate": float(hard.mean()),
        "policies": {p: {mm: float(np.mean(v[mm])) for mm in METRICS} for p, v in P.items()},
        "vlm_calls": calls, "mean_latency_ms": {p: float(v) for p, v in lat.items()},
        "timing_ms": {"backbone": float(tb), "gate": float(tg), "overlay": float(to), "vlm": float(tv), "verify": float(tver)},
        "wilcoxon_iou": {"gate+V vs always": wil(P["gate+V"]["iou"], P["always"]["iou"]),
                         "gate+V vs always+V": wil(P["gate+V"]["iou"], P["always+V"]["iou"]),
                         "gate vs always": wil(P["gate"]["iou"], P["always"]["iou"]),
                         "gate+V vs never": wil(P["gate+V"]["iou"], P["never"]["iou"]),
                         "gate+V vs gate": wil(P["gate+V"]["iou"], P["gate"]["iou"]),
                         "always vs never": wil(P["always"]["iou"], P["never"]["iou"])},
        "random_gate": {"iou_mean": float(rand_iou.mean()), "iou_ci95": np.percentile(rand_iou, [2.5, 97.5]).tolist(),
                        "p_random_ge_gate": float((rand_iou >= P["gate"]["iou"].mean()).mean()),
                        "precision_mean": float(rand_pr[:, 0].mean()), "recall_mean": float(rand_pr[:, 1].mean()),
                        "precision_ci95": np.percentile(rand_pr[:, 0], [2.5, 97.5]).tolist()},
        "gate_vs_difficulty": {"learned": prf(g), "conf_only": prf(cf),
                               "auroc_gate_prob": float(roc_auc_score(hard, gp)) if 0 < hard.sum() < n else None,
                               "auroc_conf": float(roc_auc_score(hard, -conf)) if 0 < hard.sum() < n else None},
        "calibration": calib(conf, bb),
        "harm": {"always_delta_easy": float(d_always[~hard].mean()), "always_hurt_frac_easy": float((d_always[~hard] < -1e-9).mean()),
                 "always_delta_hard": float(d_always[hard].mean()), "always_help_frac_hard": float((d_always[hard] > 1e-9).mean()),
                 "always_hurt_frac_hard": float((d_always[hard] < -1e-9).mean())},
        "vlm_behaviour": {"adjustments_all": dict(Counter(r["adj"] for r in R)),
                          "adjustments_escalated": dict(Counter(r["adj"] for r, e in zip(R, g) if e)),
                          "template_copy_rate": float(np.mean([TEMPLATE in (r.get("reasoning") or "") for r in R])),
                          "edits_helped": int((d_always > 1e-9).sum()), "edits_hurt": int((d_always < -1e-9).sum())},
        "verification": {"accept_rate_on_edits": float(acc[edits].mean()) if edits.any() else None,
                         "pointbiserial_F_vs_helped": None if pb is None else {"r": float(pb.statistic), "p": float(pb.pvalue)}},
    }
    curve = []
    for t in np.linspace(0, 1, 41):
        f = gp >= t; curve.append((float(f.mean()), float(np.where(f, arr(R, "prop", "iou"), bb).mean())))
    curve_c = []
    for q in range(0, n + 1, 5):
        f = np.zeros(n, bool); f[np.argsort(conf)[:q]] = True
        curve_c.append((q / n, float(np.where(f, arr(R, "prop", "iou"), bb).mean())))
    return out, P, {"gate": curve, "conf": curve_c, "op": (k / n, float(P["gate"]["iou"].mean()))}, (conf, bb)


def calib(conf, iou, nb=10):
    bins = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(conf, bins) - 1, 0, nb - 1); ece = 0.0; rows = []
    for j in range(nb):
        s = idx == j
        if s.sum():
            ece += s.mean() * abs(conf[s].mean() - iou[s].mean())
            rows.append({"bin": [float(bins[j]), float(bins[j + 1])], "n": int(s.sum()),
                         "mean_conf": float(conf[s].mean()), "mean_iou": float(iou[s].mean())})
    return {"ECE": float(ece), "MAE": float(np.abs(conf - iou).mean()), "pearson": float(stats.pearsonr(conf, iou)[0]),
            "spearman": float(stats.spearmanr(conf, iou)[0]), "conf_range": [float(conf.min()), float(conf.max())], "bins": rows}


def fmt(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pfmt(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rev")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    D = Path(a.dir); O = D / "analysis"; (O / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    runs, pol, curves, cal = {}, {}, {}, {}
    for f in sorted(glob.glob(str(D / "*__*.jsonl"))):
        tag = Path(f).stem; R = load(f)
        if len(R) < 10 or "prop" not in R[0]:
            continue
        runs[tag], pol[tag], curves[tag], cal[tag] = analyze_run(R, a.theta, a.tau, rng)
        print("analysed", tag, len(R))
    json.dump(runs, open(O / "summary.json", "w"), indent=2)

    L = []
    L.append(f"# Revision analysis (theta={a.theta}, tau={a.tau})\n")
    L.append("## Table A. Escalation policies (mean over n test images; IoU / Dice / Boundary-F1 / HD95 px; VLM calls; latency per image)\n")
    L.append("| Run | Policy | IoU | Dice | BF1 | HD95 | VLM calls | ms/image |\n|---|---|---|---|---|---|---|---|")
    for t, r in runs.items():
        for p in ["never", "always", "always+V", "conf-only", "gate", "gate+V", "oracle-diff", "oracle-gain"]:
            v = r["policies"][p]
            L.append(f"| {t} | {p} | {fmt(v['iou'])} | {fmt(v['dice'])} | {fmt(v['bf1'])} | {fmt(v['hd95'],1)} | "
                     f"{r['vlm_calls'][p]}/{r['n']} | {fmt(r['mean_latency_ms'][p],0)} |")
        rg = r["random_gate"]
        L.append(f"| {t} | random (same rate, 10k draws) | {fmt(rg['iou_mean'])} [{fmt(rg['iou_ci95'][0])}, {fmt(rg['iou_ci95'][1])}] | | | | "
                 f"{r['vlm_calls']['gate']}/{r['n']} | |")
    L.append("\n## Table B. Paired Wilcoxon signed-rank tests on per-image IoU\n")
    keys = list(next(iter(runs.values()))["wilcoxon_iou"]) if runs else []
    L.append("| Run | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
    for t, r in runs.items():
        L.append(f"| {t} | " + " | ".join(pfmt(r["wilcoxon_iou"][k]) for k in keys) + " |")
    L.append("\n## Table C. Gate vs. true difficulty (backbone IoU < 0.7)  [Reviewer 3, Results 1]\n")
    L.append("| Run | esc. rate | hard rate | Gate P | Gate R | Gate F1 | Conf-only P / R / F1 | Random P (95% CI) | AUROC gate | AUROC conf | P(random ≥ gate IoU) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for t, r in runs.items():
        gd, rg = r["gate_vs_difficulty"], r["random_gate"]
        L.append(f"| {t} | {fmt(r['escalation_rate'])} | {fmt(r['hard_rate'])} | {fmt(gd['learned']['precision'])} | {fmt(gd['learned']['recall'])} | "
                 f"{fmt(gd['learned']['f1'])} | {fmt(gd['conf_only']['precision'])} / {fmt(gd['conf_only']['recall'])} / {fmt(gd['conf_only']['f1'])} | "
                 f"{fmt(rg['precision_mean'])} ({fmt(rg['precision_ci95'][0])}–{fmt(rg['precision_ci95'][1])}) | {fmt(gd['auroc_gate_prob'])} | "
                 f"{fmt(gd['auroc_conf'])} | {fmt(rg['p_random_ge_gate'])} |")
    L.append("\n## Table D. Confidence-head calibration  [Reviewer 3, Related work 1]\n")
    L.append("| Run | ECE | MAE | Pearson r | Spearman ρ | conf range |\n|---|---|---|---|---|---|")
    for t, r in runs.items():
        c = r["calibration"]
        L.append(f"| {t} | {fmt(c['ECE'])} | {fmt(c['MAE'])} | {fmt(c['pearson'])} | {fmt(c['spearman'])} | {fmt(c['conf_range'][0])}–{fmt(c['conf_range'][1])} |")
    L.append("\n## Table E. Why indiscriminate escalation hurts: effect of the VLM edit by difficulty, and VLM behaviour\n")
    L.append("| Run | ΔIoU easy | % easy hurt | ΔIoU hard | % hard helped | % hard hurt | adjustments (all images) | template-copy rate | verif. accept rate | F vs helped (r, p) |\n|---|---|---|---|---|---|---|---|---|---|")
    for t, r in runs.items():
        h, vb, ve = r["harm"], r["vlm_behaviour"], r["verification"]
        pb = ve["pointbiserial_F_vs_helped"]
        L.append(f"| {t} | {fmt(h['always_delta_easy'])} | {100*h['always_hurt_frac_easy']:.0f}% | {fmt(h['always_delta_hard'])} | "
                 f"{100*h['always_help_frac_hard']:.0f}% | {100*h['always_hurt_frac_hard']:.0f}% | {vb['adjustments_all']} | "
                 f"{100*vb['template_copy_rate']:.0f}% | {fmt(ve['accept_rate_on_edits'])} | "
                 f"{'–' if pb is None else fmt(pb['r']) + ', ' + pfmt(pb['p'])} |")
    L.append("\n## Table F. Timing breakdown (ms, mean per image)\n")
    L.append("| Run | backbone | gate | overlay | VLM call | verification |\n|---|---|---|---|---|---|")
    for t, r in runs.items():
        tm = r["timing_ms"]
        L.append(f"| {t} | {fmt(tm['backbone'],1)} | {fmt(tm['gate'],2)} | {fmt(tm['overlay'],1)} | {fmt(tm['vlm'],0)} | {fmt(tm['verify'],1)} |")

    # pooled over seeds, grouped by (aug, vlm)
    groups = defaultdict(list)
    for t in runs:
        bbtag, vlm = t.split("__"); aug = bbtag.rsplit("_s", 1)[0]
        groups[(aug, vlm)].append(t)
    L.append("\n## Table G. Pooled across seeds (mean ± SD of per-seed means; pooled Wilcoxon over all image-seed pairs)\n")
    L.append("| Backbone / VLM | seeds | never | always | gate+V (proposed) | esc. rate | gate+V vs always p | gate+V vs never p |\n|---|---|---|---|---|---|---|---|")
    for (aug, vlm), ts in groups.items():
        def ms(p):
            v = [runs[t]["policies"][p]["iou"] for t in ts]; return f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
        cat = lambda p: np.concatenate([pol[t][p]["iou"] for t in ts])  # noqa: E731
        er = np.mean([runs[t]["escalation_rate"] for t in ts])
        L.append(f"| {aug} / {vlm} | {len(ts)} | {ms('never')} | {ms('always')} | {ms('gate+V')} | {er:.3f} | "
                 f"{pfmt(wil(cat('gate+V'), cat('always')))} | {pfmt(wil(cat('gate+V'), cat('never')))} |")

    # baselines
    base = sorted(glob.glob(str(D / "base_*.jsonl")))
    eff = {}
    if (D / "efficiency.json").exists():
        for e in json.load(open(D / "efficiency.json")):
            eff[e["name"]] = e
    if base:
        L.append("\n## Table H. Matched baselines (same split / resolution / augmentation / model selection)  [Reviewer 1, c6]\n")
        L.append("| Model | IoU | Dice | BF1 | HD95 | Params (M) | GMACs | ms/img |\n|---|---|---|---|---|---|---|---|")
        for f in base:
            R = [json.loads(l) for l in open(f)]
            nm = R[0]["backbone"]; e = eff.get(nm, {})
            L.append(f"| {nm} ({Path(f).stem}) | " + " | ".join(fmt(np.mean([r['bb'][m] for r in R]), 1 if m == 'hd95' else 3) for m in METRICS)
                     + f" | {fmt(e.get('params_M'),2)} | {fmt(e.get('GMACs'),2)} | {fmt(e.get('latency_ms'),1)} |")
    lam = sorted(glob.glob(str(D / "lambda_*.jsonl")))
    if lam:
        L.append("\n## Table I. Sensitivity to the confidence-loss weight λ (Eq. 1), seed 0, strong augmentation  [Reviewer 3, Method 2]\n")
        L.append("| λ | test IoU | test Dice | conf–IoU Pearson r | ECE | gate AUROC (val) |\n|---|---|---|---|---|---|")
        rows = [("0.5", str(D / "strong_s0__qwen7b_p3.jsonl"), "ckpt_rev/strong_s0")]
        rows += [(Path(f).stem.split("lam")[-1], f, "ckpt_rev/" + Path(f).stem.replace("lambda_", "")) for f in lam]
        for lv, f, ck in sorted(rows, key=lambda z: float(z[0])):
            if not Path(f).exists():
                continue
            R = [json.loads(l) for l in open(f)]
            c = np.array([r["conf"] for r in R]); i = np.array([r["bb"]["iou"] for r in R]); cb = calib(c, i)
            vr = Path(ck) / "gate.val_report.json"
            au = json.load(open(vr))["auroc_gate"] if vr.exists() else None
            L.append(f"| {lv} | {fmt(i.mean())} | {fmt(np.mean([r['bb']['dice'] for r in R]))} | {fmt(cb['pearson'])} | {fmt(cb['ECE'])} | {fmt(au)} |")
    (O / "tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    # figures: risk-coverage and reliability
    if runs:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        for t, cv in curves.items():
            x, y = zip(*sorted(cv["gate"])); ax[0].plot(x, y, "-", lw=1.6, label=f"{t} (gate)")
            x, y = zip(*cv["conf"]); ax[0].plot(x, y, ":", lw=1, color=ax[0].lines[-1].get_color())
            ax[0].plot(*cv["op"], "o", color=ax[0].lines[-1].get_color())
        ax[0].set_xlabel("Fraction of images escalated to the VLM"); ax[0].set_ylabel("Mean test IoU")
        ax[0].set_title("(a) IoU vs. escalation budget (solid: learned gate, dotted: confidence only; ● operating point)", fontsize=9)
        ax[0].legend(fontsize=7)
        for t, (c, i) in cal.items():
            b = runs[t]["calibration"]["bins"]
            ax[1].plot([z["mean_conf"] for z in b], [z["mean_iou"] for z in b], "o-", ms=4, label=t)
        ax[1].plot([0, 1], [0, 1], "--", color="#888"); ax[1].set_xlim(0.3, 1); ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Predicted confidence c"); ax[1].set_ylabel("Realized IoU (bin mean)"); ax[1].set_title("(b) Reliability diagram", fontsize=9)
        ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(O / "figures" / "risk_coverage_reliability.png", dpi=300)
    print(f"\nWrote {O / 'tables.md'} and figures.")


if __name__ == "__main__":
    main()
