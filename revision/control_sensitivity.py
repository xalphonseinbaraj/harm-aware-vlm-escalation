"""
Repeated-control sensitivity of causal verification (reviewer check, revision R1).

The faithfulness score F (paper Eq. 2) compares the logit drop when the edited region is deleted with the drop for ONE random
control region (fixed seed). This script repeats the verification with 10 different control draws (seeds 0-9) for every
escalated, edited image of an existing result file, WITHOUT calling any VLM (the recorded VLM adjustment is reused), and reports
how often the accept/reject decision changes and how much the final gate+V IoU varies across control draws.

Run from the project root, e.g.
    python revision\\control_sensitivity.py
It prints one summary line per run and writes results_rev\\control_sensitivity.json. Runtime: a few minutes on the GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import load_backbone  # noqa: E402
from common import ROOT, eval_transform, seg_metrics  # noqa: E402
from data.kvasir_loader import KvasirSegDataset  # noqa: E402
from models.causal_verify import verify_refinement  # noqa: E402
from run_pipeline import apply_boundary_adjustment  # noqa: E402

SEEDS = list(range(10))
TD = "data/TestDataset"
RUNS = [  # (result file, backbone checkpoint, external data root or None)
    *[(f"results_rev/strong_s{s}__medgemma4b_p1.jsonl", f"ckpt_rev/strong_s{s}/best.pt", None) for s in range(3)],
    *[(f"results_rev/strong_s{s}__llama11b_p1_nim.jsonl", f"ckpt_rev/strong_s{s}/best.pt", None) for s in range(3)],
    ("results_rev/segformer_s0__medgemma4b_p1.jsonl", "ckpt_rev/base_segformer_mitb2/best.pt", None),
    *[(f"results_ext/{d}_segformer_s0__medgemma4b_p1.jsonl", "ckpt_rev/base_segformer_mitb2/best.pt", f"{TD}/{d}")
      for d in ("CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB")],
]


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]; root = cfg["data"]["root"]
    summary = {}
    for res, ck, ext in RUNS:
        if not Path(res).exists():
            print(f"skip (missing) {res}"); continue
        R = {r["image_id"]: r for r in map(json.loads, open(res)) if "vlm_error" not in r}
        bb = load_backbone(ck, dev, cfg["backbone"]["base_channels"])
        if ext:
            from common import FolderSegDataset
            ds = FolderSegDataset(ext, size)
        else:
            ds = KvasirSegDataset(root, f"{root}/splits/test.txt", eval_transform, size)
        never, gate_v = [], {s: [] for s in SEEDS}
        flips = edited = 0
        for i in range(len(ds)):
            smp = ds[i]; iid = smp["image_id"]
            if iid not in R:
                continue
            r = R[iid]; gt = smp["mask"]
            bmask_iou, prop_iou = r["bb"]["iou"], r["prop"]["iou"]
            never.append(bmask_iou)
            esc = r["gate_prob"] >= 0.5
            if not esc or r["adj"] == "no_change":
                for s in SEEDS:
                    gate_v[s].append(bmask_iou)
                continue
            x = torch.from_numpy(smp["image"]).permute(2, 0, 1)[None].float().to(dev)
            with torch.no_grad():
                ml, _ = bb(x)
                bmask = (torch.sigmoid(ml) > 0.5).float().squeeze().cpu().numpy()
                pmask = apply_boundary_adjustment(bmask, r["adj"])
                decisions = []
                for s in SEEDS:
                    vr = verify_refinement(lambda im: bb(im), x, bmask, pmask, iid, 0.5, seed=s)
                    decisions.append(bool(vr.accepted))
                    gate_v[s].append(prop_iou if vr.accepted else bmask_iou)
            edited += 1
            flips += int(len(set(decisions)) > 1)
        ious = np.array([np.mean(gate_v[s]) for s in SEEDS])
        summary[res] = {"n": len(never), "edited_escalated": edited, "decision_changes": flips,
                        "gateV_mean": float(ious.mean()), "gateV_sd": float(ious.std(ddof=1)),
                        "gateV_min": float(ious.min()), "gateV_max": float(ious.max()), "never": float(np.mean(never))}
        print(f"{res}: edited+escalated {edited}, decision changed across 10 controls for {flips} "
              f"({100 * flips / max(edited, 1):.0f}%); gate+V IoU {ious.mean():.4f} +/- {ious.std(ddof=1):.4f} "
              f"(range {ious.min():.4f}-{ious.max():.4f})", flush=True)
    Path("results_rev").mkdir(exist_ok=True)
    json.dump(summary, open("results_rev/control_sensitivity.json", "w"), indent=1)
    print("\nCONTROL SENSITIVITY DONE")


if __name__ == "__main__":
    main()
