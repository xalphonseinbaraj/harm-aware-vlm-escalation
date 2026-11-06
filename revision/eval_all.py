"""
One pass over the TEST split that logs everything needed to answer the reviewers OFFLINE.

For every test image it records:
  * backbone mask metrics (IoU, Dice, Boundary-F1, HD95), backbone confidence c, gate features, gate probability
  * wall-clock latency of each stage (backbone, gate, overlay rendering, VLM call, causal verification)
  * if --vlm is given: the VLM is queried on EVERY image (this is exactly the "always-refine" condition), and
    the proposed mask's metrics + faithfulness score F are logged.
Because the VLM is deterministic (temperature 0) and every image is queried once, ALL escalation policies
(never / always / learned gate +/- verification / random gate / confidence-only gate / oracle / any theta)
are then evaluated exactly, on identical VLM outputs, by analyze.py -- no extra VLM calls.

Usage:
  python revision\\eval_all.py --backbone ckpt_rev\\strong_s0\\best.pt --gate ckpt_rev\\strong_s0\\gate.joblib ^
         --vlm qwen7b_p3 --out results_rev\\strong_s0__qwen7b_p3.jsonl
  (omit --vlm for a backbone-only pass, e.g. for baselines)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from backbones import load_backbone
from common import ROOT, eval_transform, seg_metrics
from data.kvasir_loader import KvasirSegDataset
from models.causal_verify import verify_refinement
from models.confidence_gate import ConfidenceGate, extract_gate_features


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-masks", default=None, help="folder to save backbone/proposal masks (.npz) for figures")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--data-root", default=None,
                    help="external test set folder with images/ and masks/ (e.g. data/TestDataset/CVC-ColonDB); default: Kvasir-SEG test split")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel VLM requests (use 6 for NIM cloud; keep 1 for local Ollama)")
    args = ap.parse_args()

    from run_pipeline import apply_boundary_adjustment, draw_overlay  # size-adaptive edit (paper Sec. 3.4)
    MEDSAM = bool(args.vlm) and args.vlm.startswith("medsam")
    if MEDSAM:  # Experiment C: MedSAM re-segments the escalated image from a box prompt (no VLM, no Ollama)
        from medsam_refiner import MedSAMRefiner
        refiner = None  # created after the device is known
    elif args.vlm:
        from vlm_client import ask_vlm

    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = load_backbone(args.backbone, dev, cfg["backbone"]["base_channels"])
    if MEDSAM:
        refiner = MedSAMRefiner(dev)
    gate = ConfidenceGate().load(args.gate) if args.gate else None
    root = cfg["data"]["root"]
    if args.data_root:
        from common import FolderSegDataset
        ds = FolderSegDataset(args.data_root, cfg["data"]["image_size"])
        print(f"external test set {args.data_root}: {len(ds)} images")
    else:
        ds = KvasirSegDataset(root, f"{root}/splits/{args.split}.txt", eval_transform, cfg["data"]["image_size"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r["image_id"] for r in map(json.loads, open(out)) if "vlm_error" not in r}
        print(f"resuming, {len(done)} done")
    mdir = Path(args.save_masks) if args.save_masks else None
    if mdir:
        mdir.mkdir(parents=True, exist_ok=True)
    ovl = Path("tmp_overlays_rev"); ovl.mkdir(exist_ok=True)

    # warm-up so the first image's latency is not inflated by CUDA init
    with torch.no_grad():
        bb(torch.zeros(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=dev))
    sync(dev)

    # ---- optional parallel prefetch of VLM answers (cloud APIs spend most time waiting in a queue)
    cache_path = out.with_suffix(".vlmcache.jsonl"); cache = {}
    if args.vlm and cache_path.exists():
        cache = {r["image_id"]: r["v"] for r in map(json.loads, open(cache_path))}
    if args.vlm and args.workers > 1 and not MEDSAM:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        todo = []
        for i in range(len(ds)):
            s = ds[i]; iid = s["image_id"]
            if iid in done or iid in cache:
                continue
            x = torch.from_numpy(s["image"]).permute(2, 0, 1)[None].float().to(dev)
            with torch.no_grad():
                ml, _ = bb(x)
            bmask = (torch.sigmoid(ml) > 0.5).float().squeeze().cpu().numpy()
            op = str(ovl / f"{iid}.jpg"); draw_overlay(s["image"], bmask, op); todo.append((iid, op))
        print(f"prefetching {len(todo)} VLM answers with {args.workers} parallel requests ...", flush=True)
        with ThreadPoolExecutor(args.workers) as ex, open(cache_path, "a") as cf:
            futs = {ex.submit(ask_vlm, args.vlm, op): iid for iid, op in todo}
            for k, fu in enumerate(as_completed(futs), 1):
                iid = futs[fu]
                try:
                    v = fu.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  VLM failed on {iid} ({e}); it will be retried in the main pass", flush=True); continue
                v.pop("raw", None); cache[iid] = v
                cf.write(json.dumps({"image_id": iid, "v": v}) + "\n"); cf.flush()
                print(f"  prefetched {k}/{len(todo)} {iid} adj={v['adjustment']} ({v['latency_s']:.0f}s)", flush=True)

    with open(out, "a") as f:
        for i in range(len(ds)):
            s = ds[i]; iid = s["image_id"]
            if iid in done:
                continue
            x = torch.from_numpy(s["image"]).permute(2, 0, 1)[None].float().to(dev); gt = s["mask"]
            rec = {"image_id": iid, "backbone": bb.name, "vlm": args.vlm}

            sync(dev); t = time.perf_counter()
            with torch.no_grad():
                ml, c = bb(x)
            sync(dev); rec["t_backbone_ms"] = 1000 * (time.perf_counter() - t)
            bmask = (torch.sigmoid(ml) > 0.5).float().squeeze().cpu().numpy()
            rec["conf"] = float(c.item())
            rec["bb"] = seg_metrics(bmask, gt)

            t = time.perf_counter()
            feats = extract_gate_features(ml.float(), c.float())
            rec["feats"] = feats[0].tolist()
            if gate:
                rec["gate_prob"] = float(gate.predict_proba(feats)[0])
            rec["t_gate_ms"] = 1000 * (time.perf_counter() - t)

            if args.vlm:
                t = time.perf_counter()
                op = str(ovl / f"{iid}.jpg"); draw_overlay(s["image"], bmask, op)
                rec["t_overlay_ms"] = 1000 * (time.perf_counter() - t)
                try:
                    if MEDSAM:
                        sync(dev); t0 = time.perf_counter()
                        pmask, lab = refiner(s["image"], bmask)
                        sync(dev); v = {"adjustment": lab, "reasoning": "", "latency_s": time.perf_counter() - t0,
                                        "vlm_confidence": float("nan"), "prompt_tokens": None, "completion_tokens": None}
                    else:
                        v = cache[iid] if iid in cache else ask_vlm(args.vlm, op)
                    rec.update({"adj": v["adjustment"], "reasoning": v["reasoning"], "t_vlm_s": v["latency_s"],
                                "vlm_conf": v["vlm_confidence"], "prompt_tokens": v["prompt_tokens"],
                                "completion_tokens": v["completion_tokens"]})
                    if not MEDSAM:
                        pmask = apply_boundary_adjustment(bmask, v["adjustment"])
                    rec["prop"] = seg_metrics(pmask, gt)
                    rec["pixels_changed"] = int(np.logical_xor(pmask > 0.5, bmask > 0.5).sum())
                    t = time.perf_counter()
                    vr = verify_refinement(lambda im: bb(im), x, bmask, pmask, iid, args.tau)
                    sync(dev); rec["t_verify_ms"] = 1000 * (time.perf_counter() - t)
                    rec["F"] = vr.faithfulness_score; rec["accepted"] = bool(vr.accepted)
                    if mdir:
                        np.savez_compressed(mdir / f"{iid}.npz", bb=np.packbits(bmask > 0.5), prop=np.packbits(pmask > 0.5))
                except Exception as e:  # noqa: BLE001
                    # do NOT write a record: stop, so that a re-run retries this image
                    print(f"\n*** VLM call failed on {iid}: {e}\n*** Is Ollama running (ollama serve) / is the internet up for NIM? "
                          f"Fix it and run the same command again; it resumes from this image.", flush=True)
                    sys.exit(1)
            elif mdir:
                np.savez_compressed(mdir / f"{iid}.npz", bb=np.packbits(bmask > 0.5))

            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i+1}/{len(ds)}] {iid} IoU={rec['bb']['iou']:.3f} gate={rec.get('gate_prob', float('nan')):.2f} "
                  f"adj={rec.get('adj', '-')}", flush=True)


if __name__ == "__main__":
    main()
