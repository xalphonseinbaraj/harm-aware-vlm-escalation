"""
Parameters, FLOPs and measured GPU latency for every segmentation backbone (Reviewer 3, Results 3).
VLM latency per call is measured inside eval_all.py (field t_vlm_s) and combined in analyze.py.

Usage:
  python revision\\efficiency.py --ckpts ckpt_rev\\strong_s0\\best.pt ckpt_rev\\base_unet_r34\\best.pt ... --out results_rev\\efficiency.json
"""
import argparse
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from backbones import load_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ck in a.ckpts:
        bb = load_backbone(ck, dev)
        x = torch.rand(1, 3, a.size, a.size, device=dev)
        params = sum(p.numel() for p in bb.parameters())
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            bb(x)
        flops = fc.get_total_flops()
        with torch.no_grad():
            for _ in range(20):
                bb(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(a.runs):
                bb(x)
            if dev == "cuda":
                torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t) / a.runs
        r = {"ckpt": ck, "name": bb.name, "params_M": params / 1e6, "GFLOPs": flops / 1e9, "GMACs": flops / 2e9,  # most polyp papers' "FLOPs" are MACs
             "latency_ms": ms, "fps": 1000 / ms, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
        rows.append(r); print(r)
    json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
