"""
Aggregates a run_pipeline.py output file (.jsonl) into headline metrics, and
compares two runs with a paired significance test (Wilcoxon signed-rank on
per-image IoU, matching the paired-comparison style used in the CAGE project's
McNemar's test -- Wilcoxon here since IoU is continuous, not a binary outcome).

Usage:
    python eval.py --results results_full.jsonl
    python eval.py --results results_full.jsonl --compare results_never_refine.jsonl
"""
import argparse
import json

import numpy as np
from scipy.stats import wilcoxon


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize(records: list[dict]) -> dict:
    ious = np.array([r["final_iou"] for r in records])
    n = len(records)
    n_refined = sum(1 for r in records if r.get("refined"))
    n_verified_rejected = sum(
        1 for r in records if r.get("final_mask_source") == "backbone_rejected_refinement"
    )
    n_mllm_calls = sum(1 for r in records if "mllm_adjustment" in r or "mllm_error" in r)

    return {
        "n": n,
        "mean_iou": float(ious.mean()),
        "std_iou": float(ious.std()),
        "median_iou": float(np.median(ious)),
        "mllm_call_rate_pct": 100 * n_mllm_calls / n if n else 0.0,
        "refinement_accepted_rate_pct": 100 * (n_refined - n_verified_rejected) / n if n else 0.0,
        "verification_rejection_rate_pct": (
            100 * n_verified_rejected / n_refined if n_refined else 0.0
        ),
    }


def compare(records_a: list[dict], records_b: list[dict]) -> dict:
    """Paired comparison -- assumes both runs cover the same image_ids."""
    ids_a = {r["image_id"]: r["final_iou"] for r in records_a}
    ids_b = {r["image_id"]: r["final_iou"] for r in records_b}
    shared = sorted(set(ids_a) & set(ids_b))
    if len(shared) < len(ids_a):
        print(f"Warning: only {len(shared)}/{len(ids_a)} image_ids overlap between the two runs")

    a_vals = np.array([ids_a[i] for i in shared])
    b_vals = np.array([ids_b[i] for i in shared])
    diff = a_vals - b_vals

    if np.allclose(diff, 0):
        return {"n_shared": len(shared), "mean_diff": 0.0, "p_value": 1.0}

    stat, p = wilcoxon(a_vals, b_vals)
    return {
        "n_shared": len(shared),
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--compare", default=None, help="second .jsonl file for paired comparison")
    args = parser.parse_args()

    records = load_results(args.results)
    summary = summarize(records)
    print(f"Summary for {args.results}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.compare:
        records_b = load_results(args.compare)
        cmp_result = compare(records, records_b)
        print(f"\nPaired comparison vs {args.compare}:")
        for k, v in cmp_result.items():
            print(f"  {k}: {v}")
