# """
# Full inference pipeline: backbone -> gate -> MLLM refine -> causal verify.

# --fusion-mode controls which stages are active, isolating each component's
# contribution (same pattern as the unified-segdet-fusion project's
# --fusion-mode switch):

#     never_refine   backbone mask only, gate/MLLM/verify all skipped
#                    (floor baseline -- does refinement help AT ALL?)
#     always_refine  skip the gate, MLLM refines every case, verify still runs
#                    (isolates: does the GATE add value, or would refining
#                    everything be just as good?)
#     no_verify      gate + MLLM refine gated-in cases, but accept every
#                    refinement blindly, no causal check
#                    (isolates: does VERIFICATION add value, or is the MLLM
#                    trustworthy enough to accept outright?)
#     full           gate + MLLM + causal verify -- the actual proposed method

# --refinement-method controls HOW the escalated case gets refined:
#     categorical    original expand/shrink/shift/no_change via a chat VLM
#     grounding      LocateAnything-3B bounding-box localization, combined with
#                    the backbone mask's shape

# Saves incrementally after every image (--resume support) -- data-loss lesson
# from the unified-segdet-fusion project's long multi-hour runs.
# """
# import argparse
# import json
# import os
# import time
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv()  # reads .env in the current working directory, if present

# import numpy as np
# import torch
# import yaml
# from PIL import Image, ImageDraw
# from torch.utils.data import DataLoader

# from data.kvasir_loader import KvasirSegDataset
# from data.transforms import eval_transform
# from models.backbone import PolypUNet
# from models.confidence_gate import ConfidenceGate, extract_gate_features
# from models.llm_interface import request_refinement
# from models.grounding_interface import request_grounding, apply_grounding_box
# from models.causal_verify import verify_refinement, compute_differential_region


# def draw_overlay(image_np: np.ndarray, mask_np: np.ndarray, out_path: str):
#     """Draws the predicted mask boundary in red on the original image, for
#     the MLLM to review -- per llm_interface.py's expected input."""
#     img = Image.fromarray((image_np * 255).astype(np.uint8))
#     draw = ImageDraw.Draw(img)
#     ys, xs = np.where(mask_np > 0.5)
#     if len(xs) > 0:
#         # cheap boundary approximation: draw bounding contour via edge pixels
#         edge_mask = mask_np.astype(np.uint8)
#         for y in range(1, edge_mask.shape[0] - 1):
#             for x in range(1, edge_mask.shape[1] - 1):
#                 if edge_mask[y, x] and not edge_mask[y - 1 : y + 2, x - 1 : x + 2].all():
#                     draw.point((x, y), fill=(255, 0, 0))
#     img.save(out_path)


# def apply_boundary_adjustment(mask: np.ndarray, adjustment: str) -> np.ndarray:
#     """Translates the MLLM's categorical adjustment into a concrete mask edit.

#     Adjustment magnitude scales with the polyp's current mask size (sqrt of
#     area, clipped to a sane range) rather than a fixed iteration count.
#     FIXED-MAGNITUDE VERSION (iterations=5 for every case, any polyp size) was
#     tried first and found, via a 293-case train+val calibration sweep, to
#     hurt more often than it helped (165 hurt vs 115 helped) -- a fixed edit
#     is too aggressive on small polyps and too weak on large ones. This is a
#     reproducible, well-defined edit (still needed for causal verification to
#     have a clean differential region to test), just no longer a flat constant.
#     """
#     from scipy import ndimage

#     area = float(mask.sum())
#     # sqrt-of-area scaling: small polyps get a small nudge, large ones a
#     # bigger one. Constant (0.03) and clip range (1-10) are a first-pass
#     # choice -- worth re-sweeping once this mechanism is confirmed to help
#     # at all, same as any other hyperparameter in this pipeline.
#     iterations = int(np.clip(round(0.03 * np.sqrt(max(area, 1.0))), 1, 10))

#     if adjustment == "no_change":
#         return mask.copy()
#     elif adjustment == "expand":
#         return ndimage.binary_dilation(mask, iterations=iterations).astype(np.float32)
#     elif adjustment == "shrink":
#         return ndimage.binary_erosion(mask, iterations=iterations).astype(np.float32)
#     elif adjustment == "shift_boundary":
#         half = max(1, iterations // 2)
#         return ndimage.binary_dilation(
#             ndimage.binary_erosion(mask, iterations=half), iterations=half
#         ).astype(np.float32)
#     else:
#         return mask.copy()


# def run_pipeline(cfg, fusion_mode: str, split: str, output_path: str, api_key: str, resume: bool = False, max_images: int = None, refinement_method: str = "categorical"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     backbone = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(device)
#     ckpt = torch.load(cfg["_backbone_ckpt"], map_location=device)
#     backbone.load_state_dict(ckpt["model"])
#     backbone.eval()

#     gate = None
#     if fusion_mode in ("full", "no_verify"):
#         gate = ConfidenceGate().load(cfg["_gate_path"])

#     data_root = cfg["data"]["root"]
#     ds = KvasirSegDataset(data_root, f"{data_root}/splits/{split}.txt", eval_transform, cfg["data"]["image_size"])
#     if max_images is not None:
#         ds.filenames = ds.filenames[:max_images]
#         print(f"Smoke-test mode: limited to {len(ds.filenames)} images")
#     loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

#     done_ids = set()
#     if resume and Path(output_path).exists():
#         with open(output_path) as f:
#             for line in f:
#                 done_ids.add(json.loads(line)["image_id"])
#         print(f"Resuming: {len(done_ids)} already done, skipping those")

#     overlay_dir = Path("tmp_overlays")
#     overlay_dir.mkdir(exist_ok=True)

#     total_n = len(ds)
#     with open(output_path, "a" if resume else "w") as out_f:
#         for idx, batch in enumerate(loader):
#             image_id = batch["image_id"][0]
#             if image_id in done_ids:
#                 continue
#             print(f"[{idx+1}/{total_n}] Processing {image_id}...", flush=True)

#             image_np = batch["image"][0].numpy()
#             gt_mask = batch["mask"][0].numpy()
#             image_t = batch["image"].permute(0, 3, 1, 2).float().to(device)

#             with torch.no_grad():
#                 mask_logits, confidence = backbone(image_t)
#             backbone_mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze().cpu().numpy()

#             record = {
#                 "image_id": image_id,
#                 "fusion_mode": fusion_mode,
#                 "backbone_confidence": float(confidence.item()),
#                 "refined": False,
#                 "verified_accepted": None,
#                 "final_mask_source": "backbone",
#             }

#             needs_refinement = fusion_mode == "always_refine"
#             if fusion_mode in ("full", "no_verify") and gate is not None:
#                 feats = extract_gate_features(mask_logits, confidence)
#                 needs_refinement = bool(gate.decide(feats, threshold=cfg["gate"]["decision_threshold"])[0])
#             record["gate_flagged"] = needs_refinement
#             print(f"  gate_flagged={needs_refinement}", flush=True)

#             final_mask = backbone_mask
#             if needs_refinement and fusion_mode != "never_refine":
#                 overlay_path = str(overlay_dir / f"{image_id}.jpg")
#                 draw_overlay(image_np, backbone_mask, overlay_path)
#                 try:
#                     if refinement_method == "grounding":
#                         h, w = backbone_mask.shape
#                         grounding = request_grounding(overlay_path, image_id, w, h, api_key, cfg["mllm"]["max_retries"])
#                         time.sleep(2)
#                         if grounding is None:
#                             raise RuntimeError("grounding request failed after retries")
#                         proposed_mask = apply_grounding_box(backbone_mask, grounding.box)
#                         record["grounding_box"] = list(grounding.box)
#                         record["grounding_confidence"] = grounding.confidence
#                         record["refined"] = True
#                     else:  # categorical (default, original method)
#                         refinement = request_refinement(overlay_path, image_id, api_key, cfg["mllm"]["max_retries"])
#                         time.sleep(2)  # proactive rate-limit spacing, same fix as the FaithCAM project
#                         proposed_mask = apply_boundary_adjustment(backbone_mask, refinement.boundary_adjustment)
#                         record["mllm_adjustment"] = refinement.boundary_adjustment
#                         record["mllm_reasoning"] = refinement.reasoning
#                         record["refined"] = True

#                     # Always run the causal verification check so faithfulness_score is
#                     # recorded for calibration purposes, regardless of fusion_mode --
#                     # only whether it GATES acceptance differs by mode.
#                     result = verify_refinement(
#                         lambda img: backbone(img), image_t, backbone_mask, proposed_mask,
#                         image_id, cfg["verification"]["faithfulness_threshold"],
#                     )
#                     record["faithfulness_score"] = result.faithfulness_score

#                     if fusion_mode == "full":
#                         record["verified_accepted"] = result.accepted
#                         if result.accepted:
#                             final_mask = proposed_mask
#                             record["final_mask_source"] = "mllm_verified"
#                         else:
#                             record["final_mask_source"] = "backbone_rejected_refinement"
#                     else:  # no_verify / always_refine -- accept blindly, but score is still logged above
#                         final_mask = proposed_mask
#                         record["final_mask_source"] = "mllm_unverified"

#                 except Exception as e:  # noqa: BLE001
#                     record["mllm_error"] = f"[{refinement_method}] {e}"
#                     record["final_mask_source"] = "backbone_mllm_failed"

#             # metrics against ground truth (only possible because this is eval,
#             # not real deployment -- at real deployment there's no gt_mask)
#             intersection = float((final_mask * gt_mask).sum())
#             union = float(final_mask.sum() + gt_mask.sum() - intersection)
#             record["final_iou"] = intersection / union if union > 0 else 1.0

#             out_f.write(json.dumps(record) + "\n")
#             out_f.flush()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="configs/default.yaml")
#     parser.add_argument("--fusion-mode", required=True,
#                          choices=["never_refine", "always_refine", "no_verify", "full"])
#     parser.add_argument("--split", default="test")
#     parser.add_argument("--backbone-ckpt", required=True)
#     parser.add_argument("--gate-path", default="gate.joblib")
#     parser.add_argument("--output", required=True)
#     parser.add_argument("--api-key", default=os.environ.get("NVIDIA_NIM_API_KEY"))
#     parser.add_argument("--resume", action="store_true")
#     parser.add_argument("--max-images", type=int, default=None, help="limit to first N images, for smoke-testing before a full/costly run")
#     parser.add_argument("--refinement-method", default="categorical", choices=["categorical", "grounding"],
#                          help="categorical = original expand/shrink/shift/no_change via a chat VLM; "
#                               "grounding = LocateAnything-3B bounding-box localization combined with backbone mask shape")
#     args = parser.parse_args()

#     with open(args.config) as f:
#         cfg = yaml.safe_load(f)
#     cfg["_backbone_ckpt"] = args.backbone_ckpt
#     cfg["_gate_path"] = args.gate_path

#     run_pipeline(cfg, args.fusion_mode, args.split, args.output, args.api_key, args.resume, args.max_images, args.refinement_method)

"""
Full inference pipeline: backbone -> gate -> MLLM refine -> causal verify.

--fusion-mode controls which stages are active, isolating each component's
contribution (same pattern as the unified-segdet-fusion project's
--fusion-mode switch):

    never_refine   backbone mask only, gate/MLLM/verify all skipped
                   (floor baseline -- does refinement help AT ALL?)
    always_refine  skip the gate, MLLM refines every case, verify still runs
                   (isolates: does the GATE add value, or would refining
                   everything be just as good?)
    no_verify      gate + MLLM refine gated-in cases, but accept every
                   refinement blindly, no causal check
                   (isolates: does VERIFICATION add value, or is the MLLM
                   trustworthy enough to accept outright?)
    full           gate + MLLM + causal verify -- the actual proposed method

Saves incrementally after every image (--resume support) -- data-loss lesson
from the unified-segdet-fusion project's long multi-hour runs.
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current working directory, if present

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.kvasir_loader import KvasirSegDataset
from data.transforms import eval_transform
from models.backbone import PolypUNet
from models.confidence_gate import ConfidenceGate, extract_gate_features
from models.llm_interface import request_refinement
from models.causal_verify import verify_refinement, compute_differential_region


def draw_overlay(image_np: np.ndarray, mask_np: np.ndarray, out_path: str):
    """Draws the predicted mask boundary in red on the original image, for
    the MLLM to review -- per llm_interface.py's expected input."""
    img = Image.fromarray((image_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) > 0:
        # cheap boundary approximation: draw bounding contour via edge pixels
        edge_mask = mask_np.astype(np.uint8)
        for y in range(1, edge_mask.shape[0] - 1):
            for x in range(1, edge_mask.shape[1] - 1):
                if edge_mask[y, x] and not edge_mask[y - 1 : y + 2, x - 1 : x + 2].all():
                    draw.point((x, y), fill=(255, 0, 0))
    img.save(out_path)


def apply_boundary_adjustment(mask: np.ndarray, adjustment: str) -> np.ndarray:
    """Translates the MLLM's categorical adjustment into a concrete mask edit.

    Adjustment magnitude scales with the polyp's current mask size (sqrt of
    area, clipped to a sane range) rather than a fixed iteration count.
    FIXED-MAGNITUDE VERSION (iterations=5 for every case, any polyp size) was
    tried first and found, via a 293-case train+val calibration sweep, to
    hurt more often than it helped (165 hurt vs 115 helped) -- a fixed edit
    is too aggressive on small polyps and too weak on large ones. This is a
    reproducible, well-defined edit (still needed for causal verification to
    have a clean differential region to test), just no longer a flat constant.
    """
    from scipy import ndimage

    area = float(mask.sum())
    iterations = int(np.clip(round(0.03 * np.sqrt(max(area, 1.0))), 1, 10))

    if adjustment == "no_change":
        return mask.copy()
    elif adjustment == "expand":
        return ndimage.binary_dilation(mask, iterations=iterations).astype(np.float32)
    elif adjustment == "shrink":
        return ndimage.binary_erosion(mask, iterations=iterations).astype(np.float32)
    elif adjustment == "shift_boundary":
        half = max(1, iterations // 2)
        return ndimage.binary_dilation(
            ndimage.binary_erosion(mask, iterations=half), iterations=half
        ).astype(np.float32)
    else:
        return mask.copy()


def run_pipeline(cfg, fusion_mode: str, split: str, output_path: str, api_key: str, resume: bool = False, max_images: int = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(device)
    ckpt = torch.load(cfg["_backbone_ckpt"], map_location=device)
    backbone.load_state_dict(ckpt["model"])
    backbone.eval()

    gate = None
    if fusion_mode in ("full", "no_verify"):
        gate = ConfidenceGate().load(cfg["_gate_path"])

    data_root = cfg["data"]["root"]
    ds = KvasirSegDataset(data_root, f"{data_root}/splits/{split}.txt", eval_transform, cfg["data"]["image_size"])
    if max_images is not None:
        ds.filenames = ds.filenames[:max_images]
        print(f"Smoke-test mode: limited to {len(ds.filenames)} images")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    done_ids = set()
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["image_id"])
        print(f"Resuming: {len(done_ids)} already done, skipping those")

    overlay_dir = Path("tmp_overlays")
    overlay_dir.mkdir(exist_ok=True)

    total_n = len(ds)
    with open(output_path, "a" if resume else "w") as out_f:
        for idx, batch in enumerate(loader):
            image_id = batch["image_id"][0]
            if image_id in done_ids:
                continue
            print(f"[{idx+1}/{total_n}] Processing {image_id}...", flush=True)

            image_np = batch["image"][0].numpy()
            gt_mask = batch["mask"][0].numpy()
            image_t = batch["image"].permute(0, 3, 1, 2).float().to(device)

            with torch.no_grad():
                mask_logits, confidence = backbone(image_t)
            backbone_mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze().cpu().numpy()

            record = {
                "image_id": image_id,
                "fusion_mode": fusion_mode,
                "backbone_confidence": float(confidence.item()),
                "refined": False,
                "verified_accepted": None,
                "final_mask_source": "backbone",
            }

            needs_refinement = fusion_mode == "always_refine"
            if fusion_mode in ("full", "no_verify") and gate is not None:
                feats = extract_gate_features(mask_logits, confidence)
                needs_refinement = bool(gate.decide(feats, threshold=cfg["gate"]["decision_threshold"])[0])
            record["gate_flagged"] = needs_refinement
            print(f"  gate_flagged={needs_refinement}", flush=True)

            final_mask = backbone_mask
            if needs_refinement and fusion_mode != "never_refine":
                overlay_path = str(overlay_dir / f"{image_id}.jpg")
                draw_overlay(image_np, backbone_mask, overlay_path)
                try:
                    refinement = request_refinement(overlay_path, image_id, api_key, cfg["mllm"]["max_retries"])
                    time.sleep(2)  # proactive rate-limit spacing, same fix as the FaithCAM project
                    proposed_mask = apply_boundary_adjustment(backbone_mask, refinement.boundary_adjustment)
                    record["mllm_adjustment"] = refinement.boundary_adjustment
                    record["mllm_reasoning"] = refinement.reasoning
                    record["refined"] = True

                    # Always run the causal verification check so faithfulness_score is
                    # recorded for calibration purposes, regardless of fusion_mode --
                    # only whether it GATES acceptance differs by mode.
                    result = verify_refinement(
                        lambda img: backbone(img), image_t, backbone_mask, proposed_mask,
                        image_id, cfg["verification"]["faithfulness_threshold"],
                    )
                    record["faithfulness_score"] = result.faithfulness_score

                    if fusion_mode == "full":
                        record["verified_accepted"] = result.accepted
                        if result.accepted:
                            final_mask = proposed_mask
                            record["final_mask_source"] = "mllm_verified"
                        else:
                            record["final_mask_source"] = "backbone_rejected_refinement"
                    else:  # no_verify / always_refine -- accept blindly, but score is still logged above
                        final_mask = proposed_mask
                        record["final_mask_source"] = "mllm_unverified"

                except Exception as e:  # noqa: BLE001
                    record["mllm_error"] = str(e)
                    record["final_mask_source"] = "backbone_mllm_failed"

            # metrics against ground truth (only possible because this is eval,
            # not real deployment -- at real deployment there's no gt_mask)
            intersection = float((final_mask * gt_mask).sum())
            union = float(final_mask.sum() + gt_mask.sum() - intersection)
            record["final_iou"] = intersection / union if union > 0 else 1.0

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fusion-mode", required=True,
                         choices=["never_refine", "always_refine", "no_verify", "full"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--backbone-ckpt", required=True)
    parser.add_argument("--gate-path", default="gate.joblib")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default=os.environ.get("NVIDIA_NIM_API_KEY"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="limit to first N images, for smoke-testing before a full/costly run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_backbone_ckpt"] = args.backbone_ckpt
    cfg["_gate_path"] = args.gate_path

    run_pipeline(cfg, args.fusion_mode, args.split, args.output, args.api_key, args.resume, args.max_images)