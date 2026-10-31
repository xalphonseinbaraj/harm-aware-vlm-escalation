"""
Trains the confidence gate on the val split, using a trained backbone
checkpoint to generate (features, needs_refinement_label) pairs via real
outcome supervision (actual IoU against ground truth), not a heuristic.

Includes threshold-guidance reporting (retrieval/refinement rate at each
candidate decision threshold) -- same free, no-extra-cost diagnostic used in
the CAGE project, since every retrained gate needs its own threshold and
reusing an old one risks collapse (all-refine or never-refine).

Usage:
    python train_gate.py --config configs/default.yaml \
        --backbone-ckpt checkpoints_backbone/latest.pt \
        --output gate.joblib
"""
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.kvasir_loader import KvasirSegDataset
from data.transforms import eval_transform
from models.backbone import PolypUNet, compute_iou
from models.confidence_gate import ConfidenceGate, extract_gate_features, build_refinement_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone-ckpt", required=True)
    parser.add_argument("--output", default="gate.joblib")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(device)
    ckpt = torch.load(args.backbone_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data_root = cfg["data"]["root"]
    val_ds = KvasirSegDataset(
        data_root, f"{data_root}/splits/val.txt", eval_transform, cfg["data"]["image_size"]
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    all_features, all_ious = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].permute(0, 3, 1, 2).float().to(device)
            masks = batch["mask"].unsqueeze(1).float().to(device)
            mask_logits, confidence = model(images)
            ious = compute_iou(mask_logits, masks).cpu().numpy()
            feats = extract_gate_features(mask_logits, confidence)
            all_features.append(feats)
            all_ious.append(ious)

    features = np.concatenate(all_features, axis=0)
    ious = np.concatenate(all_ious, axis=0)
    labels = build_refinement_labels(ious, iou_threshold=cfg["gate"]["iou_threshold"])

    print(f"Val set: {len(labels)} cases, {labels.sum()} flagged as needing refinement "
          f"({100*labels.mean():.1f}%) at IoU threshold {cfg['gate']['iou_threshold']}")

    gate = ConfidenceGate()
    gate.fit(features, labels)
    gate.save(args.output)
    print(f"Gate saved to {args.output}")

    # threshold-guidance reporting -- free, uses only the already-computed val features
    print("\nThreshold-guidance (refinement rate at each candidate threshold):")
    probs = gate.predict_proba(features)
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        rate = (probs >= t).mean()
        print(f"  threshold={t:.1f} -> refinement rate={100*rate:.1f}%")


if __name__ == "__main__":
    main()
