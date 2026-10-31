"""
Trains the PolypUNet backbone (mask + confidence head) on Kvasir-SEG.

Windows/laptop-GPU lessons carried over from the unified-segdet-fusion and
FaithCAM projects, applied from the start rather than discovered the hard way
again: AMP, gradient clipping, nan/inf batch-skip guard, num_workers=0
default, checkpoint-overwrite protection via an append-only run log, and
--resume support.

Usage:
    python train_backbone.py --config configs/default.yaml --output-dir checkpoints_backbone
    python train_backbone.py --config configs/default.yaml --output-dir checkpoints_backbone --resume
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.kvasir_loader import KvasirSegDataset, make_splits
from data.transforms import train_transform, eval_transform
from models.backbone import PolypUNet, BackboneLoss, compute_iou


def set_seed(seed: int):
    """Seeds Python's random, NumPy, and PyTorch (CPU + CUDA) for a genuinely
    reproducible, independently-verifiable training run -- needed so a second
    seed's result can be reported as real replication evidence, not just
    'we ran it again'."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_run(output_dir: Path, entry: dict):
    log_path = output_dir / "training_run_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    print("Starting train_backbone.py...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility (default 0)")
    args = parser.parse_args()
    print(f"Args parsed: config={args.config}, output_dir={args.output_dir}, seed={args.seed}", flush=True)
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print("Config loaded.", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["root"])
    splits_dir = data_root / "splits"
    if not (splits_dir / "train.txt").exists():
        print("No splits found, creating them...", flush=True)
        make_splits(str(data_root), cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["data"]["seed"])

    print("Loading train dataset...", flush=True)
    train_ds = KvasirSegDataset(
        str(data_root), str(splits_dir / "train.txt"), train_transform, cfg["data"]["image_size"]
    )
    print(f"Train dataset loaded: {len(train_ds)} images", flush=True)
    print("Loading val dataset...", flush=True)
    val_ds = KvasirSegDataset(
        str(data_root), str(splits_dir / "val.txt"), eval_transform, cfg["data"]["image_size"]
    )
    print(f"Val dataset loaded: {len(val_ds)} images", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["backbone"]["batch_size"], shuffle=True,
        num_workers=cfg["backbone"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["backbone"]["batch_size"], shuffle=False,
        num_workers=cfg["backbone"]["num_workers"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}, using device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(device)
    print("Model created and moved to device.", flush=True)
    criterion = BackboneLoss(confidence_weight=cfg["backbone"]["confidence_loss_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["backbone"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    total_epochs = args.epochs or cfg["backbone"]["epochs"]

    ckpt_path = output_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}, continuing at {start_epoch}")

    log_run(output_dir, {
        "event": "run_start", "resume": args.resume, "start_epoch": start_epoch,
        "total_epochs": total_epochs, "timestamp": time.time(),
    })

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        n_skipped = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            images = batch["image"].permute(0, 3, 1, 2).float().to(device)
            masks = batch["mask"].unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                mask_logits, confidence = model(images)
                losses = criterion(mask_logits, confidence, masks)

            if not torch.isfinite(losses["total"]):
                n_skipped += 1
                continue  # nan/inf guard -- skip batch rather than corrupt weights

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses["total"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # validation IoU
        model.eval()
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].permute(0, 3, 1, 2).float().to(device)
                masks = batch["mask"].unsqueeze(1).float().to(device)
                mask_logits, _ = model(images)
                val_ious.extend(compute_iou(mask_logits, masks).cpu().tolist())
        mean_val_iou = sum(val_ious) / max(len(val_ious), 1)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_IoU={mean_val_iou:.4f}, "
              f"skipped={n_skipped}, time={epoch_time/60:.1f}min")

        log_run(output_dir, {
            "event": "epoch_end", "epoch": epoch, "train_loss": avg_loss,
            "val_iou": mean_val_iou, "skipped_batches": n_skipped,
            "epoch_time_sec": epoch_time, "timestamp": time.time(),
        })

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        }, output_dir / f"epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        }, ckpt_path)  # 'latest.pt' for --resume convenience


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        print("\n\n=== CRASHED -- FULL TRACEBACK BELOW ===", flush=True)
        traceback.print_exc()
        input("\nPress Enter to close this window...")