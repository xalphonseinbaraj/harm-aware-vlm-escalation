"""
Kvasir-SEG dataset loader.

Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
1000 colonoscopy images with polyp masks (binary: polyp / not polyp).
Expected directory layout after download+extract:

    <root>/
        images/   *.jpg   (RGB colonoscopy frames)
        masks/    *.jpg   (binary masks, same filename as image)

Download instructions (run once, manually):
    wget https://datasets.simula.no/downloads/kvasir-seg.zip
    unzip kvasir-seg.zip -d <root>/
    # results in <root>/Kvasir-SEG/images and <root>/Kvasir-SEG/masks
"""
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 384,
    ):
        """
        root: path to Kvasir-SEG/ (containing images/ and masks/ subfolders)
        split_file: optional .txt file with one filename stem per line
                    (for train/val/test splits). If None, uses all images.
        transform: callable(image: np.ndarray, mask: np.ndarray) -> (image, mask)
                   see data/transforms.py for the paired-augmentation helper.
        """
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.transform = transform
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.image_dir} to exist. "
                f"Download Kvasir-SEG first -- see module docstring."
            )

        if split_file is not None:
            with open(split_file) as f:
                stems = [line.strip() for line in f if line.strip()]
            self.filenames = [f"{s}.jpg" for s in stems]
        else:
            self.filenames = sorted(os.listdir(self.image_dir))

        # sanity check: every image has a matching mask
        missing = [
            fn for fn in self.filenames if not (self.mask_dir / fn).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images have no matching mask, e.g. {missing[:3]}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        image = np.array(Image.open(self.image_dir / fn).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / fn).convert("L"))
        mask = (mask > 127).astype(np.float32)  # binarize: polyp=1, background=0

        if self.transform is not None:
            image, mask = self.transform(image, mask, self.image_size)

        return {
            "image": image,
            "mask": mask,
            "image_id": fn.replace(".jpg", ""),
        }


def make_splits(root: str, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """
    Kvasir-SEG ships with no official train/val/test split -- this creates one
    deterministically (by seed) and writes split files, matching common practice
    in prior Kvasir-SEG papers (e.g. PraNet uses an 880/120 train/test split;
    we additionally carve out a val split for gate training/calibration).
    Run once; re-running with the same seed reproduces the same split.
    """
    root = Path(root)
    image_dir = root / "images"
    filenames = sorted(os.listdir(image_dir))
    stems = [f.replace(".jpg", "") for f in filenames]

    rng = np.random.RandomState(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_stems = stems[:n_val]
    test_stems = stems[n_val : n_val + n_test]
    train_stems = stems[n_val + n_test :]

    splits_dir = root / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, stem_list in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        with open(splits_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(stem_list))

    print(f"Splits written to {splits_dir}: "
          f"train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")
    return splits_dir
