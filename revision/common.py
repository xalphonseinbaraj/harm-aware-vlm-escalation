"""
Shared utilities for the BSPC major-revision experiments.

Run every revision script FROM THE PROJECT ROOT (E:\\polyp_llm_seg), e.g.
    python revision\\train_backbone_v2.py --seed 0 --aug strong
so that the existing `models/` and `data/` packages are importable.

Contents
  * set_seed                  -- full reproducibility (python, numpy, torch, cuDNN)
  * strong_train_transform    -- stronger augmentation (Reviewer 1, comment 5)
  * light_train_transform     -- the ORIGINAL augmentation (hflip + brightness), kept for the ablation
  * seg_metrics               -- IoU, Dice, Boundary-F1 (Reviewer 3, Results 2), HD95
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.transforms import resize_pair, eval_transform, train_transform as light_train_transform  # noqa: E402,F401


# --------------------------------------------------------------------------- seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- augmentation
def strong_train_transform(image: np.ndarray, mask: np.ndarray, size: int):
    """Stronger, standard polyp-segmentation augmentation (cf. PraNet / Polyp-PVT practice).

    Geometric (applied identically to image and mask):
      - horizontal flip p=0.5, vertical flip p=0.5
      - random rotation in [-30, 30] deg, p=0.5
      - random scale (zoom) in [0.75, 1.25] with re-crop/pad to size, p=0.5
    Photometric (image only):
      - brightness, contrast, saturation in [0.8, 1.2], hue-free (keeps mucosa colour)
      - Gaussian blur sigma in [0.1, 1.5], p=0.2
    Every transformed sample is a NEW random draw each epoch, so 700 training
    images yield 700 x epochs distinct training views.
    """
    img = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    msk = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)

    if random.random() < 0.5:
        img, msk = img.transpose(Image.FLIP_LEFT_RIGHT), msk.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img, msk = img.transpose(Image.FLIP_TOP_BOTTOM), msk.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        ang = random.uniform(-30, 30)
        img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        msk = msk.rotate(ang, resample=Image.NEAREST, fillcolor=0)
    if random.random() < 0.5:
        s = random.uniform(0.75, 1.25)
        ns = max(8, int(round(size * s)))
        img2, msk2 = img.resize((ns, ns), Image.BILINEAR), msk.resize((ns, ns), Image.NEAREST)
        if ns >= size:  # zoom in: random crop
            x0, y0 = random.randint(0, ns - size), random.randint(0, ns - size)
            img, msk = img2.crop((x0, y0, x0 + size, y0 + size)), msk2.crop((x0, y0, x0 + size, y0 + size))
        else:           # zoom out: pad
            canvas_i, canvas_m = Image.new("RGB", (size, size)), Image.new("L", (size, size))
            x0, y0 = random.randint(0, size - ns), random.randint(0, size - ns)
            canvas_i.paste(img2, (x0, y0)); canvas_m.paste(msk2, (x0, y0))
            img, msk = canvas_i, canvas_m
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        if random.random() < 0.5:
            img = enh(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))

    image_out = np.asarray(img).astype(np.float32) / 255.0
    mask_out = (np.asarray(msk) > 127).astype(np.float32)
    return image_out, mask_out


TRANSFORMS = {"light": light_train_transform, "strong": strong_train_transform}


# --------------------------------------------------------------------------- metrics
def _boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(bool)
    return m & ~ndimage.binary_erosion(m, iterations=1, border_value=0)


def seg_metrics(pred: np.ndarray, gt: np.ndarray, bf_tol: int = 2) -> dict:
    """pred, gt: binary HxW arrays. Returns IoU, Dice, Boundary-F1 (tolerance bf_tol px), HD95 (px)."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter, ps, gs = (p & g).sum(), p.sum(), g.sum()
    union = ps + gs - inter
    iou = inter / union if union > 0 else 1.0
    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else 1.0

    bp, bg = _boundary(p), _boundary(g)
    if bp.sum() == 0 and bg.sum() == 0:
        bf, hd95 = 1.0, 0.0
    elif bp.sum() == 0 or bg.sum() == 0:
        bf, hd95 = 0.0, float(np.hypot(*p.shape))
    else:
        dt_g = ndimage.distance_transform_edt(~bg)
        dt_p = ndimage.distance_transform_edt(~bp)
        prec = (dt_g[bp] <= bf_tol).mean()
        rec = (dt_p[bg] <= bf_tol).mean()
        bf = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        hd95 = float(np.percentile(np.concatenate([dt_g[bp], dt_p[bg]]), 95))
    return {"iou": float(iou), "dice": float(dice), "bf1": float(bf), "hd95": hd95}


def to_tensor_batch(image_np: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)


# --------------------------------------------------------------------------- external test sets (PraNet "TestDataset" layout)
class FolderSegDataset:
    """<root>/images/* and <root>/masks/* with matching file stems (any image extension).
    Returns the same dict as KvasirSegDataset after eval_transform, so eval_all.py can use either."""
    EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(self, root: str, size: int = 384):
        root = Path(root); self.size = size
        imgs = {p.stem: p for p in sorted((root / "images").iterdir()) if p.suffix.lower() in self.EXT}
        msks = {p.stem: p for p in sorted((root / "masks").iterdir()) if p.suffix.lower() in self.EXT}
        self.items = [(k, imgs[k], msks[k]) for k in sorted(imgs) if k in msks]
        if not self.items:
            raise FileNotFoundError(f"no image/mask pairs found under {root}")
        self.prefix = root.name.replace(" ", "_")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, ip, mp = self.items[i]
        image = np.asarray(Image.open(ip).convert("RGB"))
        mask = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.float32)
        image, mask = eval_transform(image, mask, self.size)
        return {"image": image, "mask": mask, "image_id": f"{self.prefix}_{k}"}
