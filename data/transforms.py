"""
Paired image+mask transforms. Kept dependency-light (numpy + PIL only) so it
works identically whether called from training or from the inference pipeline.
"""
import numpy as np
from PIL import Image


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int):
    """Resize both image and mask to (size, size). Mask uses NEAREST to avoid
    inventing fractional class values at polyp boundaries."""
    img_pil = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    image_out = np.array(img_pil).astype(np.float32) / 255.0
    mask_out = (np.array(mask_pil) > 127).astype(np.float32)
    return image_out, mask_out


def train_transform(image: np.ndarray, mask: np.ndarray, size: int):
    """Resize + light augmentation (horizontal flip, brightness jitter).
    Kept simple deliberately -- augmentation strength is not this paper's
    novelty; over-engineering it here risks obscuring the gate/verification
    ablations with confounding variance."""
    image, mask = resize_pair(image, mask, size)

    if np.random.rand() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    if np.random.rand() < 0.3:
        factor = 0.8 + 0.4 * np.random.rand()  # 0.8x-1.2x brightness
        image = np.clip(image * factor, 0.0, 1.0)

    return image, mask


def eval_transform(image: np.ndarray, mask: np.ndarray, size: int):
    """No augmentation -- deterministic resize only, for val/test/inference."""
    return resize_pair(image, mask, size)
