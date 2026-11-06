"""
MedSAM as an alternative escalation target (Experiment C, revision R1).

Instead of asking a VLM for one of four global morphological edits, the escalated image is re-segmented by MedSAM
(a SAM ViT-B model fine-tuned on medical images; Hugging Face port "flaviagiammarino/medsam-vit-base", Apache-2.0),
prompted with the bounding box of the backbone mask enlarged by a fixed 10% margin (at least 3 pixels) on each side.
The design is fixed before any run: one box prompt, no point prompts, threshold 0.5 on the predicted probability.

CAUTION for the paper: MedSAM's training corpus includes public endoscopy polyp datasets, which may overlap with
Kvasir-SEG and the external test sets. Its results are therefore an optimistic reference for a capable escalation
target, not a fair accuracy comparison.
"""
import numpy as np
import torch
from PIL import Image

MODEL_NAME = "flaviagiammarino/medsam-vit-base"


class MedSAMRefiner:
    def __init__(self, device: str, name: str = MODEL_NAME, margin: float = 0.10):
        import os
        from transformers import SamModel, SamProcessor
        # Stability on Windows laptop GPUs: the fused GPU attention kernels (flash / memory-efficient / cuDNN SDPA) can hang
        # the driver ("GPU is lost"). Use the plain math attention, which gives the same result and is only slightly slower.
        if torch.cuda.is_available():
            for f in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
                if hasattr(torch.backends.cuda, f):
                    getattr(torch.backends.cuda, f)(False)
        # MEDSAM_DEVICE=cpu runs MedSAM on the CPU (the backbone and verification stay on the GPU).
        device = os.environ.get("MEDSAM_DEVICE", device)
        try:
            self.model = SamModel.from_pretrained(name, attn_implementation="eager").to(device).eval()
        except TypeError:
            self.model = SamModel.from_pretrained(name).to(device).eval()
        self.proc = SamProcessor.from_pretrained(name)
        self.dev = device
        self.margin = margin

    def box_from_mask(self, bmask: np.ndarray):
        ys, xs = np.nonzero(bmask > 0.5)
        if len(xs) == 0:
            return None
        h, w = bmask.shape
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        mx = max(3, int(round(self.margin * (x1 - x0 + 1))))
        my = max(3, int(round(self.margin * (y1 - y0 + 1))))
        return [float(max(0, x0 - mx)), float(max(0, y0 - my)), float(min(w - 1, x1 + mx)), float(min(h - 1, y1 + my))]

    @torch.no_grad()
    def __call__(self, image01: np.ndarray, bmask: np.ndarray):
        """image01: HxWx3 float in [0, 1] (the backbone's input); bmask: HxW binary backbone mask.
        Returns (proposed mask HxW float32, label) where label is 'medsam' or 'no_change' (empty backbone mask)."""
        box = self.box_from_mask(bmask)
        if box is None:
            return bmask.astype(np.float32).copy(), "no_change"
        img = Image.fromarray((np.clip(image01, 0, 1) * 255).astype(np.uint8))
        inp = self.proc(img, input_boxes=[[box]], return_tensors="pt").to(self.dev)
        out = self.model(**inp, multimask_output=False)
        probs = self.proc.image_processor.post_process_masks(
            out.pred_masks.sigmoid().cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu(), binarize=False)
        pm = (probs[0].squeeze().numpy() > 0.5).astype(np.float32)
        if pm.shape != bmask.shape:
            pm = np.array(Image.fromarray((pm * 255).astype(np.uint8)).resize(bmask.shape[::-1], Image.NEAREST)) > 127
            pm = pm.astype(np.float32)
        return pm, "medsam"
