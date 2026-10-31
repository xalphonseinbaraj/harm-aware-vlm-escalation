"""
Causal faithfulness verification for MLLM-proposed mask refinements.

Direct extension of the FaithCAM project's per-claim causal deletion test
(delete the region a claim is about, check whether model evidence for that
claim actually drops) -- here the "claim" is the MLLM's proposed boundary
adjustment, and "evidence" is the backbone's own logit output re-run on a
perturbed image, in LOGIT space (not probability/sigmoid space) to avoid the
saturation bug already found and fixed in the FaithCAM project.

Core idea per proposed adjustment:
  1. Compute the DIFFERENTIAL REGION -- the pixels where the proposed mask
     disagrees with the backbone's original mask (expand: newly-added pixels;
     shrink: newly-removed pixels; shift: pixels that moved across the
     boundary).
  2. Re-run the backbone on the image with that differential region blacked
     out (deletion), and separately with an equally-sized RANDOM region
     blacked out (control), matching FaithCAM's random-masking control.
  3. Compare the logit drop at the differential region's own predicted pixels
     against the control's drop. A genuinely-informative adjustment should
     show a real, above-control logit drop when its region is deleted
     (evidence the model relies on that region) -- if the drop is no bigger
     than the random control, the MLLM's claim is not causally grounded, and
     the refinement is REJECTED (fall back to the original backbone mask).

This is a proxy for ground truth (we don't have a radiologist in the loop at
inference time) -- it tests INTERNAL CONSISTENCY of the backbone's own
evidence, the same epistemic status FaithCAM's causal test has for text
claims. Report this limitation explicitly in the paper, same as FaithCAM does.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class VerificationResult:
    image_id: str
    differential_logit_drop: float
    control_logit_drop: float
    faithfulness_score: Optional[float]  # differential_logit_drop - control_logit_drop; None if not applicable (no_change)
    accepted: bool


def compute_differential_region(original_mask: np.ndarray, proposed_mask: np.ndarray) -> np.ndarray:
    """XOR of the two binary masks -- the region where they disagree."""
    return np.logical_xor(original_mask.astype(bool), proposed_mask.astype(bool)).astype(np.float32)


def _mean_logit_in_region(mask_logits: torch.Tensor, region: torch.Tensor) -> float:
    """Mean raw logit (pre-sigmoid) within a region -- logit space per the
    saturation fix, not probability space."""
    region = region.bool()
    if region.sum() == 0:
        return 0.0
    return mask_logits.squeeze(0).squeeze(0)[region].mean().item()


def verify_refinement(
    backbone_forward_fn,
    image: torch.Tensor,  # (1, 3, H, W), already normalized for the backbone
    original_mask: np.ndarray,
    proposed_mask: np.ndarray,
    image_id: str,
    faithfulness_threshold: float = 0.5,
    seed: int = 0,
) -> VerificationResult:
    """
    backbone_forward_fn: callable(image_tensor) -> (mask_logits, confidence),
        i.e. models.backbone.PolypUNet's forward, passed in rather than
        imported directly so this module stays model-agnostic.
    """
    diff_region = compute_differential_region(original_mask, proposed_mask)

    # FIX (found via seed-1 multi-seed robustness check): a "no_change" proposal
    # produces an EMPTY differential region -- there is nothing to causally test,
    # and the faithfulness score previously defaulted to 0 in this case, which
    # always fails the threshold and mechanically REJECTS every no_change
    # proposal regardless of whether no_change was the correct call. Accepting a
    # no-op can never cause harm (the mask is unchanged either way), so an empty
    # differential region is now treated as a trivial accept rather than run
    # through logit-drop machinery that has nothing to measure.
    if diff_region.sum() == 0:
        return VerificationResult(
            image_id=image_id,
            differential_logit_drop=0.0,
            control_logit_drop=0.0,
            faithfulness_score=None,  # not applicable -- nothing was tested, no_change auto-accepted
            accepted=True,
        )

    with torch.no_grad():
        base_logits, _ = backbone_forward_fn(image)
        base_score = _mean_logit_in_region(base_logits, torch.from_numpy(diff_region))

        # deletion: black out the differential region in the input image
        deleted_image = image.clone()
        region_t = torch.from_numpy(diff_region).to(image.device)
        deleted_image = deleted_image * (1 - region_t)  # broadcast over channel dim
        deleted_logits, _ = backbone_forward_fn(deleted_image)
        deleted_score = _mean_logit_in_region(deleted_logits, torch.from_numpy(diff_region))
        differential_drop = base_score - deleted_score

        # control: random region of the same size, elsewhere in the image
        rng = np.random.RandomState(seed)
        control_region = _random_region_same_size(diff_region, rng)
        control_image = image.clone()
        control_t = torch.from_numpy(control_region).to(image.device)
        control_image = control_image * (1 - control_t)
        control_logits, _ = backbone_forward_fn(control_image)
        control_score = _mean_logit_in_region(control_logits, torch.from_numpy(control_region))
        control_drop = _mean_logit_in_region(base_logits, torch.from_numpy(control_region)) - control_score

    faithfulness_score = differential_drop - control_drop
    accepted = faithfulness_score >= faithfulness_threshold

    return VerificationResult(
        image_id=image_id,
        differential_logit_drop=differential_drop,
        control_logit_drop=control_drop,
        faithfulness_score=faithfulness_score,
        accepted=accepted,
    )


def _random_region_same_size(reference_region: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Random contiguous-ish region matching the pixel COUNT of reference_region
    (not its shape) -- same control-region philosophy as FaithCAM's random
    masking baseline."""
    h, w = reference_region.shape
    n_pixels = int(reference_region.sum())
    control = np.zeros_like(reference_region)
    if n_pixels == 0:
        return control
    flat_idx = rng.choice(h * w, size=n_pixels, replace=False)
    control_flat = control.reshape(-1)
    control_flat[flat_idx] = 1.0
    return control_flat.reshape(h, w)