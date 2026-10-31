"""
Confidence gate: decides whether a backbone-produced mask should be sent to
the MLLM for refinement, or accepted as-is.

Same design philosophy as CAGE's retrieval gate: a small, externally-trained
classifier learned from real outcome supervision (did this case actually need
refinement, i.e. was the backbone's IoU low?) -- not a fixed threshold on the
backbone's own self-reported confidence, and not a training-free heuristic.

Feature vector per case (kept small and cheap -- all derived from the
backbone's own forward pass, no extra model calls):
    1. predicted confidence (backbone's own IoU-prediction head output)
    2. predicted mask area fraction (polyp pixels / total pixels)
    3. mask probability entropy (mean pixel-wise binary entropy -- spatial
       uncertainty the confidence head alone doesn't capture)
    4. boundary complexity (perimeter-to-area ratio of the predicted mask --
       proxy for "is this a clean blob or a messy/fragmented prediction")
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


def extract_gate_features(mask_logits: torch.Tensor, confidence_pred: torch.Tensor) -> np.ndarray:
    """
    mask_logits: (B, 1, H, W) raw logits from the backbone
    confidence_pred: (B,) backbone's own predicted IoU
    Returns: (B, 4) numpy feature array
    """
    probs = torch.sigmoid(mask_logits).squeeze(1)  # (B, H, W)
    B = probs.shape[0]

    area_frac = probs.mean(dim=(1, 2))  # (B,)

    eps = 1e-6
    entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
    mean_entropy = entropy.mean(dim=(1, 2))  # (B,)

    # boundary complexity: count sign changes of (prob > 0.5) along rows+cols,
    # normalized by predicted mask area -- cheap CPU-free proxy, no cv2 dependency
    binary = (probs > 0.5).float()
    row_changes = (binary[:, :, 1:] != binary[:, :, :-1]).float().sum(dim=(1, 2))
    col_changes = (binary[:, 1:, :] != binary[:, :-1, :]).float().sum(dim=(1, 2))
    perimeter_proxy = row_changes + col_changes
    area_pixels = binary.sum(dim=(1, 2)) + eps
    boundary_complexity = perimeter_proxy / area_pixels

    features = torch.stack(
        [confidence_pred, area_frac, mean_entropy, boundary_complexity], dim=1
    )
    return features.detach().cpu().numpy()


class ConfidenceGate:
    """Wraps a scikit-learn LogisticRegression -- deliberately simple model
    class, matching CAGE's gate design (a small externally-trained classifier,
    not a deep network) since the feature vector is only 4-dimensional and a
    simple model is easier to inspect/calibrate/report a threshold sweep for."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight="balanced")
        self.fitted = False

    def fit(self, features: np.ndarray, needs_refinement_labels: np.ndarray):
        """
        features: (N, 4) from extract_gate_features, accumulated over a
                  training/calibration pass over the val split
        needs_refinement_labels: (N,) binary -- 1 if backbone IoU fell below
                  the "needs refinement" threshold on that case, 0 otherwise
                  (outcome supervision -- computed from real IoU, not a guess)
        """
        X = self.scaler.fit_transform(features)
        self.clf.fit(X, needs_refinement_labels)
        self.fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Gate not fitted yet -- call fit() or load() first")
        X = self.scaler.transform(features)
        return self.clf.predict_proba(X)[:, 1]  # P(needs refinement)

    def decide(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(features) >= threshold

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "clf": self.clf}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self.fitted = True
        return self


def build_refinement_labels(iou_scores: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    """Outcome label: 1 (needs refinement) if the backbone's actual IoU on
    that case fell below iou_threshold. iou_threshold is a hyperparameter to
    sweep -- report threshold-vs-refinement-rate curve in the paper, same as
    train_gate.py's threshold-guidance reporting pattern used in CAGE."""
    return (iou_scores < iou_threshold).astype(int)
