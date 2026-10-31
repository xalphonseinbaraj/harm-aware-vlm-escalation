"""
Interactive Streamlit demo of the confidence-gated escalation pipeline.

Run locally (in the polyp_llm_seg venv, from the project root):
    pip install streamlit
    streamlit run app.py

Upload any Kvasir-SEG image (or any colonoscopy-style image) and watch it move
through the real pipeline stage by stage, mirroring Figure 1's architecture
diagram: Backbone -> Confidence Gate -> [Accept as-is | VLM Review -> Causal
Verification] -> Final Mask.

This uses your ACTUAL trained checkpoints (backbone + gate) and, for escalated
cases, makes a REAL call to the categorical refinement model configured in
models/llm_interface.py (defaults to local Ollama per the current project
setup) -- nothing here is a lookup table or a fabricated result.
"""
import streamlit as st
import numpy as np
import torch
from PIL import Image
import yaml
import time
import os
from dotenv import load_dotenv
load_dotenv()

from models.backbone import PolypUNet
from models.confidence_gate import ConfidenceGate, extract_gate_features
from models.llm_interface import request_refinement
from models.causal_verify import verify_refinement
from run_pipeline import draw_overlay, apply_boundary_adjustment
from data.transforms import eval_transform

st.set_page_config(page_title="Confidence-Gated Escalation Demo", layout="wide")

# ---------------- Load config + models (cached so this runs once per session) ----------------
@st.cache_resource
def load_pipeline():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = PolypUNet(base_ch=cfg["backbone"]["base_channels"]).to(device)
    ckpt = torch.load("checkpoints_backbone/epoch_22.pt", map_location=device)
    backbone.load_state_dict(ckpt["model"])
    backbone.eval()

    gate = ConfidenceGate().load("gate.joblib")

    return cfg, backbone, gate, device

cfg, backbone, gate, device = load_pipeline()

# ---------------- Header: architecture stages as a live status strip ----------------
st.title("Confidence-Gated Escalation — Live Pipeline Demo")
st.caption("Upload a colonoscopy image. It will move through the same four stages shown in Figure 1.")

stage_cols = st.columns(4)
stage_labels = ["1. Backbone", "2. Confidence Gate", "3. VLM Review", "4. Causal Verification"]
stage_placeholders = [c.empty() for c in stage_cols]

def render_stages(active_idx, skipped=None):
    """Redraws the stage strip, highlighting the current stage green,
    completed stages gray, and skipped stages (e.g. VLM/verify when the gate
    doesn't escalate) shown dimmed -- mirrors the branching in Figure 1."""
    skipped = skipped or set()
    for i, ph in enumerate(stage_placeholders):
        label = stage_labels[i]
        if i in skipped:
            ph.markdown(f"<div style='padding:10px;background:#EEEEEE;color:#999;border-radius:6px;text-align:center;'>{label}<br><i>skipped</i></div>", unsafe_allow_html=True)
        elif i < active_idx:
            ph.markdown(f"<div style='padding:10px;background:#D6E8F5;border-radius:6px;text-align:center;'>{label}<br>&#10003; done</div>", unsafe_allow_html=True)
        elif i == active_idx:
            ph.markdown(f"<div style='padding:10px;background:#4F8F4A;color:white;border-radius:6px;text-align:center;'><b>{label}</b><br>running...</div>", unsafe_allow_html=True)
        else:
            ph.markdown(f"<div style='padding:10px;background:#FAFAFA;border-radius:6px;text-align:center;color:#AAA;'>{label}</div>", unsafe_allow_html=True)

render_stages(-1)

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload a colonoscopy image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np, _ = eval_transform(np.array(image), np.zeros(np.array(image).shape[:2]), cfg["data"]["image_size"])
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    col_img, col_result = st.columns(2)
    col_img.image(image, caption="Uploaded image", use_container_width=True)

    # ---- Stage 1: Backbone ----
    render_stages(0)
    with torch.no_grad():
        mask_logits, confidence = backbone(image_t)
    backbone_mask = (torch.sigmoid(mask_logits) > 0.5).float().squeeze().cpu().numpy()
    conf_val = float(confidence.item())
    time.sleep(0.3)

    # ---- Stage 2: Confidence Gate ----
    render_stages(1)
    feats = extract_gate_features(mask_logits, confidence)
    escalation_prob = float(gate.predict_proba(feats)[0])
    needs_escalation = escalation_prob >= cfg["gate"]["decision_threshold"]
    time.sleep(0.3)

    final_mask = backbone_mask
    result_text = []
    result_text.append(f"**Backbone confidence:** {conf_val:.3f}")
    result_text.append(f"**Escalation score:** {escalation_prob:.3f} (threshold {cfg['gate']['decision_threshold']})")

    if not needs_escalation:
        render_stages(4, skipped={2, 3})
        result_text.append("**Gate decision:** ✅ Not escalated — backbone mask accepted as-is")
    else:
        result_text.append("**Gate decision:** ⚠️ Escalated — sending to VLM review")

        # ---- Stage 3: VLM Review ----
        render_stages(2)
        overlay_path = "tmp_streamlit_overlay.jpg"
        draw_overlay(image_np, backbone_mask, overlay_path)
        try:
            api_key = os.environ.get("NVIDIA_NIM_API_KEY", "not-needed-for-local-ollama")
            refinement = request_refinement(overlay_path, "streamlit_upload", api_key, cfg["mllm"]["max_retries"])
            proposed_mask = apply_boundary_adjustment(backbone_mask, refinement.boundary_adjustment)
            result_text.append(f"**VLM proposed adjustment:** {refinement.boundary_adjustment}")
            result_text.append(f"**VLM reasoning:** _{refinement.reasoning}_")

            # ---- Stage 4: Causal Verification ----
            render_stages(3)
            verify_result = verify_refinement(
                lambda img: backbone(img), image_t, backbone_mask, proposed_mask,
                "streamlit_upload", cfg["verification"]["faithfulness_threshold"],
            )
            result_text.append(f"**Faithfulness score:** {verify_result.faithfulness_score:.3f}")
            if verify_result.accepted:
                final_mask = proposed_mask
                result_text.append("**Verification:** ✅ Accepted — refined mask used")
            else:
                result_text.append("**Verification:** ❌ Rejected — falling back to backbone mask")
            render_stages(4)
        except Exception as e:
            result_text.append(f"**VLM call failed:** {e} — falling back to backbone mask")
            render_stages(4, skipped={3})

    # ---- Final display ----
    overlay_img = np.array(image.resize((cfg["data"]["image_size"], cfg["data"]["image_size"])))
    green = np.zeros_like(overlay_img)
    green[..., 1] = 255
    mask_bool = final_mask.astype(bool)
    alpha = 0.4
    overlay_img = np.where(mask_bool[..., None], (overlay_img * (1 - alpha) + green * alpha).astype(np.uint8), overlay_img)

    col_result.image(overlay_img, caption="Final predicted mask (green overlay)", use_container_width=True)
    col_result.markdown("\n\n".join(result_text))

else:
    st.info("Upload an image above to run it through the live pipeline.")