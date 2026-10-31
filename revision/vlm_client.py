"""
Configurable VLM client for the revision experiments.

Reviewer 3 (Method 3, Results 2) asked exactly which prompts and VLMs were used.
The submitted runs used TWO different escalation targets without the manuscript
saying so clearly:
  * seed-0 runs (13 Aug 2026): Llama-3.2-11B-Vision-Instruct via NVIDIA NIM, prompt P1 (plain instruction)
  * seed-1 runs (25 Aug 2026): Qwen2.5-VL-7B via local Ollama,              prompt P3 (few-shot)
This module makes the escalation target an explicit, logged experimental factor.

Prompts (verbatim, reproduced in the revised paper's Appendix):
  P1  plain instruction (original prompt)
  P2  over/under-segmentation checklist (explored on 30-image smoke tests only)
  P3  three worked few-shot examples (current models/llm_interface.py prompt)

Named configurations (--vlm):
  llama11b_p1_nim     meta/llama-3.2-11b-vision-instruct on NVIDIA NIM (needs NVIDIA_NIM_API_KEY in .env)
  llama11b_p1_ollama  llama3.2-vision:11b on local Ollama (same weights; use if NIM is unavailable)
  qwen7b_p3           qwen2.5vl:7b on local Ollama, few-shot prompt (as in the seed-1 runs)
  qwen7b_p1           qwen2.5vl:7b on local Ollama, plain prompt (control: separates model from prompt)
  medgemma4b_p1/_p3   medgemma:4b (Google MedGemma, medical VLM) on local Ollama, prompt P1 / P3
"""
import base64
import json
import os
import time

import requests

from common import ROOT  # noqa: F401  (puts project root on sys.path)
from models.llm_interface import _extract_json_object

JSON_SHAPE = ('{"reasoning": "<one or two sentences explaining what you see and why>", '
              '"boundary_adjustment": "<one of: expand, shrink, shift_boundary, no_change>", '
              '"adjustment_region": "<brief location, e.g. \'left edge\', \'entire boundary\'>", '
              '"confidence": <number between 0 and 1>}')

P1 = f"""You are assisting with polyp boundary review in a colonoscopy image.
You will see the original image with the current predicted polyp boundary overlaid in red.
The automatic segmentation model flagged this case as uncertain.
Decide whether the boundary should be expanded, shrunk, shifted, or left unchanged,
and give a short reasoning grounded in what you actually see in the image
(color, texture, or edge cues distinguishing polyp tissue from surrounding mucosa).

Respond with ONLY a single JSON object, no other text before or after it, in exactly this shape:
{JSON_SHAPE}"""

P3 = """You are assisting with polyp boundary review in a colonoscopy image.
You will see the original image with the current predicted polyp boundary overlaid in red.
The automatic segmentation model flagged this case as uncertain.

Different cases require different answers -- do not default to the same answer every time.
Here are three worked examples showing how the correct answer depends on what's actually visible:

Example A (correct answer: shrink): The red outline includes a strip of normal pink mucosa
along its lower edge that has the same smooth texture and color as the surrounding healthy
tissue, clearly outside the raised, differently-textured polyp itself. -> "shrink"

Example B (correct answer: expand): The polyp's raised, irregular-textured tissue visibly
continues past the red line on the right side, into an area the boundary didn't cover. -> "expand"

Example C (correct answer: no_change): The red outline closely hugs the visible edge of the
polyp on all sides, with no strip of normal tissue included and no polyp tissue left outside
the line. -> "no_change"

Now look at THIS specific image on its own merits and decide which of these three situations
(or "shift_boundary" if the boundary is roughly right-sized but offset to one side) it actually
matches. Base your answer only on what you observe in this image, not on a habitual default.

Respond with ONLY a single JSON object, no other text before or after it, in exactly this shape:
{"reasoning": "<describe the specific visual evidence in THIS image that supports your answer>", "boundary_adjustment": "<one of: expand, shrink, shift_boundary, no_change>", "adjustment_region": "<brief location, e.g. 'left edge', 'entire boundary'>", "confidence": <number between 0 and 1>}"""

NIM = "https://integrate.api.nvidia.com/v1/chat/completions"
OLLAMA = "http://localhost:11434/v1/chat/completions"

CONFIGS = {
    "llama11b_p1_nim":    {"endpoint": NIM,    "model": "meta/llama-3.2-11b-vision-instruct", "prompt": P1, "key": True,  "params_b": 11},
    "llama11b_p1_ollama": {"endpoint": OLLAMA, "model": "llama3.2-vision:11b",                "prompt": P1, "key": False, "params_b": 11},
    "qwen7b_p3":          {"endpoint": OLLAMA, "model": "qwen2.5vl:7b",                       "prompt": P3, "key": False, "params_b": 7},
    "qwen7b_p1":          {"endpoint": OLLAMA, "model": "qwen2.5vl:7b",                       "prompt": P1, "key": False, "params_b": 7},
    "medgemma4b_p1":      {"endpoint": OLLAMA, "model": "medgemma:4b",                        "prompt": P1, "key": False, "params_b": 4},
    "medgemma4b_p3":      {"endpoint": OLLAMA, "model": "medgemma:4b",                        "prompt": P3, "key": False, "params_b": 4},
}
VALID = {"expand", "shrink", "shift_boundary", "no_change"}


def ask_vlm(cfg_name: str, overlay_path: str, max_retries: int = 3, timeout: int = 300) -> dict:
    """Returns dict(adjustment, reasoning, vlm_confidence, raw, latency_s, prompt_tokens, completion_tokens)."""
    if cfg_name == "mock":  # offline dry-run: deterministic pseudo-random proposal, no network
        import hashlib
        h = int(hashlib.md5(overlay_path.encode()).hexdigest(), 16)
        adj = ["expand", "shrink", "shift_boundary", "no_change"][h % 4]
        return {"adjustment": adj, "reasoning": "mock", "vlm_confidence": 0.5, "raw": "", "latency_s": 0.0,
                "prompt_tokens": None, "completion_tokens": None}
    c = CONFIGS[cfg_name]
    b64 =base64.b64encode(open(overlay_path, "rb").read()).decode()
    payload = {
        "model": c["model"], "temperature": 0.0,
        "messages": [
            {"role": "system", "content": c["prompt"]},
            {"role": "user", "content": [
                {"type": "text", "text": "Review this polyp boundary and respond with the JSON object."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if c["key"]:
        headers["Authorization"] = f"Bearer {os.environ['NVIDIA_NIM_API_KEY']}"
    last = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(c["endpoint"], headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            js = r.json(); content = js["choices"][0]["message"]["content"]
            lat = time.perf_counter() - t0
            p = _extract_json_object(content)
            adj = str(p.get("boundary_adjustment", "no_change")).strip().lower()
            if adj not in VALID:
                adj = "no_change"
            u = js.get("usage", {}) or {}
            return {"adjustment": adj, "reasoning": p.get("reasoning", ""),
                    "vlm_confidence": float(p.get("confidence", float("nan")) or float("nan")),
                    "raw": content, "latency_s": lat,
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"VLM {cfg_name} failed: {last}")
