"""
Grounding-based refinement using nvidia/LocateAnything-3B via NVIDIA NIM.

Unlike llm_interface.py's categorical (expand/shrink/shift/no_change) approach,
this asks a dedicated VISUAL GROUNDING model -- trained specifically on
localization (bounding boxes, points) across 138M+ queries -- to directly
localize the polyp, rather than asking a general chat VLM to introspect and
describe a boundary correction in prose. This targets the root cause found in
the categorical approach: general chat VLMs showed templated, prompt-derived
text rather than genuine per-image spatial grounding.

IMPORTANT: I could not confirm LocateAnything-3B's exact NIM request/response
contract from documentation alone. This module assumes it is served through
the same OpenAI-compatible chat-completions endpoint as other NIM VLMs (a
reasonable assumption given its Qwen2.5-3B-Instruct backbone), and logs every
raw response to a debug file so the ACTUAL output format can be inspected and
the parser adjusted if the initial assumption is wrong -- same debugging
discipline used throughout this project rather than guessing blind.
"""
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
GROUNDING_MODEL_NAME = "nvidia/locateanything-3b"  # NIM catalog slug -- verify on build.nvidia.com if this 404s

GROUNDING_PROMPT = """Locate the polyp (abnormal raised tissue) in this colonoscopy image.
Respond with ONLY a JSON object giving its bounding box in pixel coordinates for this
{width}x{height} image, in exactly this shape:
{{"box": [x1, y1, x2, y2], "confidence": <number between 0 and 1>}}
where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner of the box
tightly containing the polyp."""


@dataclass
class GroundingResult:
    image_id: str
    box: tuple  # (x1, y1, x2, y2) in pixel coords
    confidence: float
    raw_response: str


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_box_from_text(text: str):
    """Tries several plausible output formats, in order, logging which one
    matched -- since the exact format LocateAnything uses via NIM chat
    completions was not confirmed ahead of time.
    1. JSON object with a "box" key (what we explicitly prompted for)
    2. Four bare numbers anywhere in the text (fallback if it ignores the
       requested JSON wrapper and just states coordinates)
    """
    # attempt 1: JSON object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(text[start : end + 1])
            if "box" in obj and len(obj["box"]) == 4:
                return tuple(float(v) for v in obj["box"]), float(obj.get("confidence", 0.5)), "json"
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # attempt 2: four bare numbers (e.g. "the box is [120, 45, 310, 280]" or "120,45,310,280")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if len(numbers) >= 4:
        box = tuple(float(n) for n in numbers[:4])
        return box, 0.5, "regex_fallback"

    return None, None, "unparsed"


def request_grounding(
    image_path: str,
    image_id: str,
    image_width: int,
    image_height: int,
    api_key: str,
    max_retries: int = 2,
) -> Optional[GroundingResult]:
    b64_image = _encode_image_b64(image_path)
    prompt = GROUNDING_PROMPT.format(width=image_width, height=image_height)

    payload = {
        "model": GROUNDING_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            },
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # ALWAYS log raw response -- we don't yet know the true output format
            with open("debug_grounding_responses.jsonl", "a", encoding="utf-8") as dbg:
                dbg.write(json.dumps({"image_id": image_id, "raw_content": content}) + "\n")

            box, confidence, method = _parse_box_from_text(content)
            if box is None:
                raise ValueError(f"Could not parse a bounding box from response: {content[:300]}")

            return GroundingResult(image_id=image_id, box=box, confidence=confidence, raw_response=content)

        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
            continue

    # log failure explicitly rather than crashing the whole pipeline run
    with open("debug_grounding_responses.jsonl", "a", encoding="utf-8") as dbg:
        dbg.write(json.dumps({"image_id": image_id, "error": str(last_error)}) + "\n")
    return None


def apply_grounding_box(backbone_mask: np.ndarray, box: tuple) -> np.ndarray:
    """
    Combines the backbone's organic mask SHAPE with LocateAnything's precise
    box LOCATION/SCALE: crops the backbone mask to its own tight bounding box
    (preserving its shape), then resizes and pastes that shape into the
    grounding model's predicted box location -- rather than using the raw
    rectangle directly, which would be a crude approximation of an organic
    polyp contour.

    If the backbone mask is empty (no predicted pixels at all), falls back to
    a filled rectangle at the grounding box, since there's no shape to reuse.
    """
    from scipy.ndimage import zoom

    h, w = backbone_mask.shape
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
    y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
    box_w, box_h = max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))

    ys, xs = np.where(backbone_mask > 0.5)
    if len(xs) == 0:
        # no existing shape to reuse -- fall back to a filled rectangle
        new_mask = np.zeros_like(backbone_mask)
        new_mask[int(y1) : int(y1) + box_h, int(x1) : int(x1) + box_w] = 1.0
        return new_mask

    bx1, bx2, by1, by2 = xs.min(), xs.max(), ys.min(), ys.max()
    cropped_shape = backbone_mask[by1 : by2 + 1, bx1 : bx2 + 1]

    src_h, src_w = cropped_shape.shape
    zoom_y, zoom_x = box_h / max(src_h, 1), box_w / max(src_w, 1)
    resized_shape = zoom(cropped_shape, (zoom_y, zoom_x), order=0)  # nearest-neighbor, keeps binary

    new_mask = np.zeros_like(backbone_mask)
    paste_y2 = min(h, int(y1) + resized_shape.shape[0])
    paste_x2 = min(w, int(x1) + resized_shape.shape[1])
    new_mask[int(y1) : paste_y2, int(x1) : paste_x2] = resized_shape[
        : paste_y2 - int(y1), : paste_x2 - int(x1)
    ]
    return (new_mask > 0.5).astype(np.float32)