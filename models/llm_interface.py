"""
MLLM interface for mask refinement on gated-in (hard/ambiguous) cases.

Reuses the same provider pattern as the FaithCAM and CAGE projects: NVIDIA
NIM API. Provider-agnostic client shape kept so a different backend can be
swapped in later without touching run_pipeline.py.

The MLLM is NOT asked to draw a mask directly (unreliable, hard to score
pixel-precisely from free text). Instead it's asked to propose a REGION
adjustment relative to the backbone's mask: which of a small set of candidate
boundary edits best matches what it sees. This keeps the refinement space
small, structured, and easy to causally verify in causal_verify.py.

NOTE: response_format=json_schema was tried initially but found to
malfunction for this model on NIM (returned an identical, non-image-grounded
{'type': 'expand'} stub for every input regardless of image content). Dropped
in favor of strong prompt-based JSON instructions plus a robust parser that
tolerates both strict JSON and single-quoted Python-dict-style output.
"""
import base64
import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

NIM_ENDPOINT = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5vl:7b"  # same model family as FaithCAM project

# SYSTEM_PROMPT = """You are assisting with polyp boundary review in a colonoscopy image.
# You will see the original image with the current predicted polyp boundary overlaid in red.
# The automatic segmentation model flagged this case as uncertain.
# Decide whether the boundary should be expanded, shrunk, shifted, or left unchanged,
# and give a short reasoning grounded in what you actually see in the image
# (color, texture, or edge cues distinguishing polyp tissue from surrounding mucosa).

# Respond with ONLY a single JSON object, no other text before or after it, in exactly this shape:
# {"reasoning": "<one or two sentences explaining what you see and why>", "boundary_adjustment": "<one of: expand, shrink, shift_boundary, no_change>", "adjustment_region": "<brief location, e.g. 'left edge', 'entire boundary'>", "confidence": <number between 0 and 1>}"""
SYSTEM_PROMPT = """You are assisting with polyp boundary review in a colonoscopy image.
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

@dataclass
class RefinementResult:
    image_id: str
    reasoning: str
    boundary_adjustment: str
    adjustment_region: str
    mllm_confidence: float
    raw_response: str


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_object(text: str) -> dict:
    """Brace-counting extraction -- same fix applied in the FaithCAM project
    after finding the model sometimes prepends prose before the JSON object
    despite instructions. Additionally falls back to ast.literal_eval for
    single-quoted, Python-dict-style output -- ast handles embedded
    apostrophes in free-text fields correctly, unlike a naive
    quote-replacement regex would."""
    import ast

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:300]}")
    depth = 0
    end = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ValueError(f"Unbalanced braces in response: {text[:300]}")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # fallback: single-quoted Python-dict-style output
    try:
        normalized = candidate.replace("true", "True").replace("false", "False").replace("null", "None")
        parsed = ast.literal_eval(normalized)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    raise ValueError(f"Could not parse as JSON or Python-dict literal. Raw content: {candidate[:300]}")


def request_refinement(
    overlay_image_path: str,
    image_id: str,
    api_key: str,
    max_retries: int = 2,
) -> RefinementResult:
    """
    overlay_image_path: path to the image with the backbone's mask boundary
                         drawn on it (see run_pipeline.py's draw_overlay()).
    Returns a RefinementResult. Raises on repeated failure after retries --
    caller (run_pipeline.py) decides fallback behavior (e.g. treat as
    'no_change' rather than crash the whole run).
    """
    b64_image = _encode_image_b64(overlay_image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Review this polyp boundary and respond with the JSON object."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            },
        ],
        "temperature": 0.0,  # deterministic, per the non-determinism lesson from the FaithCAM project
    }
    #headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    headers = {"Content-Type": "application/json"}  # no Authorization header needed for local Ollama

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # always log the raw response for debugging, success or failure --
            # cheaper than guessing blind at parsing issues a second time
            with open("debug_mllm_responses.jsonl", "a", encoding="utf-8") as dbg:
                dbg.write(json.dumps({"image_id": image_id, "raw_content": content}) + "\n")

            parsed = _extract_json_object(content)
            try:
                return RefinementResult(
                    image_id=image_id,
                    reasoning=parsed["reasoning"],
                    boundary_adjustment=parsed["boundary_adjustment"],
                    adjustment_region=parsed["adjustment_region"],
                    mllm_confidence=float(parsed["confidence"]),
                    raw_response=content,
                )
            except KeyError as ke:
                raise ValueError(
                    f"Parsed response missing expected key {ke}. "
                    f"Parsed keys were: {list(parsed.keys())}. Raw content: {content[:300]}"
                )
        except Exception as e:  # noqa: BLE001 -- broad on purpose, see retry loop
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))  # 3s, 6s backoff, matches FaithCAM's rate-limit fix
            continue

    raise RuntimeError(
        f"MLLM refinement failed after {max_retries + 1} attempts for {image_id}: {last_error}\n"
        f"(check llm_interface.py's _extract_json_object fallback logic if this recurs)"
    )