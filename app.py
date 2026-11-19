#!/usr/bin/env python3
"""
Signature Analyzer - AI Agent
Upload Signature -> Call Vision LLM (OpenAI-compatible API) for scoring & classification.
Retains OpenCV heuristics as fallback / supplementary features.
"""

from __future__ import annotations

import os
import base64
import io
import math
import json
import re
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


SCORING_PROMPT = """
You are a signature analysis expert. Analyze the signature image and return ONE JSON object only.

SCORING (each 0-100):
- stroke_complexity (45%): turns, curves, stroke segments
- character_structure_clarity (25%): readable letters or Chinese characters
- connection_overlap (25%): joins, crossings, continuous strokes
- decorative_elements (5%): flourishes, tails, ornament

Overall score = round(0.45*stroke + 0.25*clarity + 0.25*connection + 0.05*decorative)
classification from score:
- low_complexity: score <= 60
- medium_complexity: 61-80
- high_complexity: score >= 81

signature_type (strict):
- low_complexity => MUST be "unknown" (never english/chinese for scribbles)
- chinese_signature: readable Chinese characters (繁體 or 简体)
- english_signature: readable Latin letters
- mixed_or_unclear / unknown otherwise

REASONS: exactly 3 short plain-language bullets.
- high/medium Chinese: try to read the characters, e.g. "Likely reads as 李美娟"
- high/medium English: try to read the name, e.g. "Reads as Ayman"
- low: say it is a simple mark; do not invent letters/characters

OUTPUT: raw JSON only. First char { last char }. No markdown, no code fences, no thinking text.

{
  "classification": "high_complexity",
  "signature_type": "chinese_signature",
  "complexity_score": 86,
  "component_scores": {
    "stroke_complexity": 88,
    "character_structure_clarity": 85,
    "connection_overlap": 82,
    "decorative_elements": 65
  },
  "score_level": "High Complexity",
  "score_description": "Clear multi-character Chinese signature with connected brush strokes.",
  "reasons": [
    "Likely reads as 李美娟",
    "Three separate characters in a row with connected strokes",
    "Last character ends with a long downward stroke"
  ],
  "visual_observations": "Black-ink Chinese signature on white, likely 李美娟, fluid brush-style cursive."
}
"""


def _clamp_int(val, default=50) -> int:
    try:
        return max(0, min(100, int(round(float(val)))))
    except (ValueError, TypeError):
        return default


def _classify_from_score(score: int):
    """Enforce consistent classification: low <= 60, medium 61-80, high >= 81."""
    if score <= 60:
        return ("low_complexity", "Low Complexity", "Low Complexity")
    elif score <= 80:
        return ("medium_complexity", "Medium Complexity", "Medium Complexity")
    else:
        return ("high_complexity", "High Complexity", "High Complexity")


def load_signature_rgb(image_bytes: bytes) -> Image.Image:
    """
    Robust loader for PNG/JPG/WEBP (incl. transparent / dark backgrounds).
    Composites alpha onto pure white and auto-inverts light-on-dark ink.
    """
    pil = Image.open(io.BytesIO(image_bytes))

    if pil.mode in ("P", "LA"):
        pil = pil.convert("RGBA")

    if pil.mode == "RGBA":
        background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, pil)
        pil = composited.convert("RGB")
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")

    arr = np.array(pil)
    gray = np.mean(arr, axis=2)

    # Very dark overall canvas -> invert
    if float(np.mean(gray)) < 80:
        pil = ImageOps.invert(pil)
        arr = np.array(pil)
        gray = np.mean(arr, axis=2)

    flat = gray.flatten()
    if flat.size > 100:
        lo = float(np.percentile(flat, 10))
        hi = float(np.percentile(flat, 90))
        if lo > 100 and hi > 200 and float(np.mean(gray)) < 180:
            pil = ImageOps.invert(pil)

    return pil


def pil_to_cv_bgr(pil: Image.Image) -> np.ndarray:
    rgb = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)



def _clean_observation_text(text: str) -> str:
    """Turn raw model text into a short human-readable observation (never dump JSON)."""
    if not text:
        return ""
    t = text.strip()
    # Strip code fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # If it looks like JSON, try to pull visual_observations field only
    if "{" in t and "visual_observations" in t:
        m = re.search(r'"visual_observations"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
        if m:
            return bytes(m.group(1), "utf-8").decode("unicode_escape")
        # Don't show raw JSON blob to users
        return "Model returned unstructured output; visual detail unavailable."
    # Collapse whitespace / truncate
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 400:
        t = t[:400].rstrip() + "…"
    return t


def _repair_json_string(s: str) -> str:
    """Best-effort repair for truncated / slightly invalid JSON objects."""
    s = s.strip()
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # If truncated mid-string, close the string
    # Count quotes (naive)
    if s.count('"') % 2 == 1:
        s += '"'
    # Balance braces / brackets
    open_curly = s.count("{") - s.count("}")
    open_square = s.count("[") - s.count("]")
    if open_square > 0:
        s += "]" * open_square
    if open_curly > 0:
        s += "}" * open_curly
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _extract_json_object(content: str):
    """Extract and parse the first JSON object from model content. Returns dict or None."""
    if not content:
        return None
    text = content.strip()

    # 1) Fenced ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = []
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    # 2) Substring from first { to last }
    for cand in list(candidates):
        s = cand.find("{")
        e = cand.rfind("}")
        if s != -1 and e != -1 and e > s:
            candidates.append(cand[s : e + 1])
        elif s != -1:
            # truncated — take from first brace to end
            candidates.append(cand[s:])

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        for variant in (cand, _repair_json_string(cand)):
            try:
                obj = json.loads(variant)
                if isinstance(obj, dict) and (
                    "complexity_score" in obj
                    or "component_scores" in obj
                    or "classification" in obj
                    or "signature_type" in obj
                ):
                    return obj
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return None


def call_vision_llm(image_bytes: bytes, api_key: str = None, base_url: str = None, model: str = None) -> dict:
    """Call Vision API via httpx after normalizing the image."""
    import httpx

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("API Key is missing. Set OPENAI_API_KEY in .env or pass api_key.")

    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    url_base = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL or "").rstrip("/")
    endpoint = f"{url_base}/chat/completions"

    pil = load_signature_rgb(image_bytes)
    max_side = 1024
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You output valid JSON only. No markdown, no code fences, "
                    "no explanations, no chain-of-thought. Reply starts with { and ends with }."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCORING_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    if "x.ai" in url_base or "grok" in (model or "").lower():
        payload["thinking"] = {"type": "disabled"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(90.0, connect=25.0), follow_redirects=True) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            # Some providers reject response_format — retry once without it
            if resp.status_code in (400, 404, 422) and "response_format" in payload:
                payload.pop("response_format", None)
                resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            api_data = resp.json()
    except httpx.ConnectError as e:
        raise RuntimeError(f"ConnectError: Unable to reach {url_base}. ({str(e)[:200]})") from e
    except httpx.TimeoutException as e:
        raise RuntimeError(f"Request timed out: {str(e)[:200]}") from e
    except httpx.HTTPStatusError as e:
        body = e.response.text[:400] if e.response is not None else ""
        raise RuntimeError(f"API returned HTTP {e.response.status_code}: {body}") from e

    msg = api_data["choices"][0]["message"]
    content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not content:
        raise RuntimeError(f"Model returned empty content: {str(api_data)[:400]}")

    data = _extract_json_object(content)
    if data is None:
        # Salvage individual fields with regex before giving up
        def _grab_int(key, default=None):
            m = re.search(rf'"{key}"\s*:\s*(\d{{1,3}})', content)
            if m:
                return max(0, min(100, int(m.group(1))))
            return default

        def _grab_str(key, default=""):
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
            return m.group(1) if m else default

        stroke = _grab_int("stroke_complexity")
        clarity = _grab_int("character_structure_clarity")
        connection = _grab_int("connection_overlap")
        decorative = _grab_int("decorative_elements")
        overall = _grab_int("complexity_score")
        sig_type = _grab_str("signature_type", "unknown") or "unknown"
        classification = _grab_str("classification", "")
        score_level = _grab_str("score_level", "")
        score_desc = _grab_str("score_description", "") or _grab_str("score_desc", "")
        visual = _grab_str("visual_observations", "") or _clean_observation_text(content)

        salvaged_any = any(v is not None for v in (stroke, clarity, connection, decorative, overall))
        data = {
            "signature_type": sig_type,
            "classification": classification or None,
            "complexity_score": overall if overall is not None else 50,
            "confidence": 0.55 if salvaged_any else 0.35,
            "component_scores": {
                "stroke_complexity": stroke if stroke is not None else 50,
                "character_structure_clarity": clarity if clarity is not None else 50,
                "connection_overlap": connection if connection is not None else 50,
                "decorative_elements": decorative if decorative is not None else 50,
            },
            "score_level": score_level,
            "score_description": score_desc,
            "reasons": [
                "Partially recovered fields from incomplete model JSON."
                if salvaged_any else
                "Model output was not valid JSON; scores defaulted. Retry or switch model."
            ],
            "visual_observations": visual,
            "_parse_failed": True,
        }

    raw_comp = data.get("component_scores") or {}
    fallback_val = _clamp_int(data.get("complexity_score", 50))
    stroke = _clamp_int(raw_comp.get("stroke_complexity"), fallback_val)
    clarity = _clamp_int(raw_comp.get("character_structure_clarity"), fallback_val)
    connection = _clamp_int(raw_comp.get("connection_overlap"), fallback_val)
    decorative = _clamp_int(raw_comp.get("decorative_elements"), fallback_val)

    weighted_score = 0.45 * stroke + 0.25 * clarity + 0.25 * connection + 0.05 * decorative
    score = max(0, min(100, int(round(weighted_score))))
    cls, cls_label, default_level = _classify_from_score(score)

    type_map = {
        "english_signature": "English Signature",
        "chinese_signature": "Chinese Signature",
        "mixed_or_unclear": "Mixed / Unclear",
        "unknown": "Unknown",
    }
    sig_type = data.get("signature_type", "unknown")
    if sig_type not in type_map:
        sig_type = "unknown"

    # Low complexity marks must not claim English/Chinese script type
    if cls == "low_complexity":
        sig_type = "unknown"

    reasons = data.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    # Drop empty reason lines
    reasons = [str(r).strip() for r in reasons if str(r).strip()]
    visual = (data.get("visual_observations") or "").strip()

    # Infer chinese_signature from CJK only for medium/high complexity
    cjk_re = re.compile(r"[\u4e00-\u9fff]")
    blob_for_type = " ".join([
        visual or "",
        " ".join(str(r) for r in reasons),
        str(data.get("score_description") or ""),
    ])
    if (
        cls != "low_complexity"
        and sig_type in ("unknown", "mixed_or_unclear")
        and cjk_re.search(blob_for_type)
    ):
        sig_type = "chinese_signature"

    note = "Generated by Vision LLM"
    if data.get("_parse_failed"):
        note = "Vision LLM returned non-JSON text; used repaired/default fields. Consider a stronger JSON-capable model."

    return {
        "source": "vision_llm",
        "model": model,
        "classification": cls,
        "classification_label": cls_label,
        "signature_type": sig_type,
        "signature_type_label": type_map.get(sig_type, sig_type),
        "confidence": float(data.get("confidence", 0.85 if not data.get("_parse_failed") else 0.4)),
        "complexity_score": score,
        "component_scores": {
            "stroke_complexity": stroke,
            "character_structure_clarity": clarity,
            "connection_overlap": connection,
            "decorative_elements": decorative,
        },
        "score_level": data.get("score_level") or default_level,
        "score_description": data.get("score_description", ""),
        "reasons": reasons,
        "visual_observations": visual if visual and not visual.strip().startswith("{") else _clean_observation_text(visual or ""),
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "note": note,
    }


def analyze_signature_cv(image_bytes: bytes) -> dict:
    """OpenCV heuristic fallback with improved geometry scoring and richer reasons."""
    pil = load_signature_rgb(image_bytes)
    img = pil_to_cv_bgr(pil)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    fg_pixels = int(np.sum(binary > 0))
    total_pixels = h * w
    ink_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(15, (h * w) * 0.00003)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    num_contours = len(valid_contours)

    total_perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)
    total_area = sum(cv2.contourArea(c) for c in valid_contours)

    if valid_contours:
        all_pts = np.vstack(valid_contours)
        _, _, bw, bh = cv2.boundingRect(all_pts)
        bbox_area = max(1, bw * bh)
        fill_ratio = total_area / bbox_area
        aspect = bw / bh if bh > 0 else 1.0
        span_ratio = bw / max(1, w)
    else:
        fill_ratio = 0
        aspect = 1.0
        span_ratio = 0
        bw = bh = 0

    if total_area > 0:
        circularity = (4 * math.pi * total_area) / (total_perimeter ** 2 + 1e-6)
        tortuosity = max(0.0, 1.0 - min(1.0, circularity))
    else:
        tortuosity = 0.0

    poly_vertices = 0
    for c in valid_contours:
        epsilon = 0.008 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        poly_vertices += len(approx)

    internal = 0
    if hierarchy is not None and len(hierarchy) > 0:
        for i, hinfo in enumerate(hierarchy[0]):
            if hinfo[3] >= 0 and cv2.contourArea(contours[i]) >= min_area:
                internal += 1

    stroke = _clamp_int(min(100, poly_vertices * 1.35 + num_contours * 6 + tortuosity * 25))
    clarity = _clamp_int(min(100, num_contours * 12 + internal * 8 + fill_ratio * 55 + span_ratio * 25))
    connection = _clamp_int(min(100, tortuosity * 70 + min(30, total_perimeter / max(1, (bw + bh)) * 8)))
    decorative = _clamp_int(
        min(100, abs(aspect - 2.2) * 14 + tortuosity * 35 + max(0, (total_perimeter / (total_area + 1)) - 0.4) * 20)
    )

    if num_contours >= 3 and span_ratio > 0.45:
        stroke = min(100, stroke + 8)
        clarity = min(100, clarity + 10)
    if poly_vertices >= 40:
        stroke = min(100, stroke + 6)

    weighted_score = 0.45 * stroke + 0.25 * clarity + 0.25 * connection + 0.05 * decorative
    score = max(0, min(100, int(round(weighted_score))))
    cls, cls_label, default_level = _classify_from_score(score)

    sig_type = "unknown"
    sig_type_label = "Unknown (CV Fallback)"
    # Only guess script type for medium/high complexity geometry
    if score > 60:
        if num_contours >= 2 and aspect > 1.6 and span_ratio > 0.5:
            if fill_ratio > 0.12 and internal >= 1:
                sig_type = "mixed_or_unclear"
                sig_type_label = "Mixed / Unclear (CV estimate)"
        elif num_contours == 1 and tortuosity > 0.55 and aspect > 2.0:
            sig_type = "english_signature"
            sig_type_label = "English Signature (CV estimate)"

    reasons = [
        f"Detected {num_contours} stroke region(s), {poly_vertices} contour vertices, ink ratio {ink_ratio:.3f}.",
    ]
    if internal:
        reasons.append(f"Found {internal} internal hole/structure region(s) suggesting character complexity.")
    if tortuosity > 0.5:
        reasons.append("High stroke tortuosity indicates curved / connected writing paths.")
    if span_ratio > 0.5:
        reasons.append("Wide horizontal span consistent with a multi-character or full-name signature.")
    if score >= 81:
        reasons.append("Geometric density maps into High Complexity under the weighted rubric.")
    elif score <= 60:
        reasons.append("Limited structure — likely a simple mark or sparse strokes under pure geometry.")

    visual = (
        f"OpenCV view: {num_contours} region(s), bbox span {span_ratio:.0%} of width, "
        f"tortuosity {tortuosity:.2f}, fill {fill_ratio:.2f}. "
        f"Dark-ink-on-white preprocessing applied before measurement."
    )

    return {
        "source": "opencv_heuristic",
        "classification": cls,
        "classification_label": cls_label,
        "signature_type": sig_type,
        "signature_type_label": sig_type_label,
        "confidence": 0.55,
        "complexity_score": score,
        "component_scores": {
            "stroke_complexity": stroke,
            "character_structure_clarity": clarity,
            "connection_overlap": connection,
            "decorative_elements": decorative,
        },
        "score_level": default_level,
        "score_description": "Geometric estimate via OpenCV fallback (Vision LLM not used).",
        "reasons": reasons,
        "visual_observations": visual,
        "features": {
            "num_valid_contours": num_contours,
            "ink_ratio": round(ink_ratio, 4),
            "tortuosity": round(tortuosity, 4),
            "poly_vertices": poly_vertices,
            "span_ratio": round(span_ratio, 3),
            "internal_regions": internal,
        },
        "score_breakdown": {
            "formula": "round(0.45*stroke + 0.25*clarity + 0.25*connection + 0.05*decorative)"
        },
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "note": "Generated by OpenCV heuristic fallback — set OPENAI_API_KEY / XAI_API_KEY for Vision LLM.",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files and "image" not in request.form and (
        not request.is_json or "image" not in (request.json or {})
    ):
        return jsonify({"error": "Please upload a signature image."}), 400

    image_bytes = None
    filename = "signature.png"
    api_key = None
    base_url = None
    model = None
    force_cv = False

    if request.is_json:
        data = request.json or {}
        api_key = data.get("api_key")
        base_url = data.get("base_url")
        model = data.get("model")
        force_cv = data.get("force_cv", False)
        if "image" in data:
            raw = data["image"]
            if isinstance(raw, str) and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                return jsonify({"error": "Invalid base64 image."}), 400
    else:
        api_key = request.form.get("api_key") or request.headers.get("X-API-Key")
        base_url = request.form.get("base_url")
        model = request.form.get("model")
        force_cv = request.form.get("force_cv", "").lower() in ("1", "true", "yes")

        if "file" in request.files:
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"error": "No file selected."}), 400
            filename = f.filename
            image_bytes = f.read()
        elif "image" in request.form:
            raw = request.form["image"]
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                return jsonify({"error": "Invalid base64 image."}), 400

    if not image_bytes:
        return jsonify({"error": "Unable to read image."}), 400

    try:
        safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        with open(os.path.join(UPLOAD_FOLDER, safe_name), "wb") as out:
            out.write(image_bytes)
    except Exception:
        pass

    try:
        env_key = os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
        use_llm = not force_cv and bool(api_key or env_key)

        if use_llm:
            try:
                result = call_vision_llm(
                    image_bytes, api_key=api_key, base_url=base_url, model=model
                )
                try:
                    cv_result = analyze_signature_cv(image_bytes)
                    result["cv_features"] = cv_result.get("features")
                except Exception:
                    pass
            except Exception as e:
                llm_error = str(e)
                result = analyze_signature_cv(image_bytes)
                result["llm_error"] = llm_error
                result["note"] = f"Vision LLM failed ({llm_error[:120]}); used OpenCV fallback."
        else:
            result = analyze_signature_cv(image_bytes)
            if not env_key and not api_key:
                result["note"] = (
                    "No API key configured (OPENAI_API_KEY / XAI_API_KEY). "
                    "Using OpenCV geometric fallback only."
                )

        try:
            pil = load_signature_rgb(image_bytes)
            max_side = 480
            w, h = pil.size
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            result["preview_base64"] = (
                "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            )
        except Exception:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/health")
def health():
    has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY"))
    return jsonify({
        "status": "ok",
        "service": "signature-analyzer",
        "vision_llm_ready": has_key,
        "default_model": DEFAULT_MODEL,
        "default_base_url": DEFAULT_BASE_URL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
