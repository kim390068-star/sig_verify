#!/usr/bin/env python3
"""
Signature Analyzer - AI Agent
上傳電子簽名 → 呼叫 Vision LLM (OpenAI 兼容 API) 進行評分與分類
同時保留 OpenCV 啟發式作為 fallback / 補充特徵。
"""

import os
import base64
import io
import math
import json
import re
from datetime import datetime

# 自動載入 .env 檔案（放在專案根目錄）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # 沒裝 python-dotenv 也沒關係，仍可用系統環境變數

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# OpenAI 兼容客戶端（支援 OpenAI / xAI / Azure / 本地 vLLM 等）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

# ---------------------------------------------------------------------------
# 配置：優先從 .env / 環境變數讀取，也可在前端 / API 請求中暫時傳入
# ---------------------------------------------------------------------------
# 請編輯專案根目錄的 .env 檔案（可參考 .env.example）
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


# ---------------------------------------------------------------------------
# 打分標準（給 LLM 的 prompt 與文件一致）
# ---------------------------------------------------------------------------
SCORING_PROMPT = """
You are a professional signature analysis assistant.

Analyze the provided signature image and evaluate its signature complexity and signature type.

========================================
OBJECTIVE
========================================

1. Calculate a signature complexity score (0-100).
2. Classify the signature into one of three complexity levels:
   - low_complexity
   - medium_complexity
   - high_complexity
3. For medium_complexity and high_complexity signatures, determine whether the signature is primarily:
   - english_signature
   - chinese_signature
   - mixed_or_unclear

========================================
COMPLEXITY SCORING FRAMEWORK
========================================

Evaluate the signature using the following four dimensions.

Each dimension must be scored from 0 to 100.

----------------------------------------
1. Stroke Complexity (40%)
----------------------------------------

Definition:
The overall amount of visible pen movement, stroke variation, and directional changes.

Scoring Guide:

0   = single line or minimal mark
25  = very simple signature with few strokes
50  = average signature complexity
75  = numerous strokes and directional changes
100 = highly intricate, dense, and multi-stroke signature

----------------------------------------
2. Character Structure Clarity (25%)
----------------------------------------

Definition:
The degree to which recognizable character structures are present.

This includes:
- identifiable letters
- identifiable Chinese characters
- consistent writing patterns

Scoring Guide:

0   = no recognizable character structure
25  = mostly abstract strokes
50  = partially recognizable characters
75  = mostly recognizable characters
100 = highly structured and clearly formed characters

----------------------------------------
3. Connection & Overlap (20%)
----------------------------------------

Definition:
The extent of connected writing, intersections, overlaps, and continuous pen flow.

Scoring Guide:

0   = isolated strokes with no overlap
25  = minimal connections
50  = moderate connections
75  = significant overlap and linking
100 = extensive overlap, crossing, and interconnected structure

----------------------------------------
4. Decorative Elements (15%)
----------------------------------------

Definition:
The presence of non-essential stylistic features such as:

- loops
- flourishes
- long tail strokes
- ornamental curves
- artistic embellishments

Scoring Guide:

0   = no decorative elements
25  = minor decorative features
50  = noticeable stylistic elements
75  = substantial embellishment
100 = highly decorative and artistic signature

========================================
COMPLEXITY SCORE FORMULA
========================================

Complexity Score =
round(
(Stroke Complexity × 0.40) +
(Character Structure Clarity × 0.25) +
(Connection & Overlap × 0.20) +
(Decorative Elements × 0.15)
)

========================================
CLASSIFICATION RULES
========================================

low_complexity

Complexity Score: 0-30

Characteristics:
- extremely simple signature
- contains only one or a few strokes
- resembles a line, mark, initial, or very basic shape
- lacks sufficient structure for reliable language identification

----------------------------------------

medium_complexity

Complexity Score: 31-65

Characteristics:
- typical everyday signature
- recognizable writing structure
- moderate amount of personal style
- limited decorative elements

----------------------------------------

high_complexity

Complexity Score: 66-100

Characteristics:
- large number of strokes
- significant overlap or interconnected writing
- rich structural detail
- strong artistic or decorative characteristics

========================================
SIGNATURE TYPE RULES
========================================

If classification is:

low_complexity

Return:

"signature_type": "unknown"

Reason:
Insufficient visual information to reliably determine language characteristics.

----------------------------------------

If classification is:

medium_complexity
or
high_complexity

Select exactly one:

english_signature

Definition:
The signature is primarily composed of English letters, cursive writing, Latin-character forms, or recognizable alphabetic structures.

----------------------------------------

chinese_signature

Definition:
The signature is primarily composed of Chinese character structures, Chinese-style stroke composition, or identifiable Chinese writing patterns.

----------------------------------------

mixed_or_unclear

Definition:
Language characteristics cannot be determined reliably, or both English and Chinese structural features are present.

========================================
CONFIDENCE SCORE
========================================

Return a confidence score between 0 and 1.

Guidelines:

0.90-1.00 = very high confidence
0.75-0.89 = high confidence
0.60-0.74 = moderate confidence
0.40-0.59 = low confidence
below 0.40 = very uncertain

========================================
OUTPUT REQUIREMENTS
========================================

Return ONLY valid JSON.

Do NOT use Markdown.

Do NOT include explanations outside the JSON.

Do NOT include code fences.

Use the exact structure below:

{
  "classification": "medium_complexity",
  "signature_type": "english_signature",
  "complexity_score": 58,
  "confidence": 0.89,
  "component_scores": {
    "stroke_complexity": 65,
    "character_structure_clarity": 60,
    "connection_overlap": 55,
    "decorative_elements": 35
  },
  "score_level": "Medium Complexity",
  "score_description": "Contains recognizable writing structure with moderate complexity and limited decorative features.",
  "reasons": [
    "Recognizable character structure is present",
    "Moderate amount of connected strokes",
    "Limited decorative embellishments"
  ],
  "visual_observations": "The signature contains continuous handwritten strokes with identifiable character-like structures and moderate overlap."
}
"""


def get_client(api_key: str = None, base_url: str = None):
    """相容介面：有 key 就回傳 True"""
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
    return True if key else None


def call_vision_llm(image_bytes: bytes, api_key: str = None, base_url: str = None, model: str = None) -> dict:
    """使用 pure httpx 直接呼叫 Vision API，避開 OpenAI SDK 連線問題"""
    import httpx

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("未提供 API Key。請在 .env 設定 OPENAI_API_KEY，或在前端填入 api_key")

    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    url_base = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL or "").rstrip("/")
    endpoint = f"{url_base}/chat/completions"

    # 壓縮圖片
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_side = 1024
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SCORING_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "thinking": {"type": "disabled"},  # DeepSeek: 關閉思考模式，直接輸出 JSON
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(90.0, connect=25.0), follow_redirects=True) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            api_data = resp.json()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"連線失敗 (ConnectError): 無法連到 {url_base}。"
            f"請確認網路、防火牆或代理。原始錯誤: {str(e)[:200]}"
        ) from e
    except httpx.TimeoutException as e:
        raise RuntimeError(f"請求超時: {str(e)[:200]}") from e
    except httpx.HTTPStatusError as e:
        body = e.response.text[:400] if e.response is not None else ""
        raise RuntimeError(f"API 回傳錯誤 HTTP {e.response.status_code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Vision API 呼叫失敗: {type(e).__name__}: {str(e)[:300]}") from e

    try:
        msg = api_data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        # DeepSeek 有時會把主要輸出放在 reasoning_content
        if not content:
            content = (msg.get("reasoning_content") or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"模型回傳格式異常: {str(api_data)[:400]}") from e

    if not content:
        raise RuntimeError(f"模型回傳空內容。原始 response: {str(api_data)[:400]}")

    # 嘗試抽出 JSON（有些模型會包 markdown 或混入推理文字）
    json_str = content
    if "```" in content:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if match:
            json_str = match.group(1).strip()
    s = json_str.find("{")
    e = json_str.rfind("}")
    if s != -1 and e != -1 and e > s:
        json_str = json_str[s : e + 1]

    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        cls_m = re.search(r'"classification"\s*:\s*"([^"]+)"', content)
        score_m = re.search(r'"complexity_score"\s*:\s*(\d+)', content)
        if cls_m or score_m:
            data = {
                "classification": cls_m.group(1) if cls_m else "single_signature",
                "complexity_score": int(score_m.group(1)) if score_m else 50,
                "confidence": 0.6,
                "reasons": ["從非完整 JSON 中抽取欄位"],
                "visual_observations": content[:300],
            }
        else:
            data = {
                "classification": "single_signature",
                "complexity_score": 50,
                "confidence": 0.4,
                "reasons": ["LLM 回傳格式異常，已使用預設值", content[:180]],
                "score_level": "中等複雜度",
                "score_description": "無法完整解析模型輸出",
                "visual_observations": content[:300],
                "raw_response": content,
            }

    # 正規化欄位
    label_map = {
        "single_signature": "單一簽名 (Single Signature)",
        "multiple_signatures": "多重簽名 (Multiple Signatures)",
        "complex_flourish": "華麗花飾簽名 (Complex Flourish)",
        "simple_print": "簡約/印刷體 (Simple / Print-like)",
        "illegible_abstract": "難以辨認 / 抽象 (Illegible / Abstract)",
        "stamp_like": "印章風格 (Stamp-like)",
    }
    cls = data.get("classification", "single_signature")
    score = int(data.get("complexity_score", 50))
    score = max(0, min(100, score))

    if not data.get("score_level"):
        if score < 25:
            data["score_level"] = "極低複雜度"
        elif score < 45:
            data["score_level"] = "低複雜度"
        elif score < 65:
            data["score_level"] = "中等複雜度"
        elif score < 85:
            data["score_level"] = "高複雜度"
        else:
            data["score_level"] = "極高複雜度"

    result = {
        "source": "vision_llm",
        "model": model,
        "classification": cls,
        "classification_label": label_map.get(cls, cls),
        "confidence": float(data.get("confidence", 0.8)),
        "complexity_score": score,
        "score_level": data.get("score_level", ""),
        "score_description": data.get("score_description", ""),
        "reasons": data.get("reasons", []),
        "visual_observations": data.get("visual_observations", ""),
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "note": "結果由 Vision LLM 生成",
    }
    return result


def analyze_signature_cv(image_bytes: bytes) -> dict:
    """OpenCV 啟發式 fallback / 補充特徵（保留原邏輯）"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    original_size = {"width": w, "height": h}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 127:
        gray = 255 - gray
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

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20, (h * w) * 0.00005)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    num_contours = len(valid_contours)

    total_perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)
    total_area = sum(cv2.contourArea(c) for c in valid_contours)

    if valid_contours:
        all_pts = np.vstack(valid_contours)
        x, y, bw, bh = cv2.boundingRect(all_pts)
        bbox_area = bw * bh
        fill_ratio = total_area / bbox_area if bbox_area > 0 else 0
        aspect = bw / bh if bh > 0 else 1.0
    else:
        x = y = bw = bh = 0
        fill_ratio = 0
        aspect = 1.0

    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = int(np.sum(edges > 0))
    edge_density = edge_pixels / fg_pixels if fg_pixels > 0 else 0

    if total_area > 0:
        circularity = (4 * math.pi * total_area) / (total_perimeter ** 2 + 1e-6)
        tortuosity = max(0, 1.0 - circularity)
    else:
        tortuosity = 0

    poly_vertices = 0
    for c in valid_contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        poly_vertices += len(approx)

    contour_score = min(30.0, 12.0 * math.log2(1 + num_contours))
    edge_score = min(25.0, edge_density * 70.0)
    tort_score = min(25.0, tortuosity * 35.0)
    stroke_score = min(20.0, 6.0 * math.log2(1 + poly_vertices / max(1, num_contours)))
    complexity_score = round(min(100.0, contour_score + edge_score + tort_score + stroke_score), 1)

    reasons = []
    classification = "single_signature"
    confidence = 0.7

    if num_contours == 0 or fg_pixels < 50:
        classification = "illegible_abstract"
        confidence = 0.9
        reasons.append("幾乎偵測不到有效筆跡")
    elif num_contours >= 4 and fill_ratio < 0.25:
        classification = "multiple_signatures"
        confidence = 0.75
        reasons.append(f"偵測到 {num_contours} 個相對獨立的筆跡區域")
    elif tortuosity > 0.55 and poly_vertices / max(1, num_contours) > 12:
        classification = "complex_flourish"
        confidence = 0.8
        reasons.append("線條曲折度高且多邊形頂點多，呈現華麗花飾特徵")
    elif num_contours <= 2 and tortuosity < 0.25 and poly_vertices < 15:
        classification = "simple_print"
        confidence = 0.7
        reasons.append("筆畫極簡、曲折度低，接近印刷或極簡風格")
    elif fill_ratio > 0.45 and edge_density < 0.35:
        classification = "stamp_like"
        confidence = 0.65
        reasons.append("高填充率與相對封閉區域，接近印章風格")
    else:
        classification = "single_signature"
        confidence = 0.75
        reasons.append("單一主要連貫筆跡區域，符合一般簽名特徵")

    label_map = {
        "single_signature": "單一簽名 (Single Signature)",
        "multiple_signatures": "多重簽名 (Multiple Signatures)",
        "complex_flourish": "華麗花飾簽名 (Complex Flourish)",
        "simple_print": "簡約/印刷體 (Simple / Print-like)",
        "illegible_abstract": "難以辨認 / 抽象 (Illegible / Abstract)",
        "stamp_like": "印章風格 (Stamp-like)",
    }

    if complexity_score < 25:
        score_level, score_desc = "極低複雜度", "筆畫非常簡單，幾乎直線或少數曲線"
    elif complexity_score < 45:
        score_level, score_desc = "低複雜度", "一般日常簽名，線條相對順暢"
    elif complexity_score < 65:
        score_level, score_desc = "中等複雜度", "具有一定變化與個人特色的簽名"
    elif complexity_score < 85:
        score_level, score_desc = "高複雜度", "華麗或筆畫豐富，具備明顯花飾或多重迴圈"
    else:
        score_level, score_desc = "極高複雜度", "高度裝飾性，線條極度曲折或密集"

    return {
        "source": "opencv_heuristic",
        "classification": classification,
        "classification_label": label_map.get(classification, classification),
        "confidence": round(confidence, 2),
        "complexity_score": complexity_score,
        "score_level": score_level,
        "score_description": score_desc,
        "reasons": reasons,
        "features": {
            "num_valid_contours": num_contours,
            "ink_ratio": round(ink_ratio, 4),
            "edge_density": round(edge_density, 4),
            "tortuosity": round(tortuosity, 4),
            "poly_vertices": poly_vertices,
            "bounding_box": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
            "aspect_ratio": round(aspect, 3),
            "fill_ratio": round(fill_ratio, 4),
            "image_size": original_size,
        },
        "score_breakdown": {
            "contour_score": round(contour_score, 1),
            "edge_density_score": round(edge_score, 1),
            "tortuosity_score": round(tort_score, 1),
            "stroke_complexity_score": round(stroke_score, 1),
            "formula": "min(100, contour(0-30) + edge(0-25) + tortuosity(0-25) + stroke(0-20))",
        },
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "note": "本結果由計算機視覺啟發式規則產生（fallback）",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    支援兩種模式：
    1. 有 API Key → 呼叫真實 Vision LLM
    2. 無 API Key → 使用 OpenCV 啟發式
    """
    if "file" not in request.files and "image" not in request.form and (not request.is_json or "image" not in (request.json or {})):
        return jsonify({"error": "請上傳簽名圖片 (file 欄位或 base64 image)"}), 400

    image_bytes = None
    filename = "signature.png"

    # 可選參數
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
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                return jsonify({"error": "無效的 base64 圖片"}), 400
    else:
        api_key = request.form.get("api_key") or request.headers.get("X-API-Key")
        base_url = request.form.get("base_url")
        model = request.form.get("model")
        force_cv = request.form.get("force_cv", "").lower() in ("1", "true", "yes")

        if "file" in request.files:
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"error": "未選擇檔案"}), 400
            filename = f.filename
            image_bytes = f.read()
        elif "image" in request.form:
            raw = request.form["image"]
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw)
            except Exception:
                return jsonify({"error": "無效的 base64 圖片"}), 400

    if not image_bytes:
        return jsonify({"error": "無法讀取圖片"}), 400

    # 儲存上傳
    try:
        safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        path = os.path.join(UPLOAD_FOLDER, safe_name)
        with open(path, "wb") as out:
            out.write(image_bytes)
    except Exception:
        pass

    try:
        # 優先使用真實 AI
        use_llm = not force_cv and (api_key or os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY"))

        if use_llm:
            result = call_vision_llm(image_bytes, api_key=api_key, base_url=base_url, model=model)
            # 同時附上 CV 特徵作為補充（可選）
            try:
                cv_result = analyze_signature_cv(image_bytes)
                result["cv_features"] = cv_result.get("features")
                result["cv_score_breakdown"] = cv_result.get("score_breakdown")
            except Exception:
                pass
        else:
            result = analyze_signature_cv(image_bytes)

        # 預覽圖
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            max_side = 480
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            result["preview_base64"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"分析失敗: {str(e)}"}), 500


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
    print("=" * 60)
    print("Signature Analyzer (Vision LLM + OpenCV)")
    print("  預設模型:", DEFAULT_MODEL)
    print("  Base URL:", DEFAULT_BASE_URL)
    print("  API Key 已設定:" , bool(os.getenv("OPENAI_API_KEY") or os.getenv("XAI_API_KEY")))
    print("  啟動於 http://0.0.0.0:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
