import cv2
import numpy as np
import json
import argparse
import os
import time
import base64
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# OCR is disabled by default; set to True if you want OCR features.
ENABLE_OCR = True

# Default backend order: VLM first, CV fallback/supplement.
DEFAULT_BACKEND = os.environ.get("VISION_BACKEND", "auto").strip().lower()  # auto|vlm|cv
DEFAULT_STRICT = os.environ.get("VISION_STRICT", "1").strip() in ("1", "true", "yes", "on")
DEFAULT_MAX_BOXES = int(os.environ.get("VISION_MAX_BOXES", "220"))
DEFAULT_MIN_SCORE = float(os.environ.get("VISION_MIN_SCORE", "0.70"))
DEFAULT_CV_SUPPLEMENT = os.environ.get("VISION_CV_SUPPLEMENT", "0").strip() in ("1", "true", "yes", "on")
DEFAULT_VLM_ENABLED = os.environ.get("VISION_USE_VLM", "1").strip() in ("1", "true", "yes", "on")
DEFAULT_VLM_MODEL = os.environ.get("VISION_VLM_MODEL", "claude-sonnet-4-5-20250929").strip()
DEFAULT_VLM_TIMEOUT = float(os.environ.get("VISION_VLM_TIMEOUT", "10.0"))
DEFAULT_VLM_MAX_ELEMENTS = int(os.environ.get("VISION_VLM_MAX_ELEMENTS", "220"))

VLM_ACTIONABLE_TYPES = {
    "button", "link", "checkbox", "radio", "switch", "toggle", "menuitem", "tab",
    "textfield", "input", "combobox", "dropdown", "slider", "row", "cell",
    "listitem", "treeitem", "iconbutton", "segmentedcontrol", "stepper",
}
VLM_CONTAINER_TYPES = {
    "panel", "window", "group", "section", "background", "layout", "container",
    "toolbar", "sidebar", "scrollarea", "canvas", "other",
}

# -----------------------------
# Optional OCR (Tesseract)
# -----------------------------
def try_ocr_on_crop(crop_bgr):
    """
    Returns recognized text if pytesseract is installed and tesseract is available.
    If not available, returns None.
    """
    if not ENABLE_OCR:
        return None
    try:
        import pytesseract  # pip install pytesseract
    except Exception:
        return None
    # Simple preprocessing for OCR
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # PSM 6 = assume a block of text
    config = "--psm 6"
    try:
        text = pytesseract.image_to_string(thr, config=config)
        text = " ".join(text.split())
        return text if text else None
    except Exception:
        return None

def try_ocr_words(img_bgr):
    """
    Returns list of word boxes from pytesseract (if available).
    Each word item: {text, x, y, w, h, conf}
    """
    if not ENABLE_OCR:
        return []
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    try:
        data = pytesseract.image_to_data(thr, output_type=Output.DICT, config="--psm 6")
    except Exception:
        return []

    words = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    xs = data.get("left", [])
    ys = data.get("top", [])
    ws = data.get("width", [])
    hs = data.get("height", [])

    for i in range(len(texts)):
        text = str(texts[i]).strip()
        if not text:
            continue
        try:
            conf = float(confs[i])
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        words.append({
            "text": text,
            "x": int(xs[i]),
            "y": int(ys[i]),
            "w": int(ws[i]),
            "h": int(hs[i]),
            "conf": float(conf),
        })
    return words

# -----------------------------
# Geometry helpers
# -----------------------------
def area(b):
    return max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])

def iou(a, b):
    xA = max(a["x1"], b["x1"])
    yA = max(a["y1"], b["y1"])
    xB = min(a["x2"], b["x2"])
    yB = min(a["y2"], b["y2"])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    union = area(a) + area(b) - inter + 1e-9
    return inter / union

def nms(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.get("score", 1.0) * (b["area"] + 1), reverse=True)
    kept = []
    for b in boxes:
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept

def contains(outer, inner, pad=0):
    return (outer["x1"] - pad <= inner["x1"] and
            outer["y1"] - pad <= inner["y1"] and
            outer["x2"] + pad >= inner["x2"] and
            outer["y2"] + pad >= inner["y2"])

def prune_contained(boxes, containment_iou=0.95):
    """
    Remove boxes that are almost fully inside a bigger box AND highly overlapping.
    This reduces duplicates like: big panel + slightly smaller panel.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["area"], reverse=True)
    keep = []
    for b in boxes:
        redundant = False
        for k in keep:
            if contains(k, b, pad=2) and iou(k, b) > containment_iou:
                redundant = True
                break
        if not redundant:
            keep.append(b)
    return keep

def clamp_box(b, w, h):
    b["x1"] = int(max(0, min(w - 1, b["x1"])))
    b["y1"] = int(max(0, min(h - 1, b["y1"])))
    b["x2"] = int(max(0, min(w, b["x2"])))
    b["y2"] = int(max(0, min(h, b["y2"])))
    b["w"] = int(b["x2"] - b["x1"])
    b["h"] = int(b["y2"] - b["y1"])
    b["area"] = int(area(b))
    b["cx"] = int(b["x1"] + b["w"] / 2)
    b["cy"] = int(b["y1"] + b["h"] / 2)
    return b

# -----------------------------
# UI box proposals (rectangles/icons/panels)
# -----------------------------
def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _auto_canny(gray: np.ndarray) -> np.ndarray:
    """
    Auto-threshold Canny based on the median intensity.
    More robust across macOS light/dark modes and Retina screenshots.
    """
    # Use a gradient proxy (Laplacian) rather than raw intensity; intensity-based
    # thresholds tend to fail on bright UI with subtle borders.
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    v = float(np.median(np.abs(lap)))
    lower = int(max(10, 0.66 * v))
    upper = int(min(255, max(lower + 10, 1.33 * v)))
    return cv2.Canny(gray, lower, upper)

def _luma_for_ui(img_bgr: np.ndarray) -> np.ndarray:
    """
    Luma optimized for UI: LAB L-channel + CLAHE.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
    except Exception:
        pass
    return l

def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def _border_interior_ratio(mag: np.ndarray, x: int, y: int, w: int, h: int, t: int = 2) -> float:
    """
    Border gradient strength vs interior gradient strength for a rect.
    UI boxes tend to have stronger borders than interior textures.
    """
    H, W = mag.shape[:2]
    x1 = max(0, min(W - 1, int(x)))
    y1 = max(0, min(H - 1, int(y)))
    x2 = max(x1 + 1, min(W, int(x + w)))
    y2 = max(y1 + 1, min(H, int(y + h)))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 6 or bh < 6:
        return 0.0

    t = int(max(1, min(int(t), min(bw, bh) // 4)))

    top = mag[y1 : y1 + t, x1:x2]
    bottom = mag[y2 - t : y2, x1:x2]
    left = mag[y1:y2, x1 : x1 + t]
    right = mag[y1:y2, x2 - t : x2]
    border = np.concatenate([top.reshape(-1), bottom.reshape(-1), left.reshape(-1), right.reshape(-1)], axis=0)
    border_mean = float(np.mean(border)) if border.size else 0.0

    ix1 = x1 + t
    iy1 = y1 + t
    ix2 = x2 - t
    iy2 = y2 - t
    if ix2 <= ix1 or iy2 <= iy1:
        interior_mean = 1e-6
    else:
        interior = mag[iy1:iy2, ix1:ix2]
        interior_mean = float(np.mean(interior)) if interior.size else 1e-6

    return border_mean / (interior_mean + 1e-6)

def detect_ui_rectangles(img_bgr, scale: float = 1.0, min_score: float = DEFAULT_MIN_SCORE):
    """
    Finds rectangle-ish components (buttons, list rows, panels, windows)
    using threshold + morphology + contours.
    """
    h0, w0 = img_bgr.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr

    h, w = img.shape[:2]

    luma = _luma_for_ui(img)
    luma = cv2.GaussianBlur(luma, (5, 5), 0)
    mag = _sobel_mag(luma)
    # Combine an auto-thresholded edge map with a low-threshold map to keep subtle UI borders.
    edges = cv2.bitwise_or(_auto_canny(luma), cv2.Canny(luma, 20, 70))

    # Kernel sizes scaled for Retina/HiDPI screenshots (allow <1.0 for smaller images).
    k_scale = max(0.6, min(w, h) / 1400.0)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(3, round(5 * k_scale))), int(max(3, round(3 * k_scale)))))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1, iterations=2)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(3, round(9 * k_scale))), int(max(3, round(6 * k_scale)))))
    filled = cv2.dilate(closed, k2, iterations=1)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(1, bh)
        a = bw * bh

        # Hard filters (precision-first)
        if a < 350:     # tiny noise
            continue
        if a > 0.85 * (w * h):  # entire screen
            continue
        if bw < 8 or bh < 8:
            continue
        if ar > 60 or ar < 1/60:
            continue

        contour_area = float(cv2.contourArea(c))
        rectangularity = contour_area / (float(a) + 1e-6)
        if rectangularity < 0.45:
            continue

        peri = float(cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) if peri > 0 else None
        approx_n = int(len(approx)) if approx is not None else 0
        if approx_n and approx_n > 12:
            continue

        border_ratio = _border_interior_ratio(mag, x, y, bw, bh, t=max(1, int(round(2 * k_scale))))
        if border_ratio < 1.10 and a > 2500:
            continue

        rectness_score = _clamp01((rectangularity - 0.55) / 0.35)
        border_score = _clamp01((border_ratio - 1.10) / 1.30)
        aspect_score = 1.0 if (0.2 < ar < 5.0) else (0.35 if (0.07 < ar < 15.0) else 0.0)
        size_score = _clamp01((np.log(max(a, 1.0)) - np.log(350.0)) / (np.log(60000.0) - np.log(350.0)))
        approx_score = 1.0 if (4 <= approx_n <= 10) else (0.6 if approx_n else 0.35)

        score = (
            0.38 * rectness_score
            + 0.32 * border_score
            + 0.10 * aspect_score
            + 0.10 * size_score
            + 0.10 * approx_score
        )
        if float(score) < float(min_score):
            continue

        # Map back to original coordinates if scaled
        if scale != 1.0:
            x1 = int(x / scale); y1 = int(y / scale)
            x2 = int((x + bw) / scale); y2 = int((y + bh) / scale)
        else:
            x1, y1, x2, y2 = x, y, x + bw, y + bh

        boxes.append({
            "type": "ui_rect",
            "source": "cv",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": float(score)
        })

    return boxes

# -----------------------------
# Text region proposals (MSER-based)
# -----------------------------
def detect_text_regions(img_bgr):
    """
    Detect likely text regions (good for sidebars, settings lists, labels).
    This gives bounding boxes for text groups, not perfect word-level boxes.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


    # Increase contrast a bit (macOS UI has subtle anti-aliased text)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)

    # MSER finds stable regions (often characters)
    mser = create_mser()
    regions, _ = mser.detectRegions(gray)

    if regions is None:
        return []

    char_boxes = []
    for p in regions:
        x, y, bw, bh = cv2.boundingRect(p.reshape(-1, 1, 2))
        a = bw * bh
        if a < 40 or a > 15000:
            continue
        # Character-ish aspect range
        ar = bw / max(1, bh)
        if ar < 0.1 or ar > 10:
            continue
        char_boxes.append((x, y, bw, bh))

    if not char_boxes:
        return []

    # Make a mask of character regions
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, bw, bh) in char_boxes:
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

    # Group characters into lines/blocks (scale kernels for Retina/HiDPI).
    k_scale = max(0.6, min(w, h) / 1400.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(round(28 * k_scale)), int(round(6 * k_scale))))
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = bw * bh

        if a < 300:  # ignore tiny text bits
            continue
        if bw < 15 or bh < 8:
            continue

        # Require a minimum number of MSER "char" boxes inside, to reduce noise.
        char_count = 0
        for (cx, cy, cw, ch) in char_boxes:
            if cx + cw < x or cx > x + bw:
                continue
            if cy + ch < y or cy > y + bh:
                continue
            char_count += 1
            if char_count >= 6:
                break
        if char_count < 6 and a > 900:
            continue

        # Score: text blocks tend to be wider than tall
        ar = bw / max(1, bh)
        ar_score = 1.0 if ar > 2.0 else (0.6 if ar > 1.2 else 0.35)
        size_score = _clamp01((np.log(max(a, 1.0)) - np.log(300.0)) / (np.log(80000.0) - np.log(300.0)))
        score = 0.65 * ar_score + 0.35 * size_score

        boxes.append({
            "type": "text_region",
            "source": "cv",
            "x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
            "score": float(score)
        })

    return boxes

# -----------------------------
# Box refinement
# -----------------------------
def refine_box_to_content(img_bgr, box, padding=2):
    """
    Refine a bounding box to tightly fit the actual content.
    Uses color difference and edge detection to find the actual content boundaries.
    """
    orig_x1, orig_y1, orig_x2, orig_y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x1, y1, x2, y2 = orig_x1, orig_y1, orig_x2, orig_y2
    h, w = img_bgr.shape[:2]

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return box

    # Extract the region
    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return box

    # Get border color (average of edges)
    top_edge = region[0: min(5, region.shape[0]), :]
    bottom_edge = region[max(0, region.shape[0] - 5):, :]
    left_edge = region[:, 0: min(5, region.shape[1]), :]
    right_edge = region[:, max(0, region.shape[1] - 5):, :]

    border_color = np.concatenate([
        top_edge.reshape(-1, 3),
        bottom_edge.reshape(-1, 3),
        left_edge.reshape(-1, 3),
        right_edge.reshape(-1, 3)
    ]).mean(axis=0)

    # Convert to grayscale for edge detection
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Find content boundaries by looking for non-border colors
    # Scan from top
    new_y1 = 0
    for y in range(region.shape[0]):
        row_colors = region[y, :].mean(axis=0)
        if not np.allclose(row_colors, border_color, atol=20):
            new_y1 = y
            break

    # Scan from bottom
    new_y2 = region.shape[0]
    for y in range(region.shape[0] - 1, -1, -1):
        row_colors = region[y, :].mean(axis=0)
        if not np.allclose(row_colors, border_color, atol=20):
            new_y2 = y + 1
            break

    # Scan from left
    new_x1 = 0
    for x in range(region.shape[1]):
        col_colors = region[:, x, :].mean(axis=0)
        if not np.allclose(col_colors, border_color, atol=20):
            new_x1 = x
            break

    # Scan from right
    new_x2 = region.shape[1]
    for x in range(region.shape[1] - 1, -1, -1):
        col_colors = region[:, x, :].mean(axis=0)
        if not np.allclose(col_colors, border_color, atol=20):
            new_x2 = x + 1
            break

    # Apply refinement with padding
    rx1 = max(0, x1 + new_x1 - padding)
    ry1 = max(0, y1 + new_y1 - padding)
    rx2 = min(w, x1 + new_x2 + padding)
    ry2 = min(h, y1 + new_y2 + padding)

    # Safety: avoid pathological over-shrinks (common with gradients/shadows).
    orig_w = max(1, orig_x2 - orig_x1)
    orig_h = max(1, orig_y2 - orig_y1)
    new_w = max(1, rx2 - rx1)
    new_h = max(1, ry2 - ry1)
    if new_w < orig_w * 0.35 or new_h < orig_h * 0.35 or (new_w * new_h) < 120:
        return box

    box["x1"] = int(rx1)
    box["y1"] = int(ry1)
    box["x2"] = int(rx2)
    box["y2"] = int(ry2)

    # Recalculate derived values
    clamp_box(box, w, h)

    return box


def filter_overlapping_small_boxes(boxes, min_area_ratio=0.3):
    """
    Remove very small boxes that are mostly covered by larger boxes.
    This helps reduce clutter from detecting sub-elements within buttons.
    """
    boxes = sorted(boxes, key=lambda b: b["area"], reverse=True)
    removed = set()
    for i, large_box in enumerate(boxes):
        if i in removed:
            continue
        large_area = large_box["area"]
        for j in range(i + 1, len(boxes)):
            if j in removed:
                continue
            small_box = boxes[j]
            small_area = small_box["area"]
            # Skip if small box is not actually small
            if small_area > large_area * min_area_ratio:
                continue
            # If mostly overlapped/contained, drop the small box
            if iou(large_box, small_box) > 0.5 or contains(large_box, small_box, pad=2):
                removed.add(j)
    return [b for idx, b in enumerate(boxes) if idx not in removed]

def prune_text_regions_inside_rects(boxes, max_area_ratio=6.0, pad=4):
    """
    Remove text_region boxes that are inside a reasonably-sized ui_rect.
    This keeps a single box per UI element (reduces overlap).
    """
    ui_rects = [b for b in boxes if b.get("type") == "ui_rect"]
    keep = []
    for b in boxes:
        if b.get("type") != "text_region":
            keep.append(b)
            continue
        drop = False
        for r in ui_rects:
            if contains(r, b, pad=pad):
                if r["area"] <= b["area"] * max_area_ratio:
                    drop = True
                    break
        if not drop:
            keep.append(b)
    return keep

def prune_container_boxes(boxes, min_children=3, min_area=8000, max_child_area_ratio=0.6, pad=4):
    """
    Remove large container boxes that contain many smaller boxes.
    Helps avoid clicking big panels instead of specific items.
    """
    drop_idx = set()
    for i, b in enumerate(boxes):
        if b.get("type") != "ui_rect":
            continue
        if b["area"] < min_area:
            continue
        child_count = 0
        for j, c in enumerate(boxes):
            if i == j:
                continue
            if c["area"] >= b["area"] * max_child_area_ratio:
                continue
            if contains(b, c, pad=pad):
                child_count += 1
                if child_count >= min_children:
                    drop_idx.add(i)
                    break
    return [b for idx, b in enumerate(boxes) if idx not in drop_idx]

def compute_click_point(box, img_w, img_h):
    """
    Compute a safe click point inside a box.
    Uses OCR text centroid if available, otherwise center of a shrunken box.
    """
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Shrink box to avoid edges/borders
    pad = int(max(2, min(8, min(bw, bh) * 0.12)))
    ix1 = min(x2 - 1, max(x1, x1 + pad))
    iy1 = min(y2 - 1, max(y1, y1 + pad))
    ix2 = max(ix1 + 1, x2 - pad)
    iy2 = max(iy1 + 1, y2 - pad)

    cx = int(round((ix1 + ix2) / 2))
    cy = int(round((iy1 + iy2) / 2))

    # Prefer OCR text centroid when available and inside bounds
    tx = box.get("text_cx")
    ty = box.get("text_cy")
    if tx is not None and ty is not None:
        if ix1 <= tx <= ix2 and iy1 <= ty <= iy2:
            cx, cy = int(tx), int(ty)

    # Clamp to image bounds
    cx = int(max(0, min(img_w - 1, cx)))
    cy = int(max(0, min(img_h - 1, cy)))
    return cx, cy

def add_click_points(boxes, img_w, img_h):
    for b in boxes:
        cx, cy = compute_click_point(b, img_w, img_h)
        b["click_x"] = int(cx)
        b["click_y"] = int(cy)

def _runtime_default_backend() -> str:
    backend = os.environ.get("VISION_BACKEND", DEFAULT_BACKEND).strip().lower() or "auto"
    return backend if backend in {"auto", "vlm", "cv"} else "auto"

def _runtime_default_strict() -> bool:
    return os.environ.get("VISION_STRICT", "1" if DEFAULT_STRICT else "0").strip().lower() in ("1", "true", "yes", "on")

def _runtime_default_max_boxes() -> int:
    try:
        return int(os.environ.get("VISION_MAX_BOXES", str(DEFAULT_MAX_BOXES)))
    except Exception:
        return DEFAULT_MAX_BOXES

def _runtime_default_min_score() -> float:
    try:
        return float(os.environ.get("VISION_MIN_SCORE", str(DEFAULT_MIN_SCORE)))
    except Exception:
        return DEFAULT_MIN_SCORE

# -----------------------------
# VLM (vision model) detector
# -----------------------------
def _extract_json_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    # Remove fenced wrappers when present.
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # Direct parse first.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: parse first balanced {...} region.
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = t[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None

def _encode_image_block_for_vlm(img_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None

    long_edge_max = 1568
    total_pixels_max = 1_150_000
    max_bytes = 4_800_000

    scale_long = long_edge_max / float(max(w, h))
    scale_pix = np.sqrt(total_pixels_max / float(max(1, w * h)))
    scale = min(1.0, scale_long, scale_pix)

    img = img_bgr
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    encoded = buf.tobytes()
    media_type = "image/png"

    if len(encoded) > max_bytes:
        media_type = "image/jpeg"
        for quality in (90, 80, 70, 60, 50):
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                continue
            encoded = buf.tobytes()
            if len(encoded) <= max_bytes:
                break

    b64 = base64.b64encode(encoded).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }

def _parse_model_box(raw: Dict[str, Any], img_w: int, img_h: int) -> Optional[Dict[str, Any]]:
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _to_bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"1", "true", "yes", "y", "on"}:
                return True
            if t in {"0", "false", "no", "n", "off"}:
                return False
        return default

    # Accept either x1/y1/x2/y2 or x/y/w/h.
    if all(k in raw for k in ("x1", "y1", "x2", "y2")):
        x1 = _to_float(raw.get("x1"))
        y1 = _to_float(raw.get("y1"))
        x2 = _to_float(raw.get("x2"))
        y2 = _to_float(raw.get("y2"))
    elif all(k in raw for k in ("x", "y", "w", "h")):
        x1 = _to_float(raw.get("x"))
        y1 = _to_float(raw.get("y"))
        x2 = x1 + _to_float(raw.get("w"))
        y2 = y1 + _to_float(raw.get("h"))
    else:
        return None

    vals = [x1, y1, x2, y2]
    vmax = max(abs(v) for v in vals)
    vmin = min(vals)

    # Coordinate mode detection:
    # 1) normalized [0,1], 2) percent [0,100], 3) absolute pixels.
    if vmax <= 1.5 and vmin >= -0.2:
        x1 *= img_w
        x2 *= img_w
        y1 *= img_h
        y2 *= img_h
    elif vmax <= 100.0 and vmin >= -1.0:
        x1 = (x1 / 100.0) * img_w
        x2 = (x2 / 100.0) * img_w
        y1 = (y1 / 100.0) * img_h
        y2 = (y2 / 100.0) * img_h

    x1i = int(round(min(x1, x2)))
    y1i = int(round(min(y1, y2)))
    x2i = int(round(max(x1, x2)))
    y2i = int(round(max(y1, y2)))

    candidate = {"x1": x1i, "y1": y1i, "x2": x2i, "y2": y2i}
    clamp_box(candidate, img_w, img_h)
    if candidate["w"] < 6 or candidate["h"] < 6:
        return None

    conf = _to_float(raw.get("confidence", raw.get("score", 0.80)), default=0.80)
    conf = _clamp01(conf if conf <= 1.0 else (conf / 100.0))
    element_type = str(raw.get("element_type", raw.get("type", "ui_element")) or "ui_element").strip().lower()
    label = str(raw.get("label", raw.get("text", "")) or "").strip()
    is_container_like = _to_bool(raw.get("is_container"), default=element_type in VLM_CONTAINER_TYPES)
    model_interactive = _to_bool(raw.get("interactive"), default=False)
    interactive = bool(model_interactive or (element_type in VLM_ACTIONABLE_TYPES))

    # Guardrail: container-like types should not be treated as clickable by default.
    if element_type in VLM_CONTAINER_TYPES and element_type not in {"toolbar"}:
        interactive = False
    if is_container_like and element_type not in {"toolbar"}:
        interactive = False

    candidate.update(
        {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": element_type[:40],
            "is_container": bool(is_container_like),
            "interactive": interactive,
            "score": float(conf),
            **({"text": label[:200]} if label else {}),
        }
    )
    return candidate

def _visual_evidence_score(img_bgr: np.ndarray, box: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    luma = _luma_for_ui(region)
    contrast = float(np.std(luma)) / 255.0
    edges = cv2.Canny(luma, 30, 110)
    edge_ratio = float((edges > 0).mean())
    score = min(1.0, 1.3 * contrast + 3.5 * edge_ratio)
    return float(score)

def _max_iou(box: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
    if not others:
        return 0.0
    best = 0.0
    for other in others:
        best = max(best, iou(box, other))
        if best > 0.9:
            return best
    return best

def _validate_vlm_boxes(
    img_bgr: np.ndarray,
    boxes: List[Dict[str, Any]],
    strict: bool,
    cv_hints: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    h, w = img_bgr.shape[:2]
    screen_area = max(1, w * h)
    cv_hints = cv_hints or []
    kept: List[Dict[str, Any]] = []

    for b in boxes:
        ar = float(b["w"]) / float(max(1, b["h"]))
        area_ratio = float(b["area"]) / float(screen_area)
        if ar < 1 / 35 or ar > 35:
            continue
        if area_ratio < 0.00004:
            continue
        if strict and area_ratio > 0.24 and not b.get("interactive", False):
            continue
        if strict and b.get("is_container") and area_ratio > 0.10:
            continue
        if area_ratio > 0.70:
            continue

        evidence = _visual_evidence_score(img_bgr, b)
        overlap_hint = _max_iou(b, cv_hints)
        if strict:
            if evidence < 0.060 and overlap_hint < 0.10:
                continue
        else:
            if evidence < 0.035 and overlap_hint < 0.04:
                continue

        b["score"] = float(min(1.0, float(b.get("score", 0.75)) + min(0.20, 0.6 * evidence) + min(0.10, 0.2 * overlap_hint)))
        kept.append(b)

    return kept

def _prune_vlm_container_candidates(boxes: List[Dict[str, Any]], img_w: int, img_h: int, strict: bool) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    screen_area = max(1, img_w * img_h)
    drop_idx = set()
    for i, b in enumerate(boxes):
        if b.get("source") != "vlm":
            continue
        area_ratio = float(b.get("area", 0)) / float(screen_area)
        is_container = bool(b.get("is_container", False)) or (str(b.get("element_type", "")).lower() in VLM_CONTAINER_TYPES)
        child_count = 0
        interactive_children = 0

        for j, c in enumerate(boxes):
            if i == j:
                continue
            if c.get("area", 0) >= b.get("area", 0):
                continue
            if c.get("area", 0) > b.get("area", 0) * 0.85:
                continue
            if contains(b, c, pad=2):
                child_count += 1
                if c.get("interactive"):
                    interactive_children += 1
                if child_count >= 4:
                    break

        # Remove broad container-like parents that hide the real click targets.
        if child_count >= 3 and (is_container or area_ratio > 0.14):
            if strict or interactive_children >= 2:
                drop_idx.add(i)
                continue

        # Remove giant non-actionable regions even if they don't have many children.
        if is_container and not b.get("interactive", False) and area_ratio > (0.12 if strict else 0.18):
            drop_idx.add(i)
            continue

    kept = [b for idx, b in enumerate(boxes) if idx not in drop_idx]

    # Prefer the smaller actionable box when a container fully contains it.
    drop_idx = set()
    for i, outer in enumerate(kept):
        if outer.get("source") != "vlm":
            continue
        outer_container = bool(outer.get("is_container")) or str(outer.get("element_type", "")).lower() in VLM_CONTAINER_TYPES
        if not outer_container:
            continue
        for j, inner in enumerate(kept):
            if i == j or inner.get("source") != "vlm":
                continue
            if not inner.get("interactive", False):
                continue
            if contains(outer, inner, pad=2) and inner.get("area", 0) < outer.get("area", 0) * 0.8:
                drop_idx.add(i)
                break

    return [b for idx, b in enumerate(kept) if idx not in drop_idx]

def _call_vlm_once(
    img_bgr: np.ndarray,
    strict: bool,
    max_elements: int,
    timeout_s: float,
    model_name: str,
) -> List[Dict[str, Any]]:
    try:
        from anthropic import Anthropic
    except Exception:
        return []

    api_key = (os.environ.get("claudekey") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return []

    image_block = _encode_image_block_for_vlm(img_bgr)
    if not image_block:
        return []

    h, w = img_bgr.shape[:2]
    limit = max(10, int(max_elements))

    prompt = (
        "You are a macOS UI detector.\n"
        "Task: detect ONLY visible, real UI elements from this screenshot.\n"
        "Critical anti-hallucination rules:\n"
        "- Never invent elements not visibly present.\n"
        "- Ignore decorative backgrounds, gradients, shadows, empty whitespace.\n"
        "- Prefer single actionable controls (buttons, fields, menu items, rows) over giant containers.\n"
        "- Do NOT return page/window/panel/group containers when child controls are visible.\n"
        "- Keep boxes tight around the exact control; avoid loose boxes spanning multiple controls.\n"
        "- Avoid nested duplicates and heavily-overlapping duplicates.\n"
        "- Avoid boxes covering more than ~35% of the screen unless it is one isolated control.\n"
        "- Coordinates must be normalized to [0,1] of the full screenshot.\n"
        f"- Return at most {limit} elements.\n"
        "Return JSON only in this exact shape:\n"
        "{\n"
        '  "elements": [\n'
        "    {\n"
        '      "element_type": "button|input|link|checkbox|radio|switch|tab|menuitem|row|cell|icon|text|panel|other",\n'
        '      "interactive": true,\n'
        '      "is_container": false,\n'
        '      "label": "short visible text if any",\n'
        '      "confidence": 0.0,\n'
        '      "x1": 0.0,\n'
        '      "y1": 0.0,\n'
        '      "x2": 0.0,\n'
        '      "y2": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Image size context: {w}x{h}. Strict mode: {str(bool(strict)).lower()}.\n"
    )

    client = Anthropic(api_key=api_key, timeout=timeout_s)
    response = client.messages.create(
        model=model_name,
        max_tokens=2800,
        temperature=0,
        system="Detect UI elements from screenshots. Output strict JSON only.",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, image_block]}],
    )

    text = ""
    for block in getattr(response, "content", []) or []:
        block_text = getattr(block, "text", None)
        if block_text:
            text += str(block_text).strip() + "\n"
    payload = _extract_json_dict_from_text(text)
    if not payload:
        return []

    raw_elements = payload.get("elements", [])
    if not isinstance(raw_elements, list):
        return []

    out: List[Dict[str, Any]] = []
    for raw in raw_elements[:limit * 2]:
        if not isinstance(raw, dict):
            continue
        b = _parse_model_box(raw, w, h)
        if not b:
            continue
        out.append(b)
    return out

def detect_ui_elements_vlm(
    img_bgr: np.ndarray,
    strict: bool = False,
    max_elements: int = DEFAULT_VLM_MAX_ELEMENTS,
) -> List[Dict[str, Any]]:
    use_vlm = os.environ.get("VISION_USE_VLM", "1" if DEFAULT_VLM_ENABLED else "0").strip().lower() in ("1", "true", "yes", "on")
    if not use_vlm:
        return []

    model_name = (os.environ.get("VISION_VLM_MODEL") or DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL
    timeout_s = float(os.environ.get("VISION_VLM_TIMEOUT", str(DEFAULT_VLM_TIMEOUT)))
    max_elements = max(10, int(max_elements))

    passes = 2 if strict else 1
    pass_results: List[List[Dict[str, Any]]] = []
    for _ in range(passes):
        try:
            pass_results.append(_call_vlm_once(img_bgr, strict=strict, max_elements=max_elements, timeout_s=timeout_s, model_name=model_name))
        except Exception:
            pass_results.append([])

    merged = [b for boxes in pass_results for b in boxes]
    if not merged:
        return []

    # Consensus clustering across passes to reduce random one-off detections.
    clusters: List[Dict[str, Any]] = []
    for pass_idx, boxes in enumerate(pass_results):
        for b in boxes:
            matched_idx = -1
            best_iou = 0.0
            for i, c in enumerate(clusters):
                overlap = iou(b, c["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    matched_idx = i
            if matched_idx >= 0 and best_iou >= 0.55:
                c = clusters[matched_idx]
                c["members"].append(b)
                c["passes"].add(pass_idx)
                members = c["members"]
                c["box"] = {
                    **b,
                    "x1": int(round(sum(m["x1"] for m in members) / len(members))),
                    "y1": int(round(sum(m["y1"] for m in members) / len(members))),
                    "x2": int(round(sum(m["x2"] for m in members) / len(members))),
                    "y2": int(round(sum(m["y2"] for m in members) / len(members))),
                    "score": float(sum(float(m.get("score", 0.0)) for m in members) / len(members)),
                }
            else:
                clusters.append({"box": dict(b), "members": [b], "passes": {pass_idx}})

    min_support = 2 if passes >= 2 and strict else 1
    consensus = []
    h, w = img_bgr.shape[:2]
    for c in clusters:
        if len(c["passes"]) < min_support:
            continue
        b = dict(c["box"])
        clamp_box(b, w, h)
        b["score"] = float(min(1.0, float(b.get("score", 0.0)) + 0.08 * len(c["passes"])))
        consensus.append(b)

    if not consensus:
        consensus = merged

    # CV hints are used only for anti-hallucination gating.
    cv_hints = detect_ui_rectangles(img_bgr, scale=1.0, min_score=max(0.35, _runtime_default_min_score() * 0.55))
    text_hints = detect_text_regions(img_bgr)
    cv_hints.extend([t for t in text_hints if float(t.get("score", 0.0)) >= 0.45])
    for c in cv_hints:
        clamp_box(c, w, h)
    cv_hints = [c for c in cv_hints if c.get("area", 0) > 0]

    validated = _validate_vlm_boxes(img_bgr, consensus, strict=strict, cv_hints=cv_hints)
    validated = _prune_vlm_container_candidates(validated, w, h, strict=strict)
    validated = nms(validated, iou_thresh=0.30 if strict else 0.35)
    validated = prune_contained(validated, containment_iou=0.94 if strict else 0.97)

    # Rank: interactive and text-labelled elements first.
    def _rank(b: Dict[str, Any]) -> Tuple[float, float]:
        score = float(b.get("score", 0.0))
        if b.get("interactive"):
            score += 0.35
        if b.get("is_container"):
            score -= 0.35
        if b.get("text"):
            score += 0.20
        area_pen = float(b.get("area", 0.0)) / float(max(1, w * h))
        return (score, -area_pen)

    validated.sort(key=_rank, reverse=True)
    return validated[:max_elements]


# -----------------------------
# Main pipeline
# -----------------------------
def parse_ui_everything(img_bgr,
                        rect_scales=(1.0, 0.75, 0.5),
                        nms_iou=0.35,
                        containment_iou=0.95,
                        refine_boxes=True,
                        backend: Optional[str] = None,
                        strict: Optional[bool] = None,
                        max_boxes: Optional[int] = None,
                        min_score: Optional[float] = None):
    h, w = img_bgr.shape[:2]

    backend = (backend if backend is not None else _runtime_default_backend()).strip().lower()
    if backend not in {"auto", "vlm", "cv"}:
        backend = "auto"
    strict = _runtime_default_strict() if strict is None else bool(strict)
    max_boxes = _runtime_default_max_boxes() if max_boxes is None else int(max_boxes)
    min_score = _runtime_default_min_score() if min_score is None else float(min_score)
    cv_supplement = os.environ.get("VISION_CV_SUPPLEMENT", "1" if DEFAULT_CV_SUPPLEMENT else "0").strip().lower() in ("1", "true", "yes", "on")

    def _max_iou_with(box: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
        if not others:
            return 0.0
        best = 0.0
        for o in others:
            best = max(best, iou(box, o))
            if best >= 0.90:
                return best
        return best

    # 0) Vision model proposals (semantic UI understanding)
    vlm_boxes: List[Dict[str, Any]] = []
    if backend in ("auto", "vlm"):
        try:
            vlm_boxes = detect_ui_elements_vlm(img_bgr, strict=strict, max_elements=max(40, max_boxes if max_boxes else DEFAULT_VLM_MAX_ELEMENTS))
        except Exception:
            vlm_boxes = []

    # 1) CV proposals at multiple scales (fallback + optional supplement)
    cv_boxes: List[Dict[str, Any]] = []
    if backend in ("auto", "cv", "vlm"):
        # Use CV always for backend=cv.
        # In auto/vlm modes: run CV when VLM output is weak/empty, or when explicit supplement is enabled.
        weak_vlm = len(vlm_boxes) < (8 if strict else 4)
        run_cv = (backend == "cv") or weak_vlm or (backend in ("auto", "vlm") and cv_supplement)
        if run_cv:
            rect_boxes: List[Dict[str, Any]] = []
            for s in rect_scales:
                rect_boxes.extend(detect_ui_rectangles(img_bgr, scale=s, min_score=min_score))

            text_boxes = detect_text_regions(img_bgr)
            # Text regions tend to be slightly noisier than rectangles; keep a looser gate.
            text_min = min(0.55, float(min_score))
            text_boxes = [b for b in text_boxes if float(b.get("score", 0.0)) >= text_min]
            cv_boxes = rect_boxes + text_boxes

    # Merge priority:
    # - VLM first (semantic UI)
    # - CV fallback/supplement
    boxes: List[Dict[str, Any]] = vlm_boxes + cv_boxes

    # Clamp + area, center
    for b in boxes:
        clamp_box(b, w, h)

    # Drop boxes with near-zero area (after clamp)
    boxes = [b for b in boxes if b["area"] > 0]

    # If VLM produced boxes, drop overlapping CV boxes to reduce duplicate/phantom targets.
    if vlm_boxes:
        vlm_clamped = [b for b in boxes if b.get("source") == "vlm"]
        kept: List[Dict[str, Any]] = []
        for b in boxes:
            if b.get("source") != "cv":
                kept.append(b)
                continue
            if _max_iou_with(b, vlm_clamped) >= 0.30:
                continue
            kept.append(b)
        boxes = kept

    # 3) NMS (dedupe)
    boxes = nms(boxes, iou_thresh=nms_iou)

    # 4) Prune contained duplicates
    boxes = prune_contained(boxes, containment_iou=containment_iou)

    # 5) Prefer ui_rect over text_region when overlapping (reduce duplicates)
    boxes = prune_text_regions_inside_rects(boxes, max_area_ratio=6.0, pad=4)

    # 6) Remove large container boxes that surround many elements
    boxes = prune_container_boxes(boxes, min_children=3, min_area=8000, max_child_area_ratio=0.6, pad=4)

    # 7) Filter very small boxes that are inside larger ones
    boxes = filter_overlapping_small_boxes(boxes, min_area_ratio=0.3)

    # 8) Refine boxes to fit content tightly
    if refine_boxes:
        for b in boxes:
            if b.get("source") == "vlm":
                continue
            # Only refine boxes that are reasonably sized (not too big, not too small)
            if 100 < b["area"] < 500000:  # Skip full-screen and tiny boxes
                refine_box_to_content(img_bgr, b, padding=2)
                # Recalculate after refinement
                clamp_box(b, w, h)

    # 9) Compute safe click points
    add_click_points(boxes, w, h)

    # 10) Cap outputs: fewer boxes reduces agent confusion and prevents "random" targets.
    if max_boxes and len(boxes) > max_boxes:
        def _rank_key(b: Dict[str, Any]) -> Tuple[float, float]:
            score = float(b.get("score", 0.0))
            # Strongly prefer interactive model boxes, then text-bearing items.
            if b.get("source") == "vlm" and b.get("interactive") is True:
                score += 2.2
            elif b.get("source") == "vlm":
                score += 1.2
            if b.get("text"):
                score += 0.35
            # Prefer smaller targets slightly (less likely to be container panes).
            area_pen = float(b.get("area", 0)) / float(max(w * h, 1))  # [0..1]
            return (score, -area_pen)

        boxes = sorted(boxes, key=_rank_key, reverse=True)[:max_boxes]
        # Re-sort in reading order for stable IDs.
        boxes.sort(key=lambda b: (b["y1"], b["x1"]))

    # Sort in reading order (good for agent)
    else:
        boxes.sort(key=lambda b: (b["y1"], b["x1"]))

    # Assign IDs
    for i, b in enumerate(boxes):
        b["id"] = i

    return boxes

def draw(img_bgr, boxes, show_click_points=True):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Style tuning for readability at higher resolutions.
    outline_color = (0, 255, 0)   # green for element boxes (BGR)
    click_color = (0, 0, 255)     # red for click points
    label_bg = (0, 215, 255)      # gold/yellow label background (BGR)
    label_border = (0, 0, 0)      # black border for label
    label_text = (0, 0, 0)        # black text on yellow

    # Scale font/line widths based on image size to keep IDs legible.
    base = max(1, min(w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(1.2, base / 1400.0))
    text_thickness = max(1, int(round(font_scale * 2)))
    rect_thickness = max(1, int(round(font_scale * 2)))
    pad_x = max(3, int(round(font_scale * 4)))
    pad_y = max(2, int(round(font_scale * 3)))

    def _rects_overlap(a, b):
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _clamp_rect(x1, y1, rw, rh):
        x1 = max(0, min(w - rw, x1))
        y1 = max(0, min(h - rh, y1))
        return x1, y1, x1 + rw, y1 + rh

    occupied = []

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), outline_color, rect_thickness)

        # Use ID only for labels to reduce OCR ambiguity in the overlay.
        label = f'{b["id"]}'
        (tw, th), _baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

        # Prefer placing label above the box; if not possible, place inside.
        label_w = tw + 2 * pad_x
        label_h = th + 2 * pad_y
        candidates = [
            (x1, y1 - label_h),         # above left
            (x2 - label_w, y1 - label_h),# above right
            (x1, y1),                    # inside top-left
            (x2 - label_w, y1),          # inside top-right
            (x1, y2),                    # below left
            (x2 - label_w, y2),          # below right
            (x1 - label_w, y1),          # left
            (x2, y1),                    # right
        ]

        lx1 = ly1 = lx2 = ly2 = None
        for cx, cy in candidates:
            rx1, ry1, rx2, ry2 = _clamp_rect(int(cx), int(cy), int(label_w), int(label_h))
            if all(not _rects_overlap((rx1, ry1, rx2, ry2), o) for o in occupied):
                lx1, ly1, lx2, ly2 = rx1, ry1, rx2, ry2
                break

        if lx1 is None:
            # Fallback: clamp above-left even if it overlaps.
            lx1, ly1, lx2, ly2 = _clamp_rect(x1, y1 - label_h, int(label_w), int(label_h))
        occupied.append((lx1, ly1, lx2, ly2))

        # Draw label background + border.
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), label_bg, -1)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), label_border, max(1, rect_thickness - 1))

        # Draw label text.
        tx = lx1 + pad_x
        ty = ly1 + pad_y + th
        cv2.putText(out, label, (tx, ty), font, font_scale, label_text, text_thickness, cv2.LINE_AA)

        # Optional: draw click point (no coordinates)
        if show_click_points and "click_x" in b and "click_y" in b:
            cx, cy = int(b["click_x"]), int(b["click_y"])
            cv2.circle(out, (cx, cy), max(2, int(round(font_scale * 2))), click_color, -1)

    return out

def crop(img, b, pad=2):
    h, w = img.shape[:2]
    x1 = max(0, b["x1"] - pad)
    y1 = max(0, b["y1"] - pad)
    x2 = min(w, b["x2"] + pad)
    y2 = min(h, b["y2"] + pad)
    return img[y1:y2, x1:x2].copy()

def assign_words_to_boxes(boxes, words, min_conf=45):
    if not words:
        return
    # Pre-sort words to keep reading order
    words_sorted = sorted(words, key=lambda w: (w["y"], w["x"]))
    for b in boxes:
        collected = []
        cxs = []
        cys = []
        for w in words_sorted:
            if w["conf"] < min_conf:
                continue
            cx = w["x"] + w["w"] / 2
            cy = w["y"] + w["h"] / 2
            if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
                collected.append(w["text"])
                cxs.append(cx)
                cys.append(cy)
        if collected:
            b["text"] = " ".join(collected)
            b["text_cx"] = int(round(sum(cxs) / len(cxs)))
            b["text_cy"] = int(round(sum(cys) / len(cys)))
def create_mser():
    mser = cv2.MSER_create()
    try:
        mser.setDelta(5)
        mser.setMinArea(30)
        mser.setMaxArea(8000)
    except Exception:
        pass
    return mser

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to screenshot/image")
    ap.add_argument("--out_dir", default="mac_ui_outputs", help="Output folder")
    ap.add_argument("--save_crops", action="store_true", help="Save crops for each detected region")
    ap.add_argument("--ocr", action="store_true", help="Try OCR on text regions (requires pytesseract + tesseract)")
    ap.add_argument("--nms", type=float, default=0.35, help="NMS IoU threshold")
    ap.add_argument("--contain", type=float, default=0.95, help="Containment prune IoU threshold")
    ap.add_argument("--backend", choices=["auto", "vlm", "cv"], default=_runtime_default_backend(), help="UI detection backend preference")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=_runtime_default_strict(), help="Prefer fewer, more-clickable elements")
    ap.add_argument("--max_boxes", type=int, default=_runtime_default_max_boxes(), help="Cap the number of output boxes")
    ap.add_argument("--min_score", type=float, default=_runtime_default_min_score(), help="Minimum score for CV-proposed boxes")
    ap.add_argument("--use_vlm", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable VLM detector")
    ap.add_argument("--vlm_model", default=None, help="Override VLM model name")
    ap.add_argument("--vlm_timeout", type=float, default=None, help="VLM API timeout seconds")
    args = ap.parse_args()

    if args.use_vlm is not None:
        os.environ["VISION_USE_VLM"] = "1" if args.use_vlm else "0"
    if args.vlm_model:
        os.environ["VISION_VLM_MODEL"] = str(args.vlm_model)
    if args.vlm_timeout is not None:
        os.environ["VISION_VLM_TIMEOUT"] = str(float(args.vlm_timeout))

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    boxes = parse_ui_everything(
        img,
        rect_scales=(1.0, 0.75, 0.5),
        nms_iou=args.nms,
        containment_iou=args.contain,
        backend=args.backend,
        strict=args.strict,
        max_boxes=args.max_boxes,
        min_score=args.min_score,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OCR disabled by default (see ENABLE_OCR)
    ocr_words = []
    if args.ocr and ENABLE_OCR:
        ocr_words = try_ocr_words(img)
        if ocr_words:
            assign_words_to_boxes(boxes, ocr_words)
            # Recompute click points using OCR text centroids
            add_click_points(boxes, img.shape[1], img.shape[0])

    # Optional crops and OCR
    crops_dir = out_dir / f"{img_path.stem}_crops"
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for b in boxes:
        if args.save_crops or (args.ocr and b["type"] == "text_region"):
            c = crop(img, b, pad=2)

            if args.save_crops:
                cv2.imwrite(str(crops_dir / f'{b["id"]:04d}_{b["type"]}.png'), c)

            if args.ocr and ENABLE_OCR and b["type"] == "text_region" and not b.get("text"):
                text = try_ocr_on_crop(c)
                if text:
                    b["text"] = text

    # Save JSON
    json_path = out_dir / f"{img_path.stem}_ui_everything.json"
    payload = {
        "image": str(img_path),
        "num_boxes": len(boxes),
        "boxes": boxes
    }
    if args.ocr and ENABLE_OCR:
        payload["ocr_words"] = ocr_words
    json_path.write_text(json.dumps(payload, indent=2))

    # Save annotated image
    annotated = draw(img, boxes)
    ann_path = out_dir / f"{img_path.stem}_ui_everything_annotated.png"
    cv2.imwrite(str(ann_path), annotated)

    print(f"Boxes: {len(boxes)}")
    print(f"JSON:  {json_path}")
    print(f"Image: {ann_path}")
    if args.save_crops:
        print(f"Crops: {crops_dir}")
    if args.ocr:
        print("OCR: enabled (text_region boxes may include 'text' field if OCR worked)")

if __name__ == "__main__":
    main()
