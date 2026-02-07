import cv2
import numpy as np
import json
import argparse
import os
import time
import re
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# OCR is disabled by default; set to True if you want OCR features.
ENABLE_OCR = True

# Default backend order: automatic (semantic + CV fallback/supplement).
DEFAULT_BACKEND = os.environ.get("VISION_BACKEND", "auto").strip().lower()  # auto|cv
DEFAULT_STRICT = os.environ.get("VISION_STRICT", "1").strip() in ("1", "true", "yes", "on")
DEFAULT_MAX_BOXES = int(os.environ.get("VISION_MAX_BOXES", "300"))
DEFAULT_MIN_SCORE = float(os.environ.get("VISION_MIN_SCORE", "0.48"))
DEFAULT_CV_SUPPLEMENT = os.environ.get("VISION_CV_SUPPLEMENT", "0").strip() in ("1", "true", "yes", "on")
DEFAULT_MAX_AREA_RATIO = float(os.environ.get("VISION_MAX_AREA_RATIO", "0.16"))

# Florence-2 local vision model config
DEFAULT_FLORENCE_ENABLED = os.environ.get("VISION_USE_FLORENCE", "0").strip() in ("1", "true", "yes", "on")
DEFAULT_FLORENCE_MODEL = os.environ.get("VISION_FLORENCE_MODEL", "microsoft/Florence-2-base").strip()
DEFAULT_INCLUDE_LEGACY_JSON = os.environ.get("VISION_INCLUDE_LEGACY_JSON", "0").strip() in ("1", "true", "yes", "on")
DEFAULT_PRETTY_JSON = os.environ.get("VISION_PRETTY_JSON", "0").strip() in ("1", "true", "yes", "on")

VLM_ACTIONABLE_TYPES = {
    "button", "link", "checkbox", "radio", "switch", "toggle", "menuitem", "tab",
    "textfield", "input", "combobox", "dropdown", "slider", "row", "cell",
    "listitem", "treeitem", "iconbutton", "segmentedcontrol", "stepper",
}
VLM_CONTAINER_TYPES = {
    "panel", "window", "group", "section", "background", "layout", "container",
    "toolbar", "sidebar", "scrollarea", "canvas", "other",
}

ROLE_CANONICAL_MAP = {
    "button": "button",
    "iconbutton": "icon_button",
    "icon_button": "icon_button",
    "link": "link",
    "checkbox": "checkbox",
    "radio": "radio",
    "switch": "switch",
    "toggle": "switch",
    "menuitem": "menu_item",
    "menu_item": "menu_item",
    "tab": "tab",
    "textfield": "text_field",
    "input": "text_field",
    "combobox": "combo_box",
    "dropdown": "dropdown",
    "slider": "slider",
    "row": "list_row",
    "cell": "table_cell",
    "listitem": "list_row",
    "treeitem": "tree_item",
    "segmentedcontrol": "segmented_control",
    "stepper": "stepper",
    "toolbar": "toolbar",
    "sidebar": "sidebar",
    "window": "window",
    "panel": "panel",
    "group": "group",
    "text": "text",
    "text_region": "text",
    "ui_rect": "button",
    "colored_control": "button",
    "address_bar": "address_bar",
    "search_field": "search_field",
    "window_control_close": "window_control_close",
    "window_control_minimize": "window_control_minimize",
    "window_control_zoom": "window_control_zoom",
    "dock_item": "dock_item",
    "other": "other",
}

CONTAINER_ROLES = {
    "window", "panel", "group", "toolbar", "sidebar", "other"
}

CLICKABLE_ROLES = {
    "button", "icon_button", "link", "checkbox", "radio", "switch", "menu_item",
    "tab", "text_field", "combo_box", "dropdown", "slider", "list_row", "table_cell",
    "tree_item", "segmented_control", "stepper", "address_bar", "search_field",
    "window_control_close", "window_control_minimize", "window_control_zoom", "dock_item",
}

# -----------------------------
# Type color palette (BGR for OpenCV) & annotation config
# -----------------------------
TYPE_PALETTE = {
    # Interactive controls
    "button":           (244, 133, 66),   # Blue
    "iconbutton":       (244, 133, 66),   # Blue
    "link":             (187, 85, 153),   # Purple
    "input":            (83, 168, 52),    # Green
    "textfield":        (83, 168, 52),    # Green
    "combobox":         (83, 168, 52),    # Green
    "dropdown":         (203, 134, 121),  # Indigo
    "checkbox":         (4, 188, 251),    # Yellow
    "radio":            (4, 188, 251),    # Yellow
    "switch":           (4, 188, 251),    # Yellow
    "toggle":           (4, 188, 251),    # Yellow
    "slider":           (77, 183, 255),   # Amber
    "stepper":          (77, 183, 255),   # Amber
    "tab":              (193, 172, 0),    # Teal
    "segmentedcontrol": (193, 172, 0),    # Teal
    "menuitem":         (67, 112, 255),   # Orange
    "menu_item":        (67, 112, 255),   # Orange
    "listitem":         (196, 203, 128),  # Light Teal
    "list_row":         (196, 203, 128),  # Light Teal
    "treeitem":         (196, 203, 128),  # Light Teal
    "tree_item":        (196, 203, 128),  # Light Teal
    "row":              (196, 203, 128),  # Light Teal
    "table_cell":       (196, 203, 128),  # Light Teal
    "cell":             (196, 203, 128),  # Light Teal
    "icon_button":      (244, 133, 66),   # Blue
    "text_field":       (83, 168, 52),    # Green
    "search_field":     (83, 168, 52),    # Green
    "address_bar":      (83, 168, 52),    # Green
    "combo_box":        (83, 168, 52),    # Green
    "segmented_control": (193, 172, 0),   # Teal
    "window_control_close": (0, 0, 255),  # Red
    "window_control_minimize": (0, 255, 255),  # Yellow
    "window_control_zoom": (0, 180, 0),   # Green
    "dock_item":        (177, 143, 244),  # Pink
    # Non-interactive / structural
    "icon":             (177, 143, 244),  # Pink
    "text":             (158, 158, 158),  # Gray
    "panel":            (139, 125, 96),   # Blue Gray
    "window":           (139, 125, 96),   # Blue Gray
    "group":            (139, 125, 96),   # Blue Gray
    "toolbar":          (139, 125, 96),   # Blue Gray
    "sidebar":          (139, 125, 96),   # Blue Gray
    # CV-detected types
    "ui_rect":          (246, 181, 100),  # Light Blue
    "text_region":      (132, 199, 129),  # Light Green
    "colored_control":  (244, 133, 66),   # Blue (same as button)
}

TYPE_DISPLAY_NAMES = {
    "button": "Button", "iconbutton": "Icon Btn", "link": "Link",
    "input": "Input", "textfield": "Text Field", "combobox": "Combo",
    "dropdown": "Dropdown", "checkbox": "Checkbox", "radio": "Radio",
    "switch": "Switch", "toggle": "Toggle", "slider": "Slider",
    "stepper": "Stepper", "tab": "Tab", "segmentedcontrol": "Segment",
    "menuitem": "Menu Item", "listitem": "List Item", "treeitem": "Tree Item",
    "menu_item": "Menu Item", "list_row": "List Row", "tree_item": "Tree Item", "table_cell": "Table Cell",
    "icon_button": "Icon Btn", "text_field": "Text Field", "search_field": "Search Field",
    "address_bar": "Address Bar", "combo_box": "Combo", "segmented_control": "Segment",
    "window_control_close": "Close", "window_control_minimize": "Minimize", "window_control_zoom": "Zoom",
    "dock_item": "Dock Item",
    "row": "Row", "cell": "Cell", "icon": "Icon", "text": "Text",
    "panel": "Panel", "window": "Window", "group": "Group",
    "toolbar": "Toolbar", "sidebar": "Sidebar",
    "ui_rect": "UI Element", "text_region": "Text Region",
    "colored_control": "Control",
}

_DEFAULT_COLOR = (200, 200, 200)  # Fallback gray

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

def _group_ocr_words_into_lines(words: List[Dict[str, Any]], min_conf: float = 45.0) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    ws = [w for w in words if float(w.get("conf", -1)) >= float(min_conf) and str(w.get("text", "")).strip()]
    if not ws:
        return []
    ws = sorted(ws, key=lambda w: (int(w["y"]), int(w["x"])))

    lines: List[List[Dict[str, Any]]] = []
    for w in ws:
        wy = int(w["y"])
        wh = int(max(1, w["h"]))
        wmid = wy + wh // 2
        best_idx = -1
        best_delta = 10**9
        for idx, line in enumerate(lines):
            ly_vals = [int(x["y"]) + int(max(1, x["h"])) // 2 for x in line]
            ly = int(round(sum(ly_vals) / max(1, len(ly_vals))))
            avg_h = int(round(sum(int(max(1, x["h"])) for x in line) / max(1, len(line))))
            tol = int(max(8, avg_h * 0.85))
            delta = abs(wmid - ly)
            if delta <= tol and delta < best_delta:
                best_idx = idx
                best_delta = delta
        if best_idx >= 0:
            lines[best_idx].append(w)
        else:
            lines.append([w])

    for line in lines:
        line.sort(key=lambda w: int(w["x"]))
    return lines

def _ocr_line_boxes(words: List[Dict[str, Any]], img_w: int, img_h: int, min_conf: float = 45.0) -> List[Dict[str, Any]]:
    lines = _group_ocr_words_into_lines(words, min_conf=min_conf)
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not line:
            continue
        x1 = min(int(w["x"]) for w in line)
        y1 = min(int(w["y"]) for w in line)
        x2 = max(int(w["x"]) + int(w["w"]) for w in line)
        y2 = max(int(w["y"]) + int(w["h"]) for w in line)

        # Avoid giant top-line OCR strips (menu bar / status text concatenations).
        if (x2 - x1) > int(0.42 * img_w) and y2 <= int(0.10 * img_h):
            continue

        # Expand modestly so the click target includes glyph anti-aliasing.
        pad_x = int(max(2, round((x2 - x1) * 0.06)))
        pad_y = int(max(2, round((y2 - y1) * 0.25)))
        b = {
            "type": "text_region",
            "source": "ocr",
            "x1": x1 - pad_x,
            "y1": y1 - pad_y,
            "x2": x2 + pad_x,
            "y2": y2 + pad_y,
            "score": float(min(0.95, 0.60 + 0.03 * len(line))),
            "text": " ".join(str(w.get("text", "")).strip() for w in line if str(w.get("text", "")).strip()),
            "element_type": "text",
            "role": "text",
            "interactive": False,
        }
        clamp_box(b, img_w, img_h)
        if b["w"] < 8 or b["h"] < 8:
            continue
        out.append(b)
    return out

def _ocr_word_boxes(
    words: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    min_conf: float = 55.0,
) -> List[Dict[str, Any]]:
    if not words:
        return []
    out: List[Dict[str, Any]] = []
    top_menu_limit = int(max(18, min(54, img_h * 0.07)))
    for w in words:
        conf = float(w.get("conf", -1))
        if conf < float(min_conf):
            continue
        text = _normalize_label(w.get("text", ""))
        if not text:
            continue
        bw = int(max(1, w.get("w", 0)))
        bh = int(max(1, w.get("h", 0)))
        if bw < 5 or bh < 7:
            continue
        if bw > int(0.45 * img_w) or bh > int(0.20 * img_h):
            continue
        x = int(w.get("x", 0))
        y = int(w.get("y", 0))
        pad_x = int(max(1, min(8, round(bw * 0.22))))
        pad_y = int(max(1, min(8, round(bh * 0.28))))
        role = "menu_item" if y <= top_menu_limit else "text"
        candidate = {
            "type": "text_region",
            "source": "ocr_word",
            "role": role,
            "element_type": role,
            "interactive": role == "menu_item",
            "label": text,
            "text": text,
            "x1": x - pad_x,
            "y1": y - pad_y,
            "x2": x + bw + pad_x,
            "y2": y + bh + pad_y,
            "score": float(min(0.95, 0.48 + 0.005 * max(0.0, conf - min_conf))),
        }
        clamp_box(candidate, img_w, img_h)
        if candidate["w"] < 5 or candidate["h"] < 7:
            continue
        out.append(candidate)
    return out

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
    """Non-maximum suppression. Uses vectorized numpy for large box counts."""
    if not boxes:
        return []
    n = len(boxes)
    if n <= 60:
        # Small count: simple loop is fast enough.
        boxes = sorted(boxes, key=lambda b: b.get("score", 1.0) * (b["area"] + 1), reverse=True)
        kept = []
        for b in boxes:
            if all(iou(b, k) < iou_thresh for k in kept):
                kept.append(b)
        return kept

    # Vectorized NMS for larger counts.
    x1 = np.array([b["x1"] for b in boxes], dtype=np.float64)
    y1 = np.array([b["y1"] for b in boxes], dtype=np.float64)
    x2 = np.array([b["x2"] for b in boxes], dtype=np.float64)
    y2 = np.array([b["y2"] for b in boxes], dtype=np.float64)
    areas = (x2 - x1) * (y2 - y1)
    scores = np.array([b.get("score", 1.0) * (b.get("area", 1) + 1) for b in boxes])
    order = scores.argsort()[::-1]

    keep_indices: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep_indices.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        iou_vals = inter / union
        order = rest[iou_vals < iou_thresh]

    return [boxes[i] for i in keep_indices]

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

def _normalize_role(raw_role: Any) -> str:
    key = str(raw_role or "").strip().lower().replace(" ", "_")
    if not key:
        return "other"
    return ROLE_CANONICAL_MAP.get(key, key if key in ROLE_CANONICAL_MAP.values() else "other")

def _normalize_label(label: Any) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    # Collapse whitespace and trim punctuation noise.
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-_:|")
    return text[:220]

def _estimate_retina_scale(img_w: int, img_h: int) -> float:
    # Fast heuristic for screenshot pixel ratio; override with env if known.
    forced = os.environ.get("VISION_PIXEL_RATIO", "").strip()
    if forced:
        try:
            val = float(forced)
            if 0.5 <= val <= 4.0:
                return val
        except Exception:
            pass
    if img_w >= 2500 or img_h >= 1600:
        return 2.0
    return 1.0

def _stable_id_for_box(box: Dict[str, Any], img_w: int, img_h: int) -> str:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    label = _normalize_label(box.get("label", box.get("text", ""))).lower()
    nx1 = int(round(10000.0 * float(box["x1"]) / float(max(1, img_w))))
    ny1 = int(round(10000.0 * float(box["y1"]) / float(max(1, img_h))))
    nx2 = int(round(10000.0 * float(box["x2"]) / float(max(1, img_w))))
    ny2 = int(round(10000.0 * float(box["y2"]) / float(max(1, img_h))))
    key = f"{role}|{label}|{nx1}|{ny1}|{nx2}|{ny2}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"el_{digest}"

def _is_probable_container_role(role: str) -> bool:
    return role in CONTAINER_ROLES

def _default_state_for_role(role: str) -> Dict[str, Optional[bool]]:
    return {
        "enabled": True,
        "selected": False if role in {"tab", "menu_item", "list_row", "tree_item"} else None,
        "checked": False if role in {"checkbox", "radio", "switch"} else None,
        "focused": None,
        "expanded": None if role not in {"tree_item", "menu_item"} else False,
    }

def _infer_keyboard_focus_hint(box: Dict[str, Any], img_w: int, img_h: int) -> Optional[str]:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    top_band = y2 <= int(0.22 * img_h)
    centered = abs((x1 + x2) / 2.0 - img_w / 2.0) <= img_w * 0.26
    if role in {"address_bar"}:
        return "Cmd+L likely focuses this field."
    if role in {"search_field"}:
        return "Cmd+F often focuses search in this app."
    if role in {"text_field", "combo_box", "dropdown"} and top_band and centered and bw > img_w * 0.24 and bh < img_h * 0.12:
        return "Cmd+L may focus the top address/search field."
    if role == "menu_item" and y2 <= int(0.07 * img_h):
        return "Ctrl+F2 focuses the menu bar."
    return None

def _is_box_pair_mergeable(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    role_a = _normalize_role(a.get("role", a.get("element_type", a.get("type"))))
    role_b = _normalize_role(b.get("role", b.get("element_type", b.get("type"))))
    if role_a != role_b:
        # Allow merging generic "button" with icon_button for mixed detectors.
        if {role_a, role_b} != {"button", "icon_button"}:
            return False
    ia = iou(a, b)
    if ia >= 0.55:
        return True
    # Allow containment merge when labels agree.
    label_a = _normalize_label(a.get("text", a.get("label", ""))).lower()
    label_b = _normalize_label(b.get("text", b.get("label", ""))).lower()
    if label_a and label_b and label_a == label_b:
        if contains(a, b, pad=3) or contains(b, a, pad=3):
            return True
    return False

def _merge_cluster_boxes(cluster: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
    if len(cluster) == 1:
        out = dict(cluster[0])
        clamp_box(out, img_w, img_h)
        return out
    weights = [max(0.15, float(b.get("score", 0.5))) for b in cluster]
    sw = float(sum(weights))
    x1 = int(round(sum(wt * float(b["x1"]) for wt, b in zip(weights, cluster)) / sw))
    y1 = int(round(sum(wt * float(b["y1"]) for wt, b in zip(weights, cluster)) / sw))
    x2 = int(round(sum(wt * float(b["x2"]) for wt, b in zip(weights, cluster)) / sw))
    y2 = int(round(sum(wt * float(b["y2"]) for wt, b in zip(weights, cluster)) / sw))

    out = dict(max(cluster, key=lambda b: float(b.get("score", 0.0))))
    out["x1"], out["y1"], out["x2"], out["y2"] = x1, y1, x2, y2
    out["score"] = float(min(1.0, 0.10 + sum(float(b.get("score", 0.5)) for b in cluster) / max(1.0, len(cluster))))
    out["source_votes"] = sorted({str(b.get("source", "unknown")) for b in cluster})

    # Prefer richer label text from OCR-backed or longer labels.
    labels = [_normalize_label(b.get("text", b.get("label", ""))) for b in cluster]
    labels = [t for t in labels if t]
    if labels:
        out["text"] = max(labels, key=len)
        out["label"] = out["text"]

    # Any interactive vote marks the merged box as interactive.
    out["interactive"] = bool(any(bool(b.get("interactive", False)) for b in cluster))
    role_votes = [_normalize_role(b.get("role", b.get("element_type", b.get("type", "other")))) for b in cluster]
    if role_votes:
        out["role"] = max(set(role_votes), key=lambda r: role_votes.count(r))
        out["element_type"] = out["role"]

    clamp_box(out, img_w, img_h)
    return out

def _merge_candidates_by_role(boxes: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: float(b.get("score", 0.0)), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []
    for b in ordered:
        merged = False
        for cl in clusters:
            if _is_box_pair_mergeable(b, cl[0]):
                cl.append(b)
                merged = True
                break
        if not merged:
            clusters.append([b])
    out = [_merge_cluster_boxes(cluster, img_w, img_h) for cluster in clusters]
    return out

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
# Colored UI control detection (HSV-based)
# -----------------------------
def _detect_colored_controls(img_bgr: np.ndarray, min_area: int = 400, max_area_ratio: float = 0.15) -> List[Dict[str, Any]]:
    """
    Detect distinctly colored UI controls (colored buttons, badges, tags, alerts)
    via HSV saturation analysis.  Pure edge detection often misses colored buttons
    with subtle borders (e.g. blue 'Submit', red 'Delete', green 'Confirm').
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Saturated pixels with reasonable brightness stand out from gray/white UI.
    mask = ((sat > 40) & (val > 50)).astype(np.uint8) * 255

    k_scale = max(0.6, min(w, h) / 1400.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (int(max(3, round(5 * k_scale))),
                                   int(max(3, round(3 * k_scale)))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen_area = w * h
    boxes: List[Dict[str, Any]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = bw * bh
        if a < min_area or a > max_area_ratio * screen_area:
            continue
        if bw < 15 or bh < 10:
            continue
        ar = bw / max(1, bh)
        if ar > 20 or ar < 0.05:
            continue

        region_sat = sat[y:y + bh, x:x + bw]
        sat_density = float((region_sat > 40).mean())
        if sat_density < 0.30:
            continue

        contour_area = float(cv2.contourArea(c))
        rectangularity = contour_area / (float(a) + 1e-6)

        score = (0.50 * sat_density
                 + 0.30 * min(1.0, rectangularity)
                 + 0.20 * (1.0 if 0.5 < ar < 8.0 else 0.3))

        boxes.append({
            "type": "colored_control",
            "source": "cv",
            "x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
            "score": float(score),
        })
    return boxes

def _detect_icon_like_controls(
    img_bgr: np.ndarray,
    min_area: int = 300,
    max_area_ratio: float = 0.03,
) -> List[Dict[str, Any]]:
    """
    Detect icon-like controls (often square/rounded-square app icons) so they are
    labeled individually instead of merged into a long strip container.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Icon tiles are usually colorful with a solid-ish body.
    mask = ((sat > 26) & (val > 38)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screen_area = float(max(1, w * h))
    out: List[Dict[str, Any]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        a = bw * bh
        if a < min_area or a > max_area_ratio * screen_area:
            continue
        if bw < 16 or bh < 16:
            continue
        ar = float(bw) / float(max(1, bh))
        if ar < 0.62 or ar > 1.60:
            continue

        contour_area = float(cv2.contourArea(c))
        rectness = contour_area / float(max(1, a))
        if rectness < 0.40:
            continue

        roi = img_bgr[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 35, 110)
        edge_density = float((edges > 0).mean())
        if edge_density < 0.015:
            continue

        mean_sat = float(np.mean(sat[y:y + bh, x:x + bw]))
        score = (
            0.36 * _clamp01((mean_sat - 26.0) / 80.0)
            + 0.28 * _clamp01((rectness - 0.40) / 0.45)
            + 0.20 * _clamp01((edge_density - 0.015) / 0.12)
            + 0.16 * (1.0 - min(1.0, abs(ar - 1.0) / 0.65))
        )
        out.append({
            "type": "ui_rect",
            "source": "cv_icon",
            "role": "icon_button",
            "element_type": "icon_button",
            "interactive": True,
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + bw),
            "y2": int(y + bh),
            "score": float(score),
        })
    return out

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

def prune_strip_group_boxes(
    boxes: List[Dict[str, Any]],
    strict: bool,
    pad: int = 3,
) -> List[Dict[str, Any]]:
    """
    Remove long strip/container boxes that collapse multiple icon-like controls
    into one group target.
    """
    if not boxes:
        return []
    drop_idx = set()
    for i, b in enumerate(boxes):
        bw = int(max(1, b.get("w", b["x2"] - b["x1"])))
        bh = int(max(1, b.get("h", b["y2"] - b["y1"])))
        ar = float(bw) / float(max(1, bh))
        role = _normalize_role(b.get("role", b.get("element_type", b.get("type"))))
        is_strip = ar >= (3.0 if strict else 3.6) or role in {"toolbar", "group", "panel"}
        if not is_strip:
            continue
        children = 0
        iconish_children = 0
        for j, c in enumerate(boxes):
            if i == j:
                continue
            if c.get("area", 0) >= b.get("area", 0):
                continue
            if not contains(b, c, pad=pad):
                continue
            children += 1
            cw = int(max(1, c.get("w", c["x2"] - c["x1"])))
            ch = int(max(1, c.get("h", c["y2"] - c["y1"])))
            car = float(cw) / float(max(1, ch))
            crole = _normalize_role(c.get("role", c.get("element_type", c.get("type"))))
            if 0.60 <= car <= 1.65 and (crole in {"icon_button", "dock_item", "button"} or str(c.get("source", "")) in {"cv_icon", "mac_rules"}):
                iconish_children += 1
        if iconish_children >= 2 or (children >= 4 and ar > 2.8):
            drop_idx.add(i)
    return [b for idx, b in enumerate(boxes) if idx not in drop_idx]

def compute_click_point(box, img_w, img_h):
    """
    Compute a safe click point inside a box.
    Uses role-aware interior targeting with OCR centroids when possible.
    """
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type"))))

    # Shrink box to avoid edges/borders
    pad = int(max(2, min(10, min(bw, bh) * 0.12)))
    ix1 = min(x2 - 1, max(x1, x1 + pad))
    iy1 = min(y2 - 1, max(y1, y1 + pad))
    ix2 = max(ix1 + 1, x2 - pad)
    iy2 = max(iy1 + 1, y2 - pad)

    # Default center click.
    cx = int(round((ix1 + ix2) / 2.0))
    cy = int(round((iy1 + iy2) / 2.0))

    # Left-biased click points are safer for menu/list rows and sidebar items.
    if role in {"menu_item", "list_row", "tree_item", "table_cell"}:
        cx = int(round(ix1 + 0.24 * (ix2 - ix1)))
        cy = int(round((iy1 + iy2) / 2.0))

    # Text fields are usually safest at left-mid area (away from clear buttons).
    if role in {"text_field", "search_field", "address_bar", "combo_box"}:
        cx = int(round(ix1 + 0.20 * (ix2 - ix1)))
        cy = int(round((iy1 + iy2) / 2.0))

    # Prefer OCR text centroid when available and inside bounds
    tx = box.get("text_cx")
    ty = box.get("text_cy")
    if tx is not None and ty is not None:
        if ix1 <= tx <= ix2 and iy1 <= ty <= iy2:
            if role in {"text_field", "search_field", "address_bar"}:
                # For fields we don't click directly on glyph center; shift slightly left.
                cx = int(max(ix1, min(ix2, int(tx) - max(2, bw // 12))))
                cy = int(ty)
            else:
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

## Florence-2 lazy model loader
_florence_state: Dict[str, Any] = {}

def _load_florence_model() -> bool:
    """Load Florence-2 model + processor on first call. Returns True if ready."""
    if _florence_state.get("ready"):
        return True
    if _florence_state.get("failed"):
        return False

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        _florence_state["failed"] = True
        return False

    model_name = (os.environ.get("VISION_FLORENCE_MODEL") or DEFAULT_FLORENCE_MODEL).strip()
    allow_download = os.environ.get("VISION_FLORENCE_ALLOW_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes", "on")

    try:
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=(not allow_download),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=(not allow_download),
        ).to(device)
        model.eval()

        _florence_state["model"] = model
        _florence_state["processor"] = processor
        _florence_state["device"] = device
        _florence_state["dtype"] = dtype
        _florence_state["ready"] = True
        return True
    except Exception as e:
        _florence_state["failed"] = True
        _florence_state["error"] = str(e)
        return False


def _runtime_default_backend() -> str:
    backend = os.environ.get("VISION_BACKEND", DEFAULT_BACKEND).strip().lower() or "auto"
    return backend if backend in {"auto", "cv"} else "auto"

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

def _box_mean_bgr(img_bgr: np.ndarray, box: Dict[str, Any]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    if x2 <= x1 or y2 <= y1:
        return None
    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return region.reshape(-1, 3).mean(axis=0)

def _infer_role_from_box(box: Dict[str, Any], img_h: int) -> str:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    text = _normalize_label(box.get("label", box.get("text", ""))).lower()
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    if role == "text":
        # Top thin text near menu area often corresponds to menu entries.
        if y2 <= int(0.07 * img_h) and 18 <= bw <= 280 and bh <= 40:
            role = "menu_item"
        # Link-like labels
        elif any(token in text for token in ("http", "www", ".com", "learn more", "details")):
            role = "link"

    # Disambiguate text-field vs button by geometry.
    if role == "button" and bw > bh * 4.8 and bh >= 18 and bh <= 90:
        if any(k in text for k in ("search", "find", "url", "address", "type here")):
            role = "search_field"
        elif bw > 220:
            role = "text_field"

    if role == "text_field" and y2 <= int(0.22 * img_h) and bw > 220:
        if any(t in text for t in ("http", ".com", "search", "address", "url")):
            role = "address_bar"

    return role

def _infer_state_from_visuals(img_bgr: np.ndarray, box: Dict[str, Any], role: str) -> Dict[str, Optional[bool]]:
    state = _default_state_for_role(role)
    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
    if x2 <= x1 or y2 <= y1:
        return state

    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return state

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mean_sat = float(np.mean(sat))
    mean_val = float(np.mean(val))

    # Disabled controls are often low-contrast/desaturated.
    if role in CLICKABLE_ROLES:
        if mean_sat < 14 and float(np.std(val)) < 12:
            state["enabled"] = False

    # Selected rows/tabs often carry accent fill (higher saturation but not too dark).
    if role in {"tab", "menu_item", "list_row", "tree_item"}:
        state["selected"] = bool(mean_sat > 24 and mean_val > 55)

    # Checked heuristics for checkbox/radio/switch.
    if role in {"checkbox", "radio", "switch"}:
        center = region[
            int(max(0, 0.2 * region.shape[0])):int(min(region.shape[0], 0.8 * region.shape[0])),
            int(max(0, 0.2 * region.shape[1])):int(min(region.shape[1], 0.8 * region.shape[1])),
        ]
        if center.size:
            c_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            c_sat = float(np.mean(c_hsv[:, :, 1]))
            c_val = float(np.mean(c_hsv[:, :, 2]))
            state["checked"] = bool(c_sat > 22 or c_val < 120)

    # Focus ring (macOS accent blue) often appears as saturated border around field.
    if role in {"text_field", "search_field", "address_bar"} and region.shape[0] >= 8 and region.shape[1] >= 20:
        top = region[0:2, :, :]
        bottom = region[-2:, :, :]
        left = region[:, 0:2, :]
        right = region[:, -2:, :]
        border = np.concatenate(
            [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
            axis=0,
        )
        hsv_b = cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        # Accent-like hues: blue/cyan range.
        accent = ((hsv_b[:, 0] >= 85) & (hsv_b[:, 0] <= 130) & (hsv_b[:, 1] >= 60))
        state["focused"] = bool(float(np.mean(accent)) > 0.15)

    return state

def _compute_clickability_score(
    img_bgr: np.ndarray,
    box: Dict[str, Any],
    role: str,
    screen_area: float,
) -> float:
    base = float(box.get("score", 0.50))
    evidence = _visual_evidence_score(img_bgr, box)
    area_ratio = float(box.get("area", 0)) / float(max(1.0, screen_area))
    has_text = bool(_normalize_label(box.get("label", box.get("text", ""))))
    interactive_hint = bool(box.get("interactive", False)) or role in CLICKABLE_ROLES
    container_penalty = 0.25 if _is_probable_container_role(role) else 0.0

    score = 0.35 * base + 0.25 * evidence
    if interactive_hint:
        score += 0.26
    if has_text:
        score += 0.10

    # Very tiny and very huge boxes are risky click targets.
    if area_ratio < 0.00003:
        score -= 0.22
    elif area_ratio < 0.00008:
        score -= 0.10
    if area_ratio > 0.18:
        score -= 0.18
    if area_ratio > 0.35:
        score -= 0.24

    score -= container_penalty
    return float(_clamp01(score))

def _enrich_semantics_and_state(
    img_bgr: np.ndarray,
    boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    h, w = img_bgr.shape[:2]
    screen_area = float(max(1, h * w))
    retina_scale = _estimate_retina_scale(w, h)

    enriched: List[Dict[str, Any]] = []
    for b in boxes:
        c = dict(b)
        role = _infer_role_from_box(c, img_h=h)
        c["role"] = role
        c["element_type"] = role
        c["label"] = _normalize_label(c.get("label", c.get("text", "")))
        c["text"] = c["label"] if c["label"] else c.get("text", "")

        interactive = bool(c.get("interactive", False) or role in CLICKABLE_ROLES)
        if _is_probable_container_role(role):
            interactive = False
        c["interactive"] = interactive

        state = _infer_state_from_visuals(img_bgr, c, role)
        c["state"] = state
        c["confidence"] = float(_clamp01(c.get("score", 0.0)))
        c["clickability"] = _compute_clickability_score(img_bgr, c, role, screen_area=screen_area)
        c["retina_scale"] = retina_scale
        c["keyboard_focus_hint"] = _infer_keyboard_focus_hint(c, w, h)
        c["stable_id"] = _stable_id_for_box(c, w, h)
        c["bounds"] = {
            "x": int(c["x1"]),
            "y": int(c["y1"]),
            "width": int(max(1, c["x2"] - c["x1"])),
            "height": int(max(1, c["y2"] - c["y1"])),
        }
        enriched.append(c)
    return enriched

def _drop_low_clickability_noise(
    boxes: List[Dict[str, Any]],
    strict: bool,
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    keep: List[Dict[str, Any]] = []
    threshold = 0.47 if strict else 0.36
    for b in boxes:
        role = _normalize_role(b.get("role", b.get("element_type", b.get("type"))))
        clickability = float(b.get("clickability", b.get("score", 0.0)))
        src = str(b.get("source", ""))
        label = _normalize_label(b.get("label", b.get("text", "")))
        if role in CONTAINER_ROLES and clickability < 0.80:
            continue
        if role == "text":
            # OCR fallback path: keep textual candidates for high-recall overlays.
            if src in {"ocr", "ocr_word"}:
                if len(label) <= 1 and clickability < 0.44:
                    continue
            elif clickability < (0.56 if strict else 0.44):
                continue
        if clickability < threshold and not b.get("interactive", False):
            continue
        keep.append(b)
    return keep

def _drop_large_detections(
    boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    strict: bool,
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    screen_area = float(max(1, img_w * img_h))
    max_area_ratio = float(os.environ.get("VISION_MAX_AREA_RATIO", str(DEFAULT_MAX_AREA_RATIO)))
    max_area_ratio = max(0.006, min(0.20, max_area_ratio))
    hard_ratio = max_area_ratio * (0.95 if strict else 1.10)
    out: List[Dict[str, Any]] = []
    for b in boxes:
        bw = int(max(1, b.get("w", b["x2"] - b["x1"])))
        bh = int(max(1, b.get("h", b["y2"] - b["y1"])))
        area_ratio = float(b.get("area", bw * bh)) / screen_area
        role = _normalize_role(b.get("role", b.get("element_type", b.get("type"))))
        # Drop only clearly oversized non-actionable containers.
        if area_ratio > hard_ratio and (not bool(b.get("interactive", False)) or role in CONTAINER_ROLES):
            continue
        if bw > int(img_w * 0.75) and bh > int(img_h * 0.18) and not bool(b.get("interactive", False)):
            continue
        if bh > int(img_h * 0.50) and not bool(b.get("interactive", False)):
            continue
        out.append(b)
    return out

def _dedupe_near_duplicate_boxes(
    boxes: List[Dict[str, Any]],
    strict: bool,
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    ordered = sorted(
        boxes,
        key=lambda b: (
            float(b.get("clickability", b.get("score", 0.0))),
            float(b.get("score", 0.0)),
            float(b.get("area", 0.0)),
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    dup_iou = 0.68 if strict else 0.74
    for b in ordered:
        role = _normalize_role(b.get("role", b.get("element_type", b.get("type"))))
        label = _normalize_label(b.get("label", b.get("text", ""))).lower()
        duplicate = False
        for k in kept:
            k_role = _normalize_role(k.get("role", k.get("element_type", k.get("type"))))
            if role != k_role:
                continue
            ov = iou(b, k)
            if ov >= dup_iou:
                duplicate = True
                break
            if label and label == _normalize_label(k.get("label", k.get("text", ""))).lower():
                if ov >= 0.45 or (contains(k, b, pad=3) or contains(b, k, pad=3)):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(b)
    return kept

def _detect_macos_traffic_lights(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    if h < 60 or w < 120:
        return []
    # Most app windows place controls in the top-left title bar area.
    rx2 = int(min(w, max(130, 0.24 * w)))
    ry2 = int(min(h, max(56, 0.10 * h)))
    roi = img_bgr[0:ry2, 0:rx2]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "window_control_close": cv2.inRange(hsv, np.array([0, 70, 60]), np.array([12, 255, 255])),
        "window_control_minimize": cv2.inRange(hsv, np.array([18, 70, 60]), np.array([38, 255, 255])),
        "window_control_zoom": cv2.inRange(hsv, np.array([40, 55, 55]), np.array([90, 255, 255])),
    }

    detections: List[Dict[str, Any]] = []
    for role, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 7 or bh < 7:
                continue
            if bw > 40 or bh > 40:
                continue
            ar = float(bw) / float(max(1, bh))
            if ar < 0.65 or ar > 1.45:
                continue
            circle_area = float(cv2.contourArea(c))
            rect_area = float(max(1, bw * bh))
            fill = circle_area / rect_area
            if fill < 0.45:
                continue
            score = fill * min(1.0, rect_area / 190.0)
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)
        if best is not None:
            x, y, bw, bh = best
            b = {
                "type": "vlm_element",
                "source": "mac_rules",
                "role": role,
                "element_type": role,
                "interactive": True,
                "label": role.replace("window_control_", "").replace("_", " "),
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + bw),
                "y2": int(y + bh),
                "score": float(min(0.98, 0.75 + 0.20 * best_score)),
            }
            clamp_box(b, w, h)
            detections.append(b)

    # Keep only plausible left-to-right triplets; otherwise avoid false positives.
    if len(detections) >= 2:
        detections.sort(key=lambda b: b["x1"])
        ys = [d["cy"] for d in detections]
        if max(ys) - min(ys) > 20:
            return []
    return detections

def _menu_items_from_ocr_words(words: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    if not words:
        return []
    top_limit = int(max(18, min(54, img_h * 0.07)))
    # macOS menu bar entries are short, top-aligned text fragments.
    top_words = [
        w for w in words
        if int(w.get("y", 10**9)) <= top_limit and float(w.get("conf", -1)) >= 55
    ]
    if not top_words:
        return []

    lines = _group_ocr_words_into_lines(top_words, min_conf=55)
    out: List[Dict[str, Any]] = []
    for line in lines:
        for w in line:
            text = _normalize_label(w.get("text", ""))
            if not text:
                continue
            # Keep menu-like tokens (short, starts uppercase in many apps, but tolerate symbols).
            if len(text) > 24:
                continue
            bw = int(w["w"])
            bh = int(w["h"])
            if bw < 10 or bh < 10:
                continue
            b = {
                "type": "vlm_element",
                "source": "ocr",
                "role": "menu_item",
                "element_type": "menu_item",
                "interactive": True,
                "label": text,
                "x1": int(w["x"] - max(2, bw * 0.20)),
                "y1": int(w["y"] - max(2, bh * 0.30)),
                "x2": int(w["x"] + bw + max(2, bw * 0.20)),
                "y2": int(w["y"] + bh + max(2, bh * 0.30)),
                "score": 0.68,
            }
            clamp_box(b, img_w, img_h)
            out.append(b)
    return out

def _dock_icon_candidates(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    if h < 200 or w < 300:
        return []
    y1 = int(h * 0.82)
    band = img_bgr[y1:h, :]
    if band.size == 0:
        return []
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 30, 95)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    out: List[Dict[str, Any]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 22 or bh < 22:
            continue
        if bw > 180 or bh > 180:
            continue
        ar = float(bw) / float(max(1, bh))
        if ar < 0.62 or ar > 1.45:
            continue
        area_ratio = float(bw * bh) / float(max(1, w * h))
        if area_ratio < 0.00015 or area_ratio > 0.020:
            continue
        roi_sat = hsv[y:y + bh, x:x + bw, 1]
        if roi_sat.size == 0:
            continue
        if float(np.mean(roi_sat)) < 22.0:
            continue
        contour_area = float(cv2.contourArea(c))
        rectness = contour_area / float(max(1, bw * bh))
        if rectness < 0.40:
            continue
        bx = {
            "type": "ui_rect",
            "source": "mac_rules",
            "role": "dock_item",
            "element_type": "dock_item",
            "interactive": True,
            "label": "",
            "x1": int(x),
            "y1": int(y + y1),
            "x2": int(x + bw),
            "y2": int(y + y1 + bh),
            "score": float(0.54 + 0.26 * _clamp01((rectness - 0.40) / 0.45)),
        }
        clamp_box(bx, w, h)
        out.append(bx)
    # Dock heuristics can be noisy; keep only strongest and horizontally distributed.
    out = nms(out, iou_thresh=0.26)
    out = sorted(out, key=lambda b: (b["score"], b["area"]), reverse=True)
    filtered: List[Dict[str, Any]] = []
    for b in out:
        if len(filtered) >= 20:
            break
        if any(abs(b["cx"] - f["cx"]) < 14 and abs(b["cy"] - f["cy"]) < 14 for f in filtered):
            continue
        filtered.append(b)
    return filtered

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

def detect_ui_elements_vlm(
    img_bgr: np.ndarray,
    strict: bool = False,
    max_elements: int = 220,
) -> List[Dict[str, Any]]:
    return []

# -----------------------------------------------
# Florence-2 local vision model detector
# -----------------------------------------------

_FLORENCE_LABEL_MAP = {
    "button": "button",
    "text field": "textfield",
    "search bar": "textfield",
    "address bar": "address_bar",
    "checkbox": "checkbox",
    "toggle switch": "switch",
    "toggle": "switch",
    "tab": "tab",
    "menu item": "menuitem",
    "dropdown": "dropdown",
    "link": "link",
    "icon": "iconbutton",
    "slider": "slider",
    "list item": "listitem",
    "toolbar button": "iconbutton",
    "radio button": "radio",
    "window close button": "window_control_close",
    "window minimize button": "window_control_minimize",
    "window zoom button": "window_control_zoom",
    "sidebar item": "listitem",
    "table row": "row",
}

def _florence_text_prompt() -> str:
    return (
        "button. icon button. toolbar button. text field. search bar. address bar. "
        "checkbox. radio button. toggle switch. tab. menu item. dropdown. combo box. "
        "link. slider. list item. table row. sidebar item. window close button. "
        "window minimize button. window zoom button."
    )

def _run_florence_phrase_grounding(
    img_bgr: np.ndarray,
    model: Any,
    processor: Any,
    device: str,
    dtype: Any,
    max_dim: int,
) -> List[Tuple[List[float], str]]:
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError:
        return []

    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return []
    scale = min(1.0, float(max_dim) / float(max(1, max(h, w))))
    if scale < 1.0:
        resized = cv2.resize(
            img_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = img_bgr

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = _florence_text_prompt()

    inputs = processor(
        text=task_prompt + text_input,
        images=pil_img,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(dtype),
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(pil_img.width, pil_img.height),
    )
    grounding = result.get(task_prompt, {})
    bboxes = grounding.get("bboxes", []) or []
    labels = grounding.get("labels", []) or []
    out: List[Tuple[List[float], str]] = []
    for bbox, label in zip(bboxes, labels):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        fx1, fy1, fx2, fy2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if scale < 1.0:
            fx1 /= scale
            fy1 /= scale
            fx2 /= scale
            fy2 /= scale
        out.append(([fx1, fy1, fx2, fy2], str(label or "")))
    return out

def _make_overlapping_tiles(img_w: int, img_h: int, grid: int = 2, overlap: float = 0.20) -> List[Tuple[int, int, int, int]]:
    grid = max(1, int(grid))
    overlap = float(max(0.05, min(0.45, overlap)))
    if grid <= 1:
        return [(0, 0, img_w, img_h)]

    tile_w = int(math.ceil(img_w / float(grid)))
    tile_h = int(math.ceil(img_h / float(grid)))
    ov_w = int(round(tile_w * overlap))
    ov_h = int(round(tile_h * overlap))

    tiles: List[Tuple[int, int, int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            x1 = max(0, gx * tile_w - ov_w)
            y1 = max(0, gy * tile_h - ov_h)
            x2 = min(img_w, (gx + 1) * tile_w + ov_w)
            y2 = min(img_h, (gy + 1) * tile_h + ov_h)
            if x2 - x1 < 60 or y2 - y1 < 60:
                continue
            tiles.append((x1, y1, x2, y2))
    return tiles

def _florence_predictions_to_boxes(
    preds: List[Tuple[List[float], str]],
    full_w: int,
    full_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    source_stage: str = "full",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    screen_area = float(max(1, full_w * full_h))
    for bbox, label in preds:
        fx1, fy1, fx2, fy2 = bbox
        x1i = int(round(min(fx1, fx2))) + int(offset_x)
        y1i = int(round(min(fy1, fy2))) + int(offset_y)
        x2i = int(round(max(fx1, fx2))) + int(offset_x)
        y2i = int(round(max(fy1, fy2))) + int(offset_y)
        candidate = {"x1": x1i, "y1": y1i, "x2": x2i, "y2": y2i}
        clamp_box(candidate, full_w, full_h)
        if candidate["w"] < 6 or candidate["h"] < 6:
            continue

        area_ratio = float(candidate["area"]) / screen_area
        if area_ratio > 0.14:
            continue
        ar = float(candidate["w"]) / float(max(1, candidate["h"]))
        if ar < 1 / 24 or ar > 24:
            continue

        raw = str(label or "").strip()
        label_lower = raw.lower()
        element_type = _FLORENCE_LABEL_MAP.get(label_lower, "button")
        role = _normalize_role(element_type)
        is_container = role in CONTAINER_ROLES
        interactive = role in CLICKABLE_ROLES

        candidate.update(
            {
                "type": "vlm_element",
                "source": "vlm",
                "detection_stage": f"florence_{source_stage}",
                "element_type": role,
                "role": role,
                "is_container": is_container,
                "interactive": interactive,
                "text": raw[:200],
                "label": raw[:200],
                "score": 0.80 if source_stage == "full" else 0.76,
            }
        )
        out.append(candidate)
    return out

def _detect_ui_florence(img_bgr: np.ndarray, max_elements: int = 220) -> List[Dict[str, Any]]:
    """Run Florence-2 phrase grounding with full-image + tiled passes."""
    use_florence = os.environ.get(
        "VISION_USE_FLORENCE", "1" if DEFAULT_FLORENCE_ENABLED else "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    if not use_florence:
        return []
    if not _load_florence_model():
        return []

    model = _florence_state["model"]
    processor = _florence_state["processor"]
    device = _florence_state["device"]
    dtype = _florence_state["dtype"]

    h, w = img_bgr.shape[:2]
    all_boxes: List[Dict[str, Any]] = []

    # Coarse pass on the full screenshot.
    full_preds = _run_florence_phrase_grounding(img_bgr, model, processor, device, dtype, max_dim=1344)
    all_boxes.extend(_florence_predictions_to_boxes(full_preds, full_w=w, full_h=h, source_stage="full"))

    # Refinement pass on overlapping tiles for tiny controls/icons.
    use_tiled = os.environ.get("VISION_FLORENCE_TILED", "1").strip().lower() in ("1", "true", "yes", "on")
    if use_tiled and max(h, w) >= 1200:
        try:
            tile_grid = int(os.environ.get("VISION_FLORENCE_TILE_GRID", "2"))
        except Exception:
            tile_grid = 2
        try:
            tile_overlap = float(os.environ.get("VISION_FLORENCE_TILE_OVERLAP", "0.20"))
        except Exception:
            tile_overlap = 0.20
        tiles = _make_overlapping_tiles(w, h, grid=tile_grid, overlap=tile_overlap)
        for tx1, ty1, tx2, ty2 in tiles:
            tile = img_bgr[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue
            tile_preds = _run_florence_phrase_grounding(tile, model, processor, device, dtype, max_dim=1024)
            all_boxes.extend(
                _florence_predictions_to_boxes(
                    tile_preds,
                    full_w=w,
                    full_h=h,
                    offset_x=tx1,
                    offset_y=ty1,
                    source_stage="tile",
                )
            )

    if not all_boxes:
        return []

    # Merge duplicates from full/tile predictions before hard filtering.
    all_boxes = _merge_candidates_by_role(all_boxes, w, h)

    cv_hints = detect_ui_rectangles(img_bgr, scale=1.0, min_score=0.30)
    text_hints = detect_text_regions(img_bgr)
    cv_hints.extend([t for t in text_hints if float(t.get("score", 0.0)) >= 0.42])
    for c in cv_hints:
        clamp_box(c, w, h)
    cv_hints = [c for c in cv_hints if c.get("area", 0) > 0]

    all_boxes = _validate_vlm_boxes(img_bgr, all_boxes, strict=True, cv_hints=cv_hints)
    all_boxes = _prune_vlm_container_candidates(all_boxes, w, h, strict=True)
    all_boxes = nms(all_boxes, iou_thresh=0.28)
    all_boxes = prune_contained(all_boxes, containment_iou=0.90)

    # Keep strongest actionable targets first.
    all_boxes.sort(
        key=lambda b: (
            float(b.get("score", 0.0))
            + (0.35 if b.get("interactive", False) else 0.0)
            + (0.10 if _normalize_label(b.get("text", "")) else 0.0),
            -float(b.get("area", 0.0)),
        ),
        reverse=True,
    )
    return all_boxes[:max_elements]


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
    if backend not in {"auto", "cv"}:
        backend = "auto"
    strict = _runtime_default_strict() if strict is None else bool(strict)
    max_boxes = _runtime_default_max_boxes() if max_boxes is None else int(max_boxes)
    min_score = _runtime_default_min_score() if min_score is None else float(min_score)
    cv_supplement = os.environ.get("VISION_CV_SUPPLEMENT", "1" if DEFAULT_CV_SUPPLEMENT else "0").strip().lower() in ("1", "true", "yes", "on")
    use_ocr_hints = os.environ.get("VISION_USE_OCR_HINTS", "1").strip().lower() in ("1", "true", "yes", "on")
    use_mac_rules = os.environ.get("VISION_USE_MAC_RULES", "1").strip().lower() in ("1", "true", "yes", "on")
    use_ocr_word_candidates = os.environ.get("VISION_OCR_WORD_CANDIDATES", "1").strip().lower() in ("1", "true", "yes", "on")

    def _max_iou_with(box: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
        if not others:
            return 0.0
        best = 0.0
        for o in others:
            best = max(best, iou(box, o))
            if best >= 0.90:
                return best
        return best

    # Stage A: semantic detections (Florence-2) in auto mode.
    vlm_boxes: List[Dict[str, Any]] = []
    if backend == "auto":
        try:
            vlm_boxes = _detect_ui_florence(img_bgr, max_elements=max_boxes)
        except Exception:
            vlm_boxes = []

    # Stage B: CV and OCR supplements (for tiny controls, dense rows, unlabeled icons).
    cv_boxes: List[Dict[str, Any]] = []
    if backend in ("auto", "cv"):
        weak_vlm = len(vlm_boxes) < (10 if strict else 5)
        run_cv = (backend == "cv") or weak_vlm or (backend == "auto" and cv_supplement)
        if run_cv:
            rect_boxes: List[Dict[str, Any]] = []
            for s in rect_scales:
                rect_boxes.extend(detect_ui_rectangles(img_bgr, scale=s, min_score=min_score))

            text_boxes = detect_text_regions(img_bgr)
            text_min = 0.34 if strict else min(0.42, float(min_score))
            text_boxes = [b for b in text_boxes if float(b.get("score", 0.0)) >= text_min]

            color_boxes = _detect_colored_controls(img_bgr)
            color_boxes = [b for b in color_boxes if float(b.get("score", 0.0)) >= (0.40 if strict else 0.42)]
            icon_boxes = _detect_icon_like_controls(img_bgr)
            icon_boxes = [b for b in icon_boxes if float(b.get("score", 0.0)) >= (0.24 if strict else 0.28)]
            cv_boxes = rect_boxes + text_boxes + color_boxes + icon_boxes

    # OCR line detection for text+icon combos and menu/list density.
    ocr_words: List[Dict[str, Any]] = []
    ocr_line_boxes: List[Dict[str, Any]] = []
    ocr_word_boxes: List[Dict[str, Any]] = []
    if use_ocr_hints:
        ocr_words = try_ocr_words(img_bgr)
        if ocr_words:
            ocr_line_boxes = _ocr_line_boxes(ocr_words, w, h, min_conf=(52 if strict else 45))
            if use_ocr_word_candidates and (not vlm_boxes or len(vlm_boxes) < 10):
                # Dense fallback when semantic detector is missing/weak.
                ocr_word_boxes = _ocr_word_boxes(ocr_words, w, h, min_conf=(58 if strict else 50))

    # macOS-specific rules.
    mac_boxes: List[Dict[str, Any]] = []
    if use_mac_rules:
        mac_boxes.extend(_detect_macos_traffic_lights(img_bgr))
        if ocr_words:
            mac_boxes.extend(_menu_items_from_ocr_words(ocr_words, w, h))
        mac_boxes.extend(_dock_icon_candidates(img_bgr))

    # Stage C: merge candidates.
    boxes: List[Dict[str, Any]] = vlm_boxes + cv_boxes + ocr_line_boxes + ocr_word_boxes + mac_boxes
    for b in boxes:
        clamp_box(b, w, h)
    boxes = [b for b in boxes if b.get("area", 0) > 0]

    # If semantic detections exist, suppress heavily overlapping CV text noise.
    if vlm_boxes:
        vlm_clamped = [b for b in boxes if b.get("source") == "vlm"]
        kept: List[Dict[str, Any]] = []
        for b in boxes:
            src = str(b.get("source", ""))
            if src in {"cv", "ocr"} and _max_iou_with(b, vlm_clamped) >= 0.45:
                # Preserve OCR words if they add label text to unlabeled VLM candidates.
                if src == "ocr" and _normalize_label(b.get("text", "")):
                    kept.append(b)
                continue
            kept.append(b)
        boxes = kept

    # Merge by role/label before NMS to reduce duplicates from tiled Florence.
    boxes = _merge_candidates_by_role(boxes, w, h)
    boxes = nms(boxes, iou_thresh=(0.42 if strict else max(float(nms_iou), 0.40)))
    boxes = prune_contained(boxes, containment_iou=(0.92 if strict else containment_iou))
    boxes = prune_text_regions_inside_rects(boxes, max_area_ratio=5.0, pad=3)
    boxes = prune_container_boxes(boxes, min_children=3, min_area=7600, max_child_area_ratio=0.6, pad=4)
    boxes = prune_strip_group_boxes(boxes, strict=strict, pad=3)
    boxes = filter_overlapping_small_boxes(boxes, min_area_ratio=0.12)

    # Stage D: refine geometry.
    if refine_boxes:
        for b in boxes:
            src = str(b.get("source", ""))
            if src in {"vlm", "mac_rules"}:
                continue
            if 80 < int(b.get("area", 0)) < 600000:
                refine_box_to_content(img_bgr, b, padding=2)
                clamp_box(b, w, h)

    # Stage E: OCR association, semantics, state, clickability.
    if ocr_words:
        assign_words_to_boxes(boxes, ocr_words, min_conf=(52 if strict else 45))
    add_click_points(boxes, w, h)
    boxes = _enrich_semantics_and_state(img_bgr, boxes)
    boxes = _drop_low_clickability_noise(boxes, strict=strict)
    boxes = _drop_large_detections(boxes, w, h, strict=strict)
    boxes = _dedupe_near_duplicate_boxes(boxes, strict=strict)

    # If semantic backend is unavailable, preserve OCR words for visibility/debugging.
    if not vlm_boxes and ocr_word_boxes:
        seen_ids = {str(b.get("stable_id", "")) for b in boxes}
        for wb in _enrich_semantics_and_state(img_bgr, ocr_word_boxes):
            sid = str(wb.get("stable_id", ""))
            if sid and sid in seen_ids:
                continue
            seen_ids.add(sid)
            boxes.append(wb)
        boxes = _dedupe_near_duplicate_boxes(boxes, strict=strict)

    # Recompute click points now that role labels are enriched.
    add_click_points(boxes, w, h)

    # Stage F: cap outputs by clickability and semantic confidence.
    if max_boxes and len(boxes) > max_boxes:
        def _rank_key(b: Dict[str, Any]) -> Tuple[float, float]:
            score = float(b.get("clickability", b.get("score", 0.0)))
            if b.get("interactive", False):
                score += 0.36
            if b.get("source") == "vlm":
                score += 0.20
            if _normalize_label(b.get("label", b.get("text", ""))):
                score += 0.12
            area_pen = float(b.get("area", 0)) / float(max(w * h, 1))
            # Strongly prioritize smaller actionable targets.
            small_bonus = 0.12 if area_pen < 0.0035 else (0.05 if area_pen < 0.010 else 0.0)
            return (score + small_bonus - 0.45 * area_pen, -area_pen)
        boxes = sorted(boxes, key=_rank_key, reverse=True)[:max_boxes]

    # Stable reading order + stable IDs.
    boxes.sort(key=lambda b: (b["y1"], b["x1"], -float(b.get("score", 0.0))))
    for i, b in enumerate(boxes):
        b["id"] = i
        if not b.get("stable_id"):
            b["stable_id"] = _stable_id_for_box(b, w, h)
        b["click_point"] = {"x": int(b.get("click_x", b.get("cx", 0))), "y": int(b.get("click_y", b.get("cy", 0)))}
        b["role"] = _normalize_role(b.get("role", b.get("element_type", b.get("type", "other"))))
        b["element_type"] = b["role"]
        b["is_actionable"] = bool(b.get("interactive", False) and b.get("role") in CLICKABLE_ROLES)
        b["confidence"] = float(_clamp01(b.get("confidence", b.get("score", 0.0))))
        b["label"] = _normalize_label(b.get("label", b.get("text", "")))
        if b["label"]:
            b["text"] = b["label"]
        if "state" not in b:
            b["state"] = _default_state_for_role(b["role"])
        if b.get("label") and "label_confidence" not in b:
            b["label_confidence"] = _label_confidence_from_source(b, b.get("label", ""))
        risk_level, do_not_click, reason_codes = _assess_action_risk(b, w, h)
        b["risk_level"] = risk_level
        b["do_not_click"] = bool(do_not_click)
        b["reason_codes"] = reason_codes

    return boxes

def _get_type_color(box: Dict[str, Any]) -> Tuple[int, int, int]:
    """Get the BGR color for a box's element type."""
    et = str(box.get("element_type", box.get("type", ""))).strip().lower()
    return TYPE_PALETTE.get(et, _DEFAULT_COLOR)

def _text_color_for_bg(bg_color: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Return white or black text depending on background luminance."""
    b, g, r = int(bg_color[0]), int(bg_color[1]), int(bg_color[2])
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if lum < 160 else (0, 0, 0)

def _srgb_to_linear(v: int) -> float:
    x = float(max(0, min(255, v))) / 255.0
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4

def _relative_luminance_bgr(color: Tuple[int, int, int]) -> float:
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    rl = _srgb_to_linear(r)
    gl = _srgb_to_linear(g)
    bl = _srgb_to_linear(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl

def _contrast_ratio_bgr(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    la = _relative_luminance_bgr(a)
    lb = _relative_luminance_bgr(b)
    lighter = max(la, lb)
    darker = min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)

def _pick_wcag_text_color(bg: Tuple[int, int, int], prefer_red: bool = True) -> Tuple[int, int, int]:
    # Prefer bright red for labels when it remains accessible.
    candidates: List[Tuple[int, int, int]] = []
    if prefer_red:
        candidates.append((0, 0, 255))   # red
    candidates.extend([(255, 255, 255), (0, 0, 0)])  # white / black
    best = candidates[0]
    best_ratio = 0.0
    for c in candidates:
        ratio = _contrast_ratio_bgr(c, bg)
        if ratio >= 4.5:
            return c
        if ratio > best_ratio:
            best = c
            best_ratio = ratio
    return best

def _darken(color: Tuple[int, ...], factor: float = 0.6) -> Tuple[int, int, int]:
    """Darken a BGR color."""
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))

def _draw_legend(img: np.ndarray, active_types: set, font: int,
                 font_scale: float, thickness: int) -> np.ndarray:
    """Draw a compact legend bar at the bottom showing active element types."""
    if not active_types:
        return img
    h, w = img.shape[:2]

    swatch_size = max(10, int(16 * font_scale))
    item_gap = max(10, int(14 * font_scale))
    text_gap = max(4, int(6 * font_scale))
    bar_pad_x = max(10, int(14 * font_scale))
    bar_pad_y = max(6, int(8 * font_scale))
    legend_font_scale = font_scale * 0.65

    items = []
    for t in sorted(active_types):
        name = TYPE_DISPLAY_NAMES.get(t, t.replace("_", " ").title())
        color = TYPE_PALETTE.get(t, _DEFAULT_COLOR)
        (tw, th), _ = cv2.getTextSize(name, font, legend_font_scale, max(1, thickness - 1))
        items.append((t, name, color, tw, th))

    if not items:
        return img

    max_text_h = max(th for _, _, _, _, th in items)
    row_h = max(max_text_h, swatch_size) + 2 * bar_pad_y

    # Wrap items into rows that fit the image width.
    rows: List[list] = []
    current_row: list = []
    current_w = bar_pad_x
    for item in items:
        item_w = swatch_size + text_gap + item[3] + item_gap
        if current_w + item_w > w - bar_pad_x and current_row:
            rows.append(current_row)
            current_row = [item]
            current_w = bar_pad_x + item_w
        else:
            current_row.append(item)
            current_w += item_w
    if current_row:
        rows.append(current_row)

    total_bar_h = row_h * len(rows) + bar_pad_y

    # Extend image with legend area.
    legend_img = np.zeros((h + total_bar_h, w, 3), dtype=np.uint8)
    legend_img[:h, :] = img
    legend_img[h:, :] = (30, 30, 30)

    for row_idx, row in enumerate(rows):
        x = bar_pad_x
        y_base = h + bar_pad_y + row_idx * row_h
        for _t, name, color, tw, th in row:
            sy = y_base + (row_h - 2 * bar_pad_y - swatch_size) // 2
            cv2.rectangle(legend_img, (x, sy), (x + swatch_size, sy + swatch_size), color, -1)
            cv2.rectangle(legend_img, (x, sy), (x + swatch_size, sy + swatch_size), (80, 80, 80), 1)
            tx = x + swatch_size + text_gap
            ty = sy + swatch_size - max(1, (swatch_size - th) // 2)
            cv2.putText(legend_img, name, (tx, ty), font, legend_font_scale,
                        (220, 220, 220), max(1, thickness - 1), cv2.LINE_AA)
            x += swatch_size + text_gap + tw + item_gap

    return legend_img

def draw(img_bgr, boxes, show_click_points=False, show_legend=True):
    """
    Professional-grade annotated visualization of detected UI elements.

    Features:
      - Color-coded semi-transparent overlays per element type
      - Compact pill labels with IDs (color encodes the type)
      - Smart label placement avoiding overlaps
      - Crosshair click-point markers
      - Optional legend bar mapping colors to element types
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if not boxes:
        return out

    # Adaptive scale factors for resolution independence.
    base = max(1, min(w, h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.32, min(0.95, base / 1750.0))
    text_thickness = 1
    rect_thickness = max(2, int(round(font_scale * 2.2)))
    pad_x = max(3, int(round(font_scale * 5)))
    pad_y = max(2, int(round(font_scale * 4)))

    # --- Phase 1: Role-aware borders (regular red, OCR green thinner) ---
    for b in boxes:
        if _is_ocr_box(b):
            border_color = (0, 255, 0)  # green in BGR
            thickness = max(1, rect_thickness - 1)
        else:
            border_color = (0, 0, 255)  # red in BGR
            thickness = rect_thickness
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), border_color, thickness)

    # --- Phase 2: External numeric labels with leader lines ---
    placed_labels: List[Tuple[int, int, int, int]] = []

    def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _overlaps_any(rect: Tuple[int, int, int, int], rects: List[Tuple[int, int, int, int]]) -> bool:
        for r in rects:
            if _rects_overlap(rect, r):
                return True
        return False

    def _overlaps_box(rect: Tuple[int, int, int, int], bx1: int, by1: int, bx2: int, by2: int) -> bool:
        return not (rect[2] <= bx1 or rect[0] >= bx2 or rect[3] <= by1 or rect[1] >= by2)

    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = _get_type_color(b)

        label = str(b.get("id", ""))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        base_scale = max(0.52, min(1.05, font_scale * 1.10))
        gap = max(4, int(round(font_scale * 6)))

        placed = False
        for scale_factor in (1.0, 0.90, 0.82):
            local_scale = float(base_scale * scale_factor)
            (tw, th), _ = cv2.getTextSize(label, font, local_scale, max(1, text_thickness + 1))
            chip_w = int(tw + 4 * pad_x)
            chip_h = int(th + 4 * pad_y)

            candidates: List[Tuple[int, int]] = [
                (x1, y1 - chip_h - gap),
                (x2 - chip_w, y1 - chip_h - gap),
                (x2 + gap, y1),
                (x2 + gap, y2 - chip_h),
                (x1 - chip_w - gap, y1),
                (x1 - chip_w - gap, y2 - chip_h),
                (x1, y2 + gap),
                (x2 - chip_w, y2 + gap),
            ]

            # Spiral-ish offsets around the box for tougher placements.
            step = int(max(6, gap))
            for radius in (step, step * 2, step * 3, step * 4):
                candidates.extend(
                    [
                        (x1 - chip_w - gap - radius, y1 - chip_h - gap - radius),
                        (x2 + gap + radius, y1 - chip_h - gap - radius),
                        (x2 + gap + radius, y2 + gap + radius),
                        (x1 - chip_w - gap - radius, y2 + gap + radius),
                    ]
                )

            for cx, cy in candidates:
                lx1 = int(cx)
                ly1 = int(cy)
                lx2 = int(lx1 + chip_w)
                ly2 = int(ly1 + chip_h)
                rect = (lx1, ly1, lx2, ly2)
                if lx1 < 0 or ly1 < 0 or lx2 > w or ly2 > h:
                    continue
                if _overlaps_box(rect, x1, y1, x2, y2):
                    continue
                if _overlaps_any(rect, placed_labels):
                    continue
                placed = True
                break

            if placed:
                break

        if not placed:
            # Final fallback: scan for a free spot in a top-down grid.
            local_scale = base_scale * 0.82
            (tw, th), _ = cv2.getTextSize(label, font, local_scale, max(1, text_thickness + 1))
            chip_w = int(tw + 4 * pad_x)
            chip_h = int(th + 4 * pad_y)
            for gy in range(gap, max(gap + 1, h - chip_h), chip_h + gap):
                for gx in range(gap, max(gap + 1, w - chip_w), chip_w + gap):
                    rect = (gx, gy, gx + chip_w, gy + chip_h)
                    if _overlaps_box(rect, x1, y1, x2, y2):
                        continue
                    if _overlaps_any(rect, placed_labels):
                        continue
                    lx1, ly1, lx2, ly2 = rect
                    placed = True
                    break
                if placed:
                    break

        if not placed:
            # If we truly cannot place, skip the label to avoid overlaps.
            continue

        placed_labels.append((lx1, ly1, lx2, ly2))

        # Leader line from box edge to label chip.
        anchor_x = int(max(x1, min(x2, lx1 + (lx2 - lx1) // 2)))
        anchor_y = int(max(y1, min(y2, ly1 + (ly2 - ly1) // 2)))
        box_anchor_x = int(max(x1, min(x2, anchor_x)))
        box_anchor_y = int(max(y1, min(y2, anchor_y)))
        cv2.line(out, (box_anchor_x, box_anchor_y), (anchor_x, anchor_y), color, max(1, rect_thickness - 1))

        # High-contrast chip for index number (outside the box).
        chip = out.copy()
        chip_bg = (18, 18, 18)
        cv2.rectangle(chip, (lx1, ly1), (lx2, ly2), chip_bg, -1)
        cv2.addWeighted(chip, 0.90, out, 0.10, 0, out)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2)
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 0, 0), 1)
        tx = int(lx1 + max(1, (lx2 - lx1 - tw) // 2))
        ty = int(ly1 + max(th, (ly2 - ly1 + th) // 2 - 1))
        text_col = _pick_wcag_text_color(chip_bg, prefer_red=False)
        cv2.putText(out, label, (tx, ty), font, local_scale, text_col, 2, cv2.LINE_AA)
    # --- Phase 3: Legend bar ---
    if show_legend and boxes:
        active_types: set = set()
        for b in boxes:
            et = str(b.get("element_type", b.get("type", ""))).strip().lower()
            active_types.add(et)
        out = _draw_legend(out, active_types, font, font_scale, text_thickness)

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
    words_sorted = sorted(
        [w for w in words if float(w.get("conf", -1)) >= float(min_conf)],
        key=lambda w: (int(w["y"]), int(w["x"])),
    )
    if not words_sorted:
        return

    for b in boxes:
        bx1, by1, bx2, by2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        # Expanded association box links nearby text labels (icon + text patterns).
        ex = int(max(4, round(bw * 0.25)))
        ey = int(max(4, round(bh * 0.35)))
        ax1, ay1, ax2, ay2 = bx1 - ex, by1 - ey, bx2 + ex, by2 + ey

        collected: List[Tuple[int, int, str]] = []
        cxs: List[float] = []
        cys: List[float] = []
        confs: List[float] = []
        for w in words_sorted:
            cx = float(w["x"]) + float(w["w"]) / 2.0
            cy = float(w["y"]) + float(w["h"]) / 2.0
            if not (ax1 <= cx <= ax2 and ay1 <= cy <= ay2):
                continue

            # Prefer direct containment, but allow close neighbors for icon+label controls.
            inside = (bx1 <= cx <= bx2 and by1 <= cy <= by2)
            if not inside:
                dx = max(bx1 - cx, 0.0, cx - bx2)
                dy = max(by1 - cy, 0.0, cy - by2)
                if (dx * dx + dy * dy) > float(max(36, (max(bw, bh) * 0.6) ** 2)):
                    continue

            txt = _normalize_label(w.get("text", ""))
            if not txt:
                continue
            collected.append((int(w["y"]), int(w["x"]), txt))
            cxs.append(cx)
            cys.append(cy)
            try:
                confs.append(float(w.get("conf", 0.0)))
            except Exception:
                pass

        if collected:
            collected.sort()
            text = " ".join(t for _, _, t in collected)
            b["text"] = text
            b["label"] = text
            b["text_cx"] = int(round(sum(cxs) / len(cxs)))
            b["text_cy"] = int(round(sum(cys) / len(cys)))
            if confs:
                avg_conf = float(sum(confs) / max(1, len(confs)))
                b["label_confidence"] = float(_clamp01(avg_conf / 100.0))
def create_mser():
    mser = cv2.MSER_create()
    try:
        mser.setDelta(5)
        mser.setMinArea(30)
        mser.setMaxArea(8000)
    except Exception:
        pass
    return mser

def _normalize_state_dict(state: Optional[Dict[str, Any]], role: str) -> Dict[str, Optional[bool]]:
    base = _default_state_for_role(role)
    if not isinstance(state, dict):
        return base
    for k in ("enabled", "selected", "checked", "focused", "expanded"):
        if k in state:
            v = state.get(k)
            if isinstance(v, bool) or v is None:
                base[k] = v
            else:
                base[k] = bool(v)
    return base

def _label_confidence_from_source(box: Dict[str, Any], label: str) -> float:
    if not label:
        return 0.0
    src = str(box.get("source", ""))
    if src == "vlm":
        return 0.62
    if src == "mac_rules":
        return 0.58
    if src in {"ocr", "ocr_word"}:
        return 0.56
    if src == "cv":
        return 0.46
    return 0.50

def _is_ocr_box(box: Dict[str, Any]) -> bool:
    src = str(box.get("source", ""))
    return src in {"ocr", "ocr_word"}

def _assess_action_risk(
    box: Dict[str, Any],
    img_w: int,
    img_h: int,
) -> Tuple[str, bool, List[str]]:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", x1 + 1))
    y2 = int(box.get("y2", y1 + 1))
    area = float(box.get("area", max(1, (x2 - x1) * (y2 - y1))))
    area_ratio = area / float(max(1, img_w * img_h))
    label = _normalize_label(box.get("label", box.get("text", "")))
    clickability = float(box.get("clickability", box.get("score", 0.0)))
    interactive = bool(box.get("interactive", False))
    state = box.get("state", {}) if isinstance(box.get("state"), dict) else {}

    reasons: List[str] = []
    if _is_probable_container_role(role):
        reasons.append("container_role")
    if not label:
        reasons.append("no_label")
    if area_ratio < 0.00006:
        reasons.append("very_small")
    elif area_ratio < 0.00010:
        reasons.append("small")
    if area_ratio > 0.18:
        reasons.append("very_large")
    elif area_ratio > 0.12:
        reasons.append("large")
    if clickability < 0.45:
        reasons.append("low_clickability")
    if state.get("enabled") is False:
        reasons.append("disabled")
    if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
        reasons.append("edge_proximity")

    do_not_click = False
    if _is_probable_container_role(role) and not interactive:
        do_not_click = True
    if state.get("enabled") is False:
        do_not_click = True
    if area_ratio > 0.20 and not interactive:
        do_not_click = True

    if do_not_click:
        risk = "risky"
    elif interactive and clickability >= 0.66 and label and "very_small" not in reasons and "very_large" not in reasons and "container_role" not in reasons:
        risk = "safe"
    else:
        risk = "uncertain" if interactive or clickability >= 0.52 else "risky"

    return risk, do_not_click, reasons

def _box_to_strict_element(box: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    bounds = {
        "x": int(box["x1"]),
        "y": int(box["y1"]),
        "width": int(max(1, box["x2"] - box["x1"])),
        "height": int(max(1, box["y2"] - box["y1"])),
    }
    cx = int(box.get("click_x", box.get("cx", bounds["x"] + bounds["width"] // 2)))
    cy = int(box.get("click_y", box.get("cy", bounds["y"] + bounds["height"] // 2)))
    cx = int(max(bounds["x"], min(bounds["x"] + bounds["width"] - 1, cx)))
    cy = int(max(bounds["y"], min(bounds["y"] + bounds["height"] - 1, cy)))

    out = {
        "id": int(box.get("id", -1)),
        "stable_id": str(box.get("stable_id") or _stable_id_for_box(box, img_w, img_h)),
        "role": role,
        "label": _normalize_label(box.get("label", box.get("text", ""))),
        "label_confidence": float(_clamp01(box.get("label_confidence", 0.0))),
        "interactive": bool(box.get("interactive", False)),
        "confidence": float(_clamp01(box.get("confidence", box.get("score", 0.0)))),
        "clickability": float(_clamp01(box.get("clickability", box.get("score", 0.0)))),
        "state": _normalize_state_dict(box.get("state"), role),
        "bounds": bounds,
        "click_point": {"x": cx, "y": cy},
        "source": str(box.get("source", "unknown")),
        "source_votes": box.get("source_votes", [str(box.get("source", "unknown"))]),
        "keyboard_focus_hint": box.get("keyboard_focus_hint"),
        "risk_level": str(box.get("risk_level", "uncertain")),
        "do_not_click": bool(box.get("do_not_click", False)),
        "reason_codes": list(box.get("reason_codes", [])),
    }
    return out


def _box_to_agent_click_target(box: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    role = _normalize_role(box.get("role", box.get("element_type", box.get("type", "other"))))
    label = _normalize_label(box.get("label", box.get("text", "")))
    x1 = int(box.get("x1", 0))
    y1 = int(box.get("y1", 0))
    x2 = int(box.get("x2", x1 + 1))
    y2 = int(box.get("y2", y1 + 1))
    click_x = int(box.get("click_x", box.get("cx", x1)))
    click_y = int(box.get("click_y", box.get("cy", y1)))
    click_x = int(max(x1, min(max(x1, x2 - 1), click_x)))
    click_y = int(max(y1, min(max(y1, y2 - 1), click_y)))
    return {
        "id": int(box.get("id", -1)),
        "stable_id": str(box.get("stable_id") or _stable_id_for_box(box, img_w, img_h)),
        "role": role,
        "label": label,
        "label_confidence": float(_clamp01(box.get("label_confidence", 0.0))),
        "interactive": bool(box.get("interactive", False)),
        "is_actionable": bool(box.get("interactive", False) and role in CLICKABLE_ROLES),
        "click_point": {"x": click_x, "y": click_y},
        "risk_level": str(box.get("risk_level", "uncertain")),
        "do_not_click": bool(box.get("do_not_click", False)),
        "reason_codes": list(box.get("reason_codes", [])),
        "click_element_params": {
            "element_id": int(box.get("id", -1)),
            "click_type": "left",
        },
        "click_coords_params": {
            "x": click_x,
            "y": click_y,
            "click_type": "left",
        },
    }


def build_strict_ui_payload(
    image_path: str,
    img_bgr: np.ndarray,
    boxes: List[Dict[str, Any]],
    ocr_words: Optional[List[Dict[str, Any]]] = None,
    include_legacy: bool = True,
) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    retina_scale = _estimate_retina_scale(w, h)
    regular_boxes = [b for b in boxes if not _is_ocr_box(b)]
    ocr_boxes = [b for b in boxes if _is_ocr_box(b)]
    elements = [_box_to_strict_element(b, w, h) for b in regular_boxes]
    ocr_elements = [_box_to_strict_element(b, w, h) for b in ocr_boxes]
    payload: Dict[str, Any] = {
        "schema_version": "ui-elements-v2",
        "image": {
            "path": str(image_path),
            "width": int(w),
            "height": int(h),
            "retina_scale": float(retina_scale),
            "coord_space": "pixel_top_left_origin",
        },
        "elements": elements,
        "ocr_elements": ocr_elements,
        "metrics": {
            "num_elements": int(len(elements)),
            "actionable_elements": int(sum(1 for e in elements if e.get("interactive") and e.get("role") in CLICKABLE_ROLES)),
            "avg_clickability": float(sum(float(e.get("clickability", 0.0)) for e in elements) / max(1, len(elements))),
        },
        "ocr_metrics": {
            "num_elements": int(len(ocr_elements)),
            "avg_clickability": float(sum(float(e.get("clickability", 0.0)) for e in ocr_elements) / max(1, len(ocr_elements))),
        },
        "agent_contract": {
            "schema_version": "navai-agent-action-v1",
            "coordinate_space": "screenshot_pixels",
            "supported_actions": [
                "click_element",
                "click_coords",
                "type",
                "press_key",
                "hotkey",
                "scroll",
                "wait",
                "done",
            ],
            "click_targets": [_box_to_agent_click_target(b, w, h) for b in regular_boxes],
            "ocr_click_targets": [_box_to_agent_click_target(b, w, h) for b in ocr_boxes],
        },
    }
    if ocr_words is not None:
        payload["ocr_words"] = ocr_words
    if include_legacy:
        payload["num_boxes"] = len(boxes)
        payload["boxes"] = boxes
    return payload

def _pairwise_duplicate_rate(boxes: List[Dict[str, Any]]) -> float:
    if len(boxes) <= 1:
        return 0.0
    pairs = 0
    dups = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            pairs += 1
            a, b = boxes[i], boxes[j]
            role_a = _normalize_role(a.get("role", a.get("element_type", a.get("type", "other"))))
            role_b = _normalize_role(b.get("role", b.get("element_type", b.get("type", "other"))))
            if role_a != role_b:
                continue
            if iou(a, b) >= 0.72:
                dups += 1
    return float(dups / max(1, pairs))

def _clickpoint_inside_rate(boxes: List[Dict[str, Any]]) -> float:
    if not boxes:
        return 1.0
    ok = 0
    for b in boxes:
        x = int(b.get("click_x", b.get("cx", -1)))
        y = int(b.get("click_y", b.get("cy", -1)))
        if int(b["x1"]) <= x < int(b["x2"]) and int(b["y1"]) <= y < int(b["y2"]):
            ok += 1
    return float(ok / max(1, len(boxes)))

def _text_recall_proxy(words: List[Dict[str, Any]], boxes: List[Dict[str, Any]]) -> float:
    if not words:
        return 0.0
    covered = 0
    valid_words = [w for w in words if float(w.get("conf", -1)) >= 45]
    for w in valid_words:
        cx = float(w["x"]) + float(w["w"]) / 2.0
        cy = float(w["y"]) + float(w["h"]) / 2.0
        if any(b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"] for b in boxes):
            covered += 1
    return float(covered / max(1, len(valid_words)))

def evaluate_screenshot_folder(
    image_dir: str,
    out_json: str,
    backend: str = "auto",
    strict: bool = True,
    max_boxes: int = DEFAULT_MAX_BOXES,
    min_score: float = DEFAULT_MIN_SCORE,
    save_debug_overlays: bool = False,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    p = Path(image_dir)
    images = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])
    results: List[Dict[str, Any]] = []
    if save_debug_overlays and debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.time()
        boxes = parse_ui_everything(
            img,
            backend=backend,
            strict=strict,
            max_boxes=max_boxes,
            min_score=min_score,
        )
        dt_ms = (time.time() - t0) * 1000.0
        words = try_ocr_words(img)
        screen_area = float(max(1, img.shape[0] * img.shape[1]))
        avg_area_ratio = float(sum(float(b.get("area", 0.0)) for b in boxes) / max(1.0, len(boxes) * screen_area))
        small_rate = float(sum(1 for b in boxes if int(b.get("w", 0)) < 16 or int(b.get("h", 0)) < 16) / max(1, len(boxes)))
        actionable_ratio = float(sum(1 for b in boxes if b.get("interactive", False)) / max(1, len(boxes)))

        item = {
            "image": str(img_path),
            "num_elements": len(boxes),
            "duplicate_rate": _pairwise_duplicate_rate(boxes),
            "avg_box_area_ratio": avg_area_ratio,
            "small_box_rate": small_rate,
            "actionable_ratio": actionable_ratio,
            "clickpoint_inside_rate": _clickpoint_inside_rate(boxes),
            "text_recall_proxy": _text_recall_proxy(words, boxes),
            "latency_ms": float(dt_ms),
        }
        results.append(item)

        if save_debug_overlays and debug_dir:
            ann = draw(img, boxes, show_click_points=True, show_legend=True)
            cv2.imwrite(str(Path(debug_dir) / f"{img_path.stem}_debug.png"), ann)

    summary = {
        "schema_version": "vision-eval-v1",
        "image_dir": str(image_dir),
        "num_images": len(results),
        "metrics": {
            "avg_num_elements": float(sum(r["num_elements"] for r in results) / max(1, len(results))),
            "avg_duplicate_rate": float(sum(r["duplicate_rate"] for r in results) / max(1, len(results))),
            "avg_box_area_ratio": float(sum(r["avg_box_area_ratio"] for r in results) / max(1, len(results))),
            "avg_small_box_rate": float(sum(r["small_box_rate"] for r in results) / max(1, len(results))),
            "avg_actionable_ratio": float(sum(r["actionable_ratio"] for r in results) / max(1, len(results))),
            "avg_clickpoint_inside_rate": float(sum(r["clickpoint_inside_rate"] for r in results) / max(1, len(results))),
            "avg_text_recall_proxy": float(sum(r["text_recall_proxy"] for r in results) / max(1, len(results))),
            "avg_latency_ms": float(sum(r["latency_ms"] for r in results) / max(1, len(results))),
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in results], 95)) if results else 0.0,
        },
        "per_image": results,
    }
    Path(out_json).write_text(json.dumps(summary, indent=2))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to screenshot/image")
    ap.add_argument("--eval_dir", default=None, help="Folder of screenshots to evaluate in batch mode")
    ap.add_argument("--eval_out", default=None, help="Path for evaluation summary JSON")
    ap.add_argument("--out_dir", default="mac_ui_outputs", help="Output folder")
    ap.add_argument("--strict_json_only", action="store_true", default=(not DEFAULT_INCLUDE_LEGACY_JSON), help="Write strict schema JSON without legacy 'boxes'")
    ap.add_argument("--save_debug_overlays", action="store_true", default=False, help="Save debug overlays for --eval_dir runs")
    ap.add_argument("--debug_dir", default=None, help="Debug overlay output dir for --eval_dir")
    ap.add_argument("--save_crops", action="store_true", help="Save crops for each detected region")
    ap.add_argument("--ocr", action="store_true", help="Try OCR on text regions (requires pytesseract + tesseract)")
    ap.add_argument("--nms", type=float, default=0.35, help="NMS IoU threshold")
    ap.add_argument("--contain", type=float, default=0.95, help="Containment prune IoU threshold")
    ap.add_argument("--backend", choices=["auto", "cv"], default=_runtime_default_backend(), help="UI detection backend preference")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=_runtime_default_strict(), help="Prefer fewer, more-clickable elements")
    ap.add_argument("--max_boxes", type=int, default=_runtime_default_max_boxes(), help="Cap the number of output boxes")
    ap.add_argument("--min_score", type=float, default=_runtime_default_min_score(), help="Minimum score for CV-proposed boxes")
    ap.add_argument("--florence_model", default=None, help="Override Florence-2 model name")
    ap.add_argument("--no-florence", action="store_true", default=False, help="Disable Florence-2 in auto mode")
    args = ap.parse_args()
    if args.florence_model:
        os.environ["VISION_FLORENCE_MODEL"] = str(args.florence_model)
    if args.no_florence:
        os.environ["VISION_USE_FLORENCE"] = "0"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_dir:
        eval_out = args.eval_out or str(out_dir / "vision_eval_metrics.json")
        debug_dir = args.debug_dir or (str(out_dir / "vision_eval_debug") if args.save_debug_overlays else None)
        summary = evaluate_screenshot_folder(
            image_dir=args.eval_dir,
            out_json=eval_out,
            backend=args.backend,
            strict=bool(args.strict),
            max_boxes=int(args.max_boxes),
            min_score=float(args.min_score),
            save_debug_overlays=bool(args.save_debug_overlays),
            debug_dir=debug_dir,
        )
        print(f"Evaluated images: {summary.get('num_images', 0)}")
        print(f"Eval JSON: {eval_out}")
        if args.save_debug_overlays and debug_dir:
            print(f"Debug overlays: {debug_dir}")
        return

    if not args.image:
        raise ValueError("Provide --image for single-image mode, or --eval_dir for batch evaluation.")

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

            if args.ocr and ENABLE_OCR and b.get("type") == "text_region" and not b.get("text"):
                text = try_ocr_on_crop(c)
                if text:
                    b["text"] = text

    # Save strict JSON payload (+ optional legacy compatibility fields).
    json_path = out_dir / f"{img_path.stem}_ui_everything.json"
    payload = build_strict_ui_payload(
        image_path=str(img_path),
        img_bgr=img,
        boxes=boxes,
        ocr_words=ocr_words if (args.ocr and ENABLE_OCR) else None,
        include_legacy=(not args.strict_json_only),
    )
    if DEFAULT_PRETTY_JSON:
        json_path.write_text(json.dumps(payload, indent=2))
    else:
        json_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))

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
