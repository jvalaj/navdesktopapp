import cv2
import numpy as np
import json
import argparse
from pathlib import Path

# OCR is disabled by default; set to True if you want OCR features.
ENABLE_OCR = False

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
def detect_ui_rectangles(img_bgr, scale=1.0):
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge emphasis helps with UI borders
    edges = cv2.Canny(gray, 60, 160)

    # Close gaps in edges to form rectangles
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1, iterations=2)

    # Fill rectangles a bit
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    filled = cv2.dilate(closed, k2, iterations=1)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(1, bh)
        a = bw * bh

        # Filters tuned for screenshots; you WILL tune these per your pipeline
        if a < 250:     # tiny noise
            continue
        if a > 0.92 * (w * h):  # entire screen
            continue
        if bw < 8 or bh < 8:
            continue
        if ar > 60 or ar < 1/60:
            continue

        # Normalize score: prefer "box-like" shapes
        # (you can improve this later with a classifier)
        score = 1.0
        if 0.2 < ar < 5.0:
            score += 0.25
        if a > 2000:
            score += 0.15

        # Map back to original coordinates if scaled
        if scale != 1.0:
            x1 = int(x / scale); y1 = int(y / scale)
            x2 = int((x + bw) / scale); y2 = int((y + bh) / scale)
        else:
            x1, y1, x2, y2 = x, y, x + bw, y + bh

        boxes.append({
            "type": "ui_rect",
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


    # Increase contrast a bit
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

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

    # Group characters into lines/blocks
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
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

        # Score: text blocks tend to be wider than tall
        score = 1.0
        if bw / max(1, bh) > 2.0:
            score += 0.3
        if a > 2000:
            score += 0.15

        boxes.append({
            "type": "text_region",
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
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
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
    box["x1"] = max(0, x1 + new_x1 - padding)
    box["y1"] = max(0, y1 + new_y1 - padding)
    box["x2"] = min(w, x1 + new_x2 + padding)
    box["y2"] = min(h, y1 + new_y2 + padding)

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


# -----------------------------
# Main pipeline
# -----------------------------
def parse_ui_everything(img_bgr,
                        rect_scales=(1.0, 0.75, 0.5),
                        nms_iou=0.35,
                        containment_iou=0.95,
                        refine_boxes=True):
    h, w = img_bgr.shape[:2]

    # 1) Rect proposals at multiple scales
    rect_boxes = []
    for s in rect_scales:
        rect_boxes.extend(detect_ui_rectangles(img_bgr, scale=s))

    # 2) Text proposals
    text_boxes = detect_text_regions(img_bgr)

    # Merge
    boxes = rect_boxes + text_boxes

    # Clamp + area, center
    for b in boxes:
        clamp_box(b, w, h)

    # Drop boxes with near-zero area (after clamp)
    boxes = [b for b in boxes if b["area"] > 0]

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
            # Only refine boxes that are reasonably sized (not too big, not too small)
            if 100 < b["area"] < 500000:  # Skip full-screen and tiny boxes
                refine_box_to_content(img_bgr, b, padding=2)
                # Recalculate after refinement
                clamp_box(b, w, h)

    # 9) Compute safe click points
    add_click_points(boxes, w, h)

    # Sort in reading order (good for agent)
    boxes.sort(key=lambda b: (b["y1"], b["x1"]))

    # Assign IDs
    for i, b in enumerate(boxes):
        b["id"] = i

    return boxes

def draw(img_bgr, boxes, show_click_points=True):
    out = img_bgr.copy()
    outline_color = (0, 255, 0)  # green for all elements (BGR)
    text_color = (0, 0, 0)       # black for ID labels
    click_color = (0, 0, 255)    # red for click points
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), outline_color, 2)

        # label text
        label = f'{b["id"]}'
        # optional: add a bit of OCR text to help visual matching
        if b.get("text"):
            t = b["text"][:18]
            label = f'{b["id"]}: {t}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        # filled background (green to match outline)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), outline_color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Optional: draw click point (no coordinates)
        if show_click_points and "click_x" in b and "click_y" in b:
            cx, cy = int(b["click_x"]), int(b["click_y"])
            cv2.circle(out, (cx, cy), 3, click_color, -1)

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
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    boxes = parse_ui_everything(
        img,
        rect_scales=(1.0, 0.75, 0.5),
        nms_iou=args.nms,
        containment_iou=args.contain
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
