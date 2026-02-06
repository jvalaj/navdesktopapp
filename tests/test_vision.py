import unittest

import cv2
import numpy as np

import vision


def _make_synth_ui(width: int = 900, height: int = 600) -> np.ndarray:
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # Panel
    cv2.rectangle(img, (40, 50), (420, 520), (235, 235, 235), -1)
    cv2.rectangle(img, (40, 50), (420, 520), (170, 170, 170), 2)

    # Button
    cv2.rectangle(img, (520, 120), (820, 180), (230, 230, 230), -1)
    cv2.rectangle(img, (520, 120), (820, 180), (120, 120, 120), 2)
    cv2.putText(img, "OK", (645, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 2, cv2.LINE_AA)

    # Text line
    cv2.putText(img, "Settings", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)

    return img


class VisionTests(unittest.TestCase):
    def test_extract_json_dict_from_text_handles_fence(self) -> None:
        text = "```json\n{\"elements\":[{\"x1\":0.1,\"y1\":0.1,\"x2\":0.2,\"y2\":0.2}]}\n```"
        payload = vision._extract_json_dict_from_text(text)
        self.assertIsInstance(payload, dict)
        self.assertIn("elements", payload)

    def test_parse_model_box_supports_normalized_and_percent(self) -> None:
        normalized = vision._parse_model_box(
            {"x1": 0.10, "y1": 0.20, "x2": 0.30, "y2": 0.40, "confidence": 0.9, "element_type": "button"},
            1000,
            500,
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["x1"], 100)
        self.assertEqual(normalized["y1"], 100)

        percent = vision._parse_model_box(
            {"x1": 10, "y1": 20, "x2": 30, "y2": 40, "confidence": 90, "element_type": "button"},
            1000,
            500,
        )
        self.assertIsNotNone(percent)
        self.assertEqual(percent["x1"], 100)
        self.assertEqual(percent["y1"], 100)

    def test_validate_vlm_boxes_filters_blank_regions(self) -> None:
        img = np.full((500, 900, 3), 245, dtype=np.uint8)
        candidate = {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": "button",
            "interactive": True,
            "x1": 100,
            "y1": 100,
            "x2": 180,
            "y2": 140,
            "score": 0.9,
        }
        vision.clamp_box(candidate, img.shape[1], img.shape[0])
        kept = vision._validate_vlm_boxes(img, [candidate], strict=True, cv_hints=[])
        self.assertEqual(len(kept), 0)

    def test_prune_vlm_container_candidates_drops_large_parent(self) -> None:
        outer = {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": "panel",
            "is_container": True,
            "interactive": False,
            "x1": 40,
            "y1": 40,
            "x2": 860,
            "y2": 460,
            "score": 0.9,
        }
        child1 = {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": "button",
            "is_container": False,
            "interactive": True,
            "x1": 100,
            "y1": 100,
            "x2": 220,
            "y2": 150,
            "score": 0.9,
        }
        child2 = {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": "button",
            "is_container": False,
            "interactive": True,
            "x1": 260,
            "y1": 100,
            "x2": 380,
            "y2": 150,
            "score": 0.9,
        }
        child3 = {
            "type": "vlm_element",
            "source": "vlm",
            "element_type": "textfield",
            "is_container": False,
            "interactive": True,
            "x1": 100,
            "y1": 180,
            "x2": 380,
            "y2": 230,
            "score": 0.9,
        }
        boxes = [outer, child1, child2, child3]
        for b in boxes:
            vision.clamp_box(b, 900, 500)

        kept = vision._prune_vlm_container_candidates(boxes, 900, 500, strict=True)
        self.assertEqual(len(kept), 3)
        self.assertTrue(all(b.get("element_type") != "panel" for b in kept))

    def test_detect_ui_rectangles_finds_synth_shapes(self) -> None:
        img = _make_synth_ui()
        boxes = vision.detect_ui_rectangles(img, scale=1.0, min_score=0.30)
        self.assertTrue(len(boxes) >= 1)

        # Expect a box overlapping the button region.
        target = {"x1": 520, "y1": 120, "x2": 820, "y2": 180}
        best = max((vision.iou(b, target) for b in boxes), default=0.0)
        self.assertGreater(best, 0.40)

    def test_parse_ui_everything_caps_and_sanitizes(self) -> None:
        img = _make_synth_ui()
        boxes = vision.parse_ui_everything(
            img,
            backend="cv",
            strict=True,
            max_boxes=5,
            min_score=0.20,
            refine_boxes=True,
        )
        self.assertLessEqual(len(boxes), 5)
        for idx, b in enumerate(boxes):
            self.assertEqual(b.get("id"), idx)
            for k in ("x1", "y1", "x2", "y2", "w", "h", "area", "cx", "cy", "click_x", "click_y"):
                self.assertIn(k, b)
            self.assertGreater(b["w"], 0)
            self.assertGreater(b["h"], 0)
            self.assertGreaterEqual(b["click_x"], 0)
            self.assertGreaterEqual(b["click_y"], 0)
            self.assertLess(b["click_x"], img.shape[1])
            self.assertLess(b["click_y"], img.shape[0])

if __name__ == "__main__":
    unittest.main()
