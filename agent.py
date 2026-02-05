"""
Simple Computer Use Agent with Zai GLM-4.6v Vision Model.

Workflow:
1. SEE - Take regular screenshot, LLM looks at it
2. CLICK - When LLM needs to click:
   - Run vision.py to get annotated screenshot + JSON
   - LLM sees element IDs in annotated image
   - LLM specifies element ID to click
   - Agent gets coordinates from JSON and clicks
3. KEYBOARD actions (type, hotkey, press) work directly
"""

import os
import re
import json
import time
import cv2
import pyautogui
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic
from Quartz import CoreGraphics as CG

# Import our modules
import vision
import typeandclick

# Load environment variables
load_dotenv()

# Suppress dock icon on macOS (prevents bouncing/badging)
# This must be done before any GUI operations
try:
    from AppKit import NSApplication, NSActivationPolicyAccessory
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSActivationPolicyAccessory)  # No dock icon, no menu bar
except Exception:
    pass  # AppKit not available or failed - not critical

# Screen capture settings
def _env_path(key: str) -> Optional[str]:
    val = os.environ.get(key)
    if val is None:
        return None
    val = str(val).strip()
    return val or None


SCREENSHOT_DIR = Path(_env_path("NAVAI_SCREENSHOTS_DIR") or "screenshots")
MEMORY_DIR = Path(_env_path("NAVAI_CONVERSATIONS_DIR") or "conversations")

# Create directories
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class ConversationMemory:
    """Manages conversation memory as a text file."""

    def __init__(self, conversation_id: Optional[str] = None):
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_id = conversation_id
        self.memory_file = MEMORY_DIR / f"{conversation_id}.txt"
        self.history: List[Dict[str, str]] = []
        self._load_existing()
        # Create the log file immediately when a conversation starts,
        # even before the first message is written.
        try:
            if not self.memory_file.exists():
                self.memory_file.write_text("")
        except Exception:
            # Don't fail agent startup if the log can't be created.
            pass

    def _load_existing(self):
        """Load existing conversation if file exists."""
        if self.memory_file.exists():
            content = self.memory_file.read_text()
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        self.history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        self._save()

    def _save(self):
        """Save history to file."""
        with open(self.memory_file, 'w') as f:
            for entry in self.history:
                f.write(json.dumps(entry) + '\n')

    def get_context_summary(self, max_entries: int = 50) -> str:
        """Get a text summary of recent conversation."""
        recent = self.history[-max_entries:]
        summary = []
        for entry in recent:
            role = entry["role"].upper()
            content = entry["content"]
            metadata = entry.get("metadata", {})
            meta_str = ""
            if metadata:
                parts = [f"{k}={v}" for k, v in metadata.items()]
                meta_str = f" [{', '.join(parts)}]"
            summary.append(f"{role}{meta_str}: {content}")
        return "\n".join(summary)


class Agent:
    """Computer Use Agent that sees UI and acts on it."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        conversation_id: Optional[str] = None,
        thinking_enabled: bool = True
    ):
        if api_key is None:
            api_key = os.environ.get("claudekey")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.memory = ConversationMemory(conversation_id)
        self.thinking_enabled = thinking_enabled
        self.step_count = 0
        self.last_vision_data = None  # Stores last vision analysis results
        self.last_raw_screenshot: Optional[str] = None
        self.last_visual_delta: Optional[float] = None
        self.last_action_name: Optional[str] = None
        self.last_action_success: Optional[bool] = None
        self.progress_score: float = 0.0
        self.last_progress_note: str = "unknown"
        # State management for smarter decisions
        self.clicked_elements: List[int] = []  # Track clicked element IDs
        self.action_history: List[Dict[str, Any]] = []  # Track recent actions
        self.last_screenshot_hash: Optional[str] = None  # Detect UI changes

    def _compute_screenshot_hash(self, screenshot_path: str) -> str:
        """Compute a simple hash of screenshot to detect UI changes."""
        import hashlib
        img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
        # Resize to small size for fast hashing
        small = cv2.resize(img, (100, 100))
        return hashlib.md5(small.tobytes()).hexdigest()

    def _compute_visual_delta(self, prev_path: str, curr_path: str) -> Optional[float]:
        """
        Compute a coarse visual change percentage between two screenshots.
        Returns a percent in [0, 100], or None if comparison fails.
        """
        try:
            img1 = cv2.imread(prev_path)
            img2 = cv2.imread(curr_path)
            if img1 is None or img2 is None:
                return None
            # Downscale both to a common size to stabilize diff cost.
            size = (128, 128)
            img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g2)
            mean_diff = float(diff.mean()) / 255.0
            return max(0.0, min(100.0, mean_diff * 100.0))
        except Exception:
            return None

    def _update_progress(self, visual_delta: Optional[float]) -> None:
        """
        Update a coarse progress estimate based on visual change and action success.
        This is a heuristic signal, not a task-specific completion guarantee.
        """
        if visual_delta is None:
            self.last_progress_note = "unknown"
            return

        change = 0.0
        if visual_delta >= 10.0:
            change += 8.0
            note = "major UI change"
        elif visual_delta >= 5.0:
            change += 5.0
            note = "moderate UI change"
        elif visual_delta >= 2.0:
            change += 2.0
            note = "minor UI change"
        elif visual_delta < 0.5:
            change -= 3.0
            note = "no visible change"
        else:
            change -= 1.0
            note = "very small change"

        if self.last_action_success is False:
            change -= 4.0
            note = f"{note}; last action failed"

        # Wait/see/detect are informational; avoid inflating progress.
        if self.last_action_name in ("wait", "see", "detect_elements"):
            change = min(change, 1.0)

        self.progress_score = max(0.0, min(100.0, self.progress_score + change))
        self.last_progress_note = note

    def _has_ui_changed(self, new_screenshot_path: str) -> bool:
        """Check if UI has changed since last screenshot."""
        new_hash = self._compute_screenshot_hash(new_screenshot_path)
        if self.last_screenshot_hash is None:
            self.last_screenshot_hash = new_hash
            return True
        changed = new_hash != self.last_screenshot_hash
        self.last_screenshot_hash = new_hash
        return changed

    def take_screenshot(self) -> str:
        """Take a regular screenshot and return the path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        screenshot = pyautogui.screenshot()
        screenshot_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
        screenshot.save(str(screenshot_path))
        return str(screenshot_path)

    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen dimensions for coordinate mapping."""
        main_display = CG.CGMainDisplayID()
        bounds = CG.CGDisplayBounds(main_display)
        physical_w = CG.CGDisplayPixelsWide(main_display)
        physical_h = CG.CGDisplayPixelsHigh(main_display)
        # Prefer display mode pixel size when available (more reliable on Retina)
        try:
            mode = CG.CGDisplayCopyDisplayMode(main_display)
            if mode:
                mw = CG.CGDisplayModeGetPixelWidth(mode)
                mh = CG.CGDisplayModeGetPixelHeight(mode)
                if mw and mh:
                    physical_w = int(mw)
                    physical_h = int(mh)
        except Exception:
            pass
        pyauto_w, pyauto_h = pyautogui.size()
        logical_w = int(bounds.size.width)
        logical_h = int(bounds.size.height)

        return {
            "logical_width": logical_w,
            "logical_height": logical_h,
            "physical_width": physical_w,
            "physical_height": physical_h,
            "pyauto_width": pyauto_w,
            "pyauto_height": pyauto_h,
            "scale_factor": physical_w / logical_w if logical_w > 0 else 1.0,
        }

    def _ensure_vision(self) -> Dict[str, Any]:
        """Ensure we have fresh vision data."""
        if self.last_vision_data is None:
            return self.run_vision_analysis()
        return self.last_vision_data

    def _infer_scale(self, shot_width: int, shot_height: int, screen_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer per-axis scale factors from screenshot-space to PyAutoGUI-space.
        Prefer direct size ratios to avoid Quartz-space ambiguity.
        """
        shot = (int(shot_width), int(shot_height))
        pyauto = (int(screen_info["pyauto_width"]), int(screen_info["pyauto_height"]))
        logical = (int(screen_info["logical_width"]), int(screen_info["logical_height"]))
        physical = (int(screen_info["physical_width"]), int(screen_info["physical_height"]))

        sx = pyauto[0] / shot[0] if shot[0] else 1.0
        sy = pyauto[1] / shot[1] if shot[1] else 1.0

        return {
            "scale_x": float(sx),
            "scale_y": float(sy),
            "reason": "ratio_pyauto_to_shot",
            "shot": shot,
            "pyauto": pyauto,
            "logical": logical,
            "physical": physical,
        }

    def _box_contains_point(self, box: Dict[str, Any], x: int, y: int) -> bool:
        return box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]

    def _child_count(self, parent: Dict[str, Any], boxes: List[Dict[str, Any]]) -> int:
        count = 0
        for b in boxes:
            if b is parent:
                continue
            if b["x1"] >= parent["x1"] and b["y1"] >= parent["y1"] and b["x2"] <= parent["x2"] and b["y2"] <= parent["y2"]:
                count += 1
        return count

    def _select_best_box_for_point(self, x: int, y: int, boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Pick the most specific box around a point.
        Prefer containing boxes; score by distance + size + container penalty.
        """
        if not boxes:
            return None
        containing = [b for b in boxes if self._box_contains_point(b, x, y)]
        candidates = containing if containing else boxes

        best = None
        best_score = None
        for b in candidates:
            bx = int(b.get("click_x", b["cx"]))
            by = int(b.get("click_y", b["cy"]))
            dist = ((bx - x) ** 2 + (by - y) ** 2) ** 0.5
            area_penalty = b["area"] * 0.0005
            child_penalty = self._child_count(b, boxes) * 30.0
            score = dist + area_penalty + child_penalty
            if best is None or score < best_score:
                best = b
                best_score = score
        return best

    def _refine_container_target(self, box: Dict[str, Any], boxes: List[Dict[str, Any]], screen_area: int) -> Dict[str, Any]:
        """
        If the target is a large container with many children, pick a more specific child.
        """
        if not box or not boxes:
            return box
        if box.get("type") == "text_region":
            return box
        if screen_area <= 0:
            return box
        if box["area"] < screen_area * 0.06:
            return box
        if self._child_count(box, boxes) < 3:
            return box

        # Consider children inside the box, smaller than parent
        children = [b for b in boxes if b is not box and self._box_contains_point(box, b["cx"], b["cy"]) and b["area"] < box["area"] * 0.6]
        if not children:
            return box
        target_x = int(box.get("click_x", box["cx"]))
        target_y = int(box.get("click_y", box["cy"]))
        refined = self._select_best_box_for_point(target_x, target_y, children)
        return refined or box

    def run_vision_analysis(self) -> Dict[str, Any]:
        """
        Run vision.py analysis on current screen.

        Returns:
            Dict with screenshot, annotated, boxes, json_path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot_path = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
        screenshot.save(str(screenshot_path))

        # Compute visual delta vs previous screenshot (if any)
        visual_delta = None
        if self.last_raw_screenshot:
            visual_delta = self._compute_visual_delta(self.last_raw_screenshot, str(screenshot_path))
        self.last_raw_screenshot = str(screenshot_path)
        self.last_visual_delta = visual_delta
        self._update_progress(visual_delta)

        # Get screenshot dimensions
        shot_width, shot_height = screenshot.size

        # Convert to OpenCV format
        img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Log and store screen info for calibration
        screen_info = self.get_screen_info()
        print(
            "Screen info: "
            f"shot={shot_width}x{shot_height}, "
            f"pyauto={screen_info['pyauto_width']}x{screen_info['pyauto_height']}, "
            f"logical={screen_info['logical_width']}x{screen_info['logical_height']}, "
            f"physical={screen_info['physical_width']}x{screen_info['physical_height']}, "
            f"scale_factor={screen_info['scale_factor']:.3f}"
        )

        # Run vision analysis
        boxes = vision.parse_ui_everything(img_bgr)

        # Try OCR if enabled to get text for each box
        if vision.ENABLE_OCR:
            ocr_words = vision.try_ocr_words(img_bgr)
            if ocr_words:
                vision.assign_words_to_boxes(boxes, ocr_words)
                # Recompute click points using OCR text centroids
                vision.add_click_points(boxes, shot_width, shot_height)

        # Create annotated image
        annotated_img = vision.draw(img_bgr, boxes)
        annotated_path = SCREENSHOT_DIR / f"annotated_{timestamp}.png"
        cv2.imwrite(str(annotated_path), annotated_img)

        # Save JSON
        json_path = SCREENSHOT_DIR / f"boxes_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "screenshot": str(screenshot_path),
            "num_boxes": len(boxes),
            "boxes": boxes
        }
        json_path.write_text(json.dumps(payload, indent=2))

        # Store for reference
        self.last_vision_data = {
            "screenshot": str(screenshot_path),
            "annotated": str(annotated_path),
            "boxes": boxes,
            "json_path": str(json_path),
            "shot_width": shot_width,
            "shot_height": shot_height,
            "screen_info": screen_info,
            "visual_delta": visual_delta
        }

        return self.last_vision_data

    def get_box_by_id(self, box_id: int) -> Optional[Dict]:
        """Get a box by its ID from the last vision analysis."""
        if self.last_vision_data is None:
            return None
        for box in self.last_vision_data["boxes"]:
            if box.get("id") == box_id:
                return box
        return None

    def get_ui_summary(self, boxes: List[Dict]) -> str:
        """Get a text summary of detected UI elements."""
        if not boxes:
            return "No UI elements detected."

        summary = []
        summary.append(f"Detected {len(boxes)} UI elements:")
        if self.last_visual_delta is not None:
            summary.append(f"Visual change since last step: ~{self.last_visual_delta:.1f}%")
            summary.append(f"Progress estimate: ~{self.progress_score:.0f}/100 ({self.last_progress_note})\n")
        else:
            summary.append("")

        for box in boxes:
            box_type = box.get("type", "unknown")
            box_id = box.get("id", "?")
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cx, cy = box["cx"], box["cy"]
            w, h = box["w"], box["h"]
            click_x = box.get("click_x", cx)
            click_y = box.get("click_y", cy)

            desc = f"  [{box_id}] {box_type} center ({cx}, {cy}), click ({click_x}, {click_y}), size {w}x{h}"

            if box.get("text"):
                desc += f" - text: '{box['text']}'"

            summary.append(desc)

        return "\n".join(summary)

    # ========================================================================
    # ACTIONS THAT LLM CAN CALL
    # ========================================================================

    def action_see(self) -> Dict[str, Any]:
        """
        Take a screenshot and show it to the LLM.
        Use this when you just need to see what's on screen.
        """
        screenshot_path = self.take_screenshot()
        return {
            "success": True,
            "action": "see",
            "screenshot": screenshot_path,
            "message": f"Took screenshot: {screenshot_path}"
        }

    def action_click_element(self, element_id: int = None, click_type: str = "left", index: int = None, **kwargs) -> Dict[str, Any]:
        """
        Click on a UI element by its ID from the vision analysis.

        Args:
            element_id: The ID number of the element to click (shown in annotated image)
            click_type: Type of click - 'left', 'right', or 'double'
            index: Alias for element_id (some LLMs use this name)
            **kwargs: Additional parameters (ignored, e.g., 'position' from LLM)
        """
        # Handle both element_id and index parameter names
        if element_id is None and index is not None:
            element_id = index
        elif element_id is None and index is None:
            return {
                "success": False,
                "error": "Missing element_id or index parameter"
            }

        # Check if we've already clicked this element recently (avoid loops)
        if element_id in self.clicked_elements[-5:]:  # Check last 5 clicks
            return {
                "success": False,
                "error": f"Element ID {element_id} was already clicked recently. Try a different element or action."
            }

        # Validate element_id - accept numbers or extract number from strings
        if isinstance(element_id, str):
            # Try to extract a number from various formats: "[5]", "element 5", "id:5", etc.
            match = re.search(r'(\d+)', element_id)
            if match:
                element_id = int(match.group(1))
            else:
                return {
                    "success": False,
                    "error": f"Could not extract element ID from '{element_id}'. Use just the number like {{'element_id': 5}}"
                }
        elif not isinstance(element_id, (int, float)) or isinstance(element_id, bool):
            return {
                "success": False,
                "error": f"element_id must be a number, not '{element_id}'. Use the ID from the annotated image."
            }
        else:
            element_id = int(element_id)

        # Run vision if not done yet
        vision_data = self._ensure_vision()

        # Find the box
        box = self.get_box_by_id(element_id)
        if box is None:
            return {
                "success": False,
                "error": f"Element ID {element_id} not found. Available IDs: {[b.get('id') for b in vision_data['boxes']]}"
            }

        # Refine large container targets to a more specific child
        screen_area = vision_data.get("shot_width", 0) * vision_data.get("shot_height", 0)
        refined_box = self._refine_container_target(box, vision_data["boxes"], screen_area)
        if refined_box is not None and refined_box.get("id") != box.get("id"):
            box = refined_box
            element_id = box.get("id", element_id)

        # Get screen info and calculate scale factors
        screen_info = self.get_screen_info()
        shot_width = vision_data.get("shot_width", screen_info["pyauto_width"])
        shot_height = vision_data.get("shot_height", screen_info["pyauto_height"])

        scale_info = self._infer_scale(shot_width, shot_height, screen_info)
        scale_x = scale_info["scale_x"]
        scale_y = scale_info["scale_y"]

        # Get click coordinates from vision data and scale to screen coordinates
        cx = box.get("click_x", box["cx"])
        cy = box.get("click_y", box["cy"])
        scaled_x = round(cx * scale_x)
        scaled_y = round(cy * scale_y)
        box_type = box.get("type", "unknown")
        text = box.get("text", "")

        print(
            "Scale mapping: "
            f"scale_x={scale_x:.3f} "
            f"scale_y={scale_y:.3f} "
            f"reason={scale_info['reason']} "
            f"shot={scale_info['shot']} "
            f"pyauto={scale_info['pyauto']} "
            f"logical={scale_info['logical']} "
            f"physical={scale_info['physical']}"
        )

        # Execute the click with scaled coordinates
        if click_type == "left":
            result = typeandclick.left_click(scaled_x, scaled_y)
        elif click_type == "right":
            result = typeandclick.right_click(scaled_x, scaled_y)
        elif click_type == "double":
            result = typeandclick.double_click(scaled_x, scaled_y)
        else:
            return {
                "success": False,
                "error": f"Invalid click_type: {click_type}. Use 'left', 'right', or 'double'"
            }

        if result.get("success"):
            # Track clicked element to avoid repetition
            self.clicked_elements.append(element_id)
            # Keep only last 20 clicks
            if len(self.clicked_elements) > 20:
                self.clicked_elements = self.clicked_elements[-20:]

            result["message"] = f"Clicked element [{element_id}] ({box_type}) - original coords: ({cx}, {cy}) -> clicked at: ({scaled_x}, {scaled_y})"
            if text:
                result["message"] += f" '{text}'"
        else:
            # Error recovery: provide fallback hint
            result["recovery_hint"] = f"If element {element_id} failed, try clicking a nearby element or using coordinates."

        return result

    def action_click_coords(self, x: int, y: int, click_type: str = "left") -> Dict[str, Any]:
        """
        Click at exact screen coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            click_type: Type of click - 'left', 'right', or 'double'
        """
        # Snap coordinate clicks to nearest detected element for precision
        vision_data = self._ensure_vision()
        screen_info = self.get_screen_info()
        shot_width = vision_data.get("shot_width", screen_info["pyauto_width"])
        shot_height = vision_data.get("shot_height", screen_info["pyauto_height"])
        scale_info = self._infer_scale(shot_width, shot_height, screen_info)
        scale_x = scale_info["scale_x"] if scale_info["scale_x"] != 0 else 1.0
        scale_y = scale_info["scale_y"] if scale_info["scale_y"] != 0 else 1.0

        # Map screen coords -> screenshot coords
        shot_x = int(round(x / scale_x))
        shot_y = int(round(y / scale_y))
        target_box = self._select_best_box_for_point(shot_x, shot_y, vision_data["boxes"])
        if target_box:
            return self.action_click_element(element_id=target_box.get("id"), click_type=click_type)

        # Fallback: direct coordinate click
        if click_type == "left":
            return typeandclick.left_click(x, y)
        elif click_type == "right":
            return typeandclick.right_click(x, y)
        elif click_type == "double":
            return typeandclick.double_click(x, y)
        else:
            return {"success": False, "error": f"Invalid click_type: {click_type}"}

    def action_type(self, text: str) -> Dict[str, Any]:
        """Type text at the current cursor position."""
        return typeandclick.type_text(text)

    def action_press_key(self, key: str, presses: int = 1) -> Dict[str, Any]:
        """Press a keyboard key (e.g., 'enter', 'tab', 'escape', 'cmd', etc.)."""
        return typeandclick.press_key(key, presses=presses)

    def action_hotkey(self, *keys: str) -> Dict[str, Any]:
        """Press key combination (e.g., ['cmd', 'c'] for copy)."""
        return typeandclick.hotkey(*keys)

    def action_scroll(self, clicks: int) -> Dict[str, Any]:
        """Scroll the mouse wheel (positive=up, negative=down)."""
        return typeandclick.scroll(clicks)

    def action_wait(self, seconds: float = 1) -> Dict[str, Any]:
        """Pause for the given number of seconds."""
        time.sleep(max(0, min(seconds, 10)))  # Cap at 10s
        return {"success": True, "action": "wait", "message": f"Waited {seconds}s"}

    def action_detect_elements(self) -> Dict[str, Any]:
        """
        Run vision analysis and show the annotated screenshot with element IDs.
        Use this when you need to see what elements are available to click.
        Reuses last_vision_data when already available to avoid duplicate screenshots.
        """
        if self.last_vision_data:
            vision_data = self.last_vision_data
        else:
            vision_data = self.run_vision_analysis()
        summary = self.get_ui_summary(vision_data["boxes"])

        return {
            "success": True,
            "action": "detect_elements",
            "annotated_screenshot": vision_data["annotated"],
            "screenshot": vision_data["screenshot"],
            "json_path": vision_data["json_path"],
            "num_elements": len(vision_data["boxes"]),
            "summary": summary,
            "message": f"Analyzed screen, found {len(vision_data['boxes'])} elements. See annotated image for IDs."
        }

    def _record_action(self, action: str, params: Dict, result: Dict):
        """Record an action in history for stuck detection."""
        self.action_history.append({
            "step": self.step_count,
            "action": action,
            "params": params,
            "success": result.get("success", False)
        })
        # Keep only last 10 actions
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]

    def _check_task_completion(self, user_goal: str, current_screenshot: str) -> tuple[bool, str]:
        """
        Check if the task appears to be completed by analyzing the goal and screenshot.
        Returns (is_complete, reason)
        """
        import re
        from PIL import Image

        goal_lower = user_goal.lower()

        # For "open" goals, we can't easily verify without more context
        if "open" in goal_lower and len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if last_action["success"] and last_action["action"] in ("click_element", "click_coords"):
                return True, "Target appears to be opened"

        # Check for error messages on screen that might indicate completion
        img = Image.open(current_screenshot)
        # This is a simple heuristic - could be enhanced with actual OCR

        return False, ""

    def _is_stuck(self) -> tuple[bool, str]:
        """Detect if agent is stuck in a loop."""
        if len(self.action_history) < 3:
            return False, ""

        # Check for repeated same action on same element
        last_5 = self.action_history[-5:]
        failed_clicks = [a for a in last_5 if a["action"] == "click_element" and not a["success"]]
        if len(failed_clicks) >= 2:
            return True, f"Repeated failed clicks on element"

        # Check if clicking same element multiple times
        clicked_ids = [a["params"].get("element_id") for a in last_5 if a["action"] == "click_element" and a["success"]]
        if len(clicked_ids) >= 3 and len(set(clicked_ids)) == 1:
            return True, f"Clicked same element {clicked_ids[0]} three times without progress"

        # Check for repeated failures
        recent_failures = [a for a in last_5 if not a["success"]]
        if len(recent_failures) >= 3:
            return True, f"Multiple recent failures ({len(recent_failures)})"

        return False, ""

    def _get_action_memory_summary(self) -> str:
        """Get a summary of recent actions for the LLM context."""
        if not self.action_history:
            return "No previous actions yet."

        recent = self.action_history[-5:]  # Last 5 actions
        summary = []
        for a in recent:
            status = "✓" if a["success"] else "✗"
            action_str = a["action"]
            if a["action"] == "click_element":
                elem_id = a["params"].get("element_id", "?")
                action_str = f"click element [{elem_id}]"
            elif a["action"] == "type":
                text_preview = a["params"].get("text", "")[:30]
                action_str = f"type '{text_preview}'"
            elif a["action"] == "press_key":
                action_str = f"press {a['params'].get('key', '')}"
            elif a["action"] == "scroll":
                action_str = f"scroll {a['params'].get('clicks', '')}"
            summary.append(f"{status} {action_str}")
        return " | ".join(summary)

    def build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        action_memory = self._get_action_memory_summary()
        clicked_hint = ""
        if self.clicked_elements:
            clicked_hint = f"\n\nNOTE: You have already clicked elements: {list(set(self.clicked_elements[-10:]))}. Avoid clicking the same element repeatedly - try something different."

        return f"""You are a reliable computer-use agent controlling a macOS desktop.

You will be given, every step:
1) USER_GOAL (what to accomplish)
2) VISION_SUMMARY (list of detected UI elements with integer IDs and optional text)
3) ANNOTATED_SCREENSHOT (same screen image with element IDs drawn)

OPTIONALLY, you may also receive:
4) CONVERSATION_HISTORY (previous messages and actions in this session)

CRITICAL: You automatically receive fresh VISION_SUMMARY + ANNOTATED_SCREENSHOT at the start of EACH step. Do NOT request screenshots or vision analysis - you already have them.

Your job:
- Choose exactly ONE action per step that moves toward the goal.
- After you act, the system will provide a fresh VISION_SUMMARY + ANNOTATED_SCREENSHOT on the next step.
- Keep going until the goal is complete, then return action "done".

IMPORTANT RULES
- Output MUST be valid JSON only. No markdown, no extra commentary.
- Use ONLY the actions listed below. Do not invent tools.
- Never output multiple actions in one response.
- Never guess element_id. Only use IDs that exist in the provided VISION_SUMMARY / annotated screenshot.
- If you need to type into a field, you usually must click it first, then type, then press_key "enter" (across multiple steps).
- You do NOT need to take screenshots or run vision analysis - you receive fresh data automatically every step.

AVAILABLE ACTIONS (the only tool API you can call)
1) click_element
   params:
     - element_id: integer (required)
     - click_type: "left" | "right" | "double" (optional, default "left")
   Notes:
     - ALWAYS set click_type explicitly when it matters.
     - element_id must be a plain integer (examples: 0, 1, 2, 17). Do not include brackets or descriptions.

2) type
   params:
     - text: string (required)

3) press_key
   params:
     - key: string (required)  examples: "enter", "tab", "escape", "space", "backspace"
     - presses: integer (optional, default 1)

4) hotkey
   params:
     - keys: array of strings (required) examples: ["cmd","c"], ["cmd","v"], ["cmd","l"], ["cmd","tab"]

5) scroll
   params:
     - clicks: integer (required)  negative = down, positive = up

6) wait
   params:
     - seconds: number (optional, default 0.5). Use 0.5–2.0 for UI to settle.

7) done
   params:
     - summary: string (required) brief description of what was accomplished or why you are stopping.

HOW TO DECIDE WHAT TO DO (simple policy)
- Look at the VISION_SUMMARY to find element IDs - they are provided every step automatically.
- Prefer clicking a specific, small, relevant element over large containers.
- If there are multiple similar targets, pick the one whose nearby text best matches the goal.
- If the screen likely needs time to update (page load, modal opening), use wait(1.0) next step.
- AVOID CLICKING THE SAME ELEMENT MULTIPLE TIMES - if a click doesn't work, try a different approach.
- If you cannot find any relevant element ID in the VISION_SUMMARY for the next move:
  - Try scrolling (clicks = -8) to reveal more options.
  - If still stuck after 2 scroll attempts, finish with done and explain what is missing.

STOP CONDITION
- Use action "done" when:
  - the goal is clearly complete, OR
  - you are blocked because the needed UI element is not present / not visible / ambiguous.

RESPONSE JSON SCHEMA (ALWAYS THIS SHAPE)
{{
  "thought": "One short sentence describing what you will do next.",
  "action": "click_element | type | press_key | hotkey | scroll | wait | done",
  "params": {{ ... }}
}}

EXAMPLES

Example 1: Click a button using element ID from vision summary
{{
  "thought": "Click the Login button (element 12).",
  "action": "click_element",
  "params": {{ "element_id": 12, "click_type": "left" }}
}}

Example 2: Type text
{{
  "thought": "Type the search query.",
  "action": "type",
  "params": {{ "text": "software engineer role" }}
}}

Example 3: Finish
{{
  "thought": "The requested page is open and the task is complete.",
  "action": "done",
  "params": {{ "summary": "Opened the target page and completed the requested steps." }}
}}

The user's goal will be provided in the user message."""

    def _normalize_params(self, action: str, params: Dict) -> Dict:
        """Normalize params to handle common LLM mistakes (e.g. element/element_info -> element_id)."""
        params = dict(params)
        original_action = action  # Store original for click type detection

        if action == "click_element":
            # Auto-detect click_type from action name
            if "click_type" not in params:
                if "right" in original_action:
                    params["click_type"] = "right"
                elif "double" in original_action:
                    params["click_type"] = "double"
                else:
                    params["click_type"] = "left"  # default

            # Prioritize element_id first, then check other aliases
            # IMPORTANT: element_id must come FIRST so we prefer it over descriptions
            aliases = ("element_id", "index", "element", "element_info", "target_element", "target", "box", "bbox", "coordinates")
            first_value = None

            # Find the first value (element_id is checked first now!)
            # NOTE: Check if key exists, don't check truthiness (0 is valid!)
            for alias in aliases:
                if alias in params and params[alias] is not None:
                    first_value = params[alias]
                    break

            # Clear all aliases
            for alias in aliases:
                params.pop(alias, None)

            # Set element_id to the first value found
            if first_value is not None:
                params["element_id"] = first_value

        elif action == "hotkey":
            # Handle various key formats: "keys": ["cmd", "c"] or "key": "cmd+c" or as individual params
            if "keys" not in params:
                # Try to find keys in other formats
                possible_keys = []
                if "key" in params:
                    key_val = params["key"]
                    if isinstance(key_val, str):
                        # Handle "cmd+c" format
                        if "+" in key_val:
                            possible_keys = [k.strip() for k in key_val.split("+")]
                        else:
                            possible_keys = [key_val]
                    elif isinstance(key_val, list):
                        possible_keys = key_val
                    params.pop("key", None)

                # Also check for individual key1, key2, etc.
                i = 1
                while f"key{i}" in params:
                    possible_keys.append(params[f"key{i}"])
                    params.pop(f"key{i}", None)
                    i += 1

                # Also check if any values look like keys (short strings like 'cmd', 'c', 'ctrl')
                for k, v in list(params.items()):
                    if k not in ("keys", "key") and isinstance(v, str) and len(v) <= 5 and v.islower():
                        possible_keys.append(v)
                        params.pop(k, None)

                if possible_keys:
                    params["keys"] = possible_keys

        return params

    def execute_action(self, action: str, params: Dict = None) -> Dict[str, Any]:
        """Execute an action by name."""
        params = params or {}

        # Normalize action name FIRST (before param normalization, which uses action type)
        action = str(action).lower().strip()
        # If click with x,y coords (no element_id), use click_coords
        if action in ("left_click", "right_click", "double_click", "click") and isinstance(params, dict):
            if "x" in params and "y" in params and "element_id" not in params:
                orig = action
                action = "click_coords"
                params = dict(params)
                if "click_type" not in params:
                    params["click_type"] = "right" if "right" in orig else "double" if "double" in orig else "left"
        action_aliases = {
            "click": "click_element",
            "click_element": "click_element",
            "left_click": "click_element",
            "right_click": "click_element",
            "double_click": "click_element",
            "key": "press_key",
            "press": "press_key",
            "detect_elements": "detect_elements",
            "detect": "detect_elements",
            "scan": "detect_elements",
            "find": "detect_elements",
            "analyze": "detect_elements"
        }
        action = action_aliases.get(action, action)
        normalized_action = action

        # Ensure params is a dict; handle common non-dict cases from LLMs
        if not isinstance(params, dict):
            if action == "press_key":
                params = {"key": params}
            elif action == "type":
                params = {"text": params}
            elif action == "scroll":
                params = {"clicks": params}
            elif action == "click_coords":
                if isinstance(params, (list, tuple)) and len(params) >= 2:
                    params = {"x": params[0], "y": params[1]}
                else:
                    params = {"x": None, "y": None}
            elif action == "click_element":
                params = {"element_id": params}
            else:
                params = {"value": params}

        params = self._normalize_params(action, params)

        action_map = {
            "see": lambda: self.action_see(),
            "detect_elements": lambda: self.action_detect_elements(),
            "click_element": lambda: self.action_click_element(**params),
            "click_coords": lambda: self.action_click_coords(**params),
            "type": lambda: self.action_type(**params),
            "press_key": lambda: self.action_press_key(**params),
            "hotkey": lambda: self.action_hotkey(*[str(k) for k in params.get("keys", [])]) if params.get("keys") else self.action_hotkey(),
            "scroll": lambda: self.action_scroll(**params),
            "wait": lambda: self.action_wait(**params),
        }

        if action == "done":
            return {"success": True, "done": True}

        if action not in action_map:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Available: {list(action_map.keys())}"
            }

        result = action_map[action]()
        if isinstance(result, dict):
            result["_normalized_action"] = normalized_action
        # Track last action for progress heuristics
        self.last_action_name = normalized_action
        if isinstance(result, dict):
            self.last_action_success = bool(result.get("success")) if "success" in result else None
        else:
            self.last_action_success = None
        return result

    def get_llm_decision(
        self,
        user_goal: str,
        screenshot_path: Optional[str] = None,
        vision_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the next action from the LLM."""
        import re
        import base64
        from io import BytesIO
        from PIL import Image

        # Build message content blocks for Anthropic
        content_blocks = []

        # Add memory context
        memory_summary = self.memory.get_context_summary(max_entries=20)
        if memory_summary:
            content_blocks.append({
                "type": "text",
                "text": f"CONVERSATION HISTORY:\n{memory_summary}\n"
            })

        # Add recent action memory for better decisions
        action_memory = self._get_action_memory_summary()
        clicked_hint = ""
        if self.clicked_elements:
            clicked_hint = f"\nAlready clicked elements: {list(set(self.clicked_elements[-10:]))}. Try different elements."

        # Build current state text
        state_text = f"USER GOAL: {user_goal}\n"
        state_text += f"\nRECENT ACTIONS: {action_memory}{clicked_hint}\n"
        if vision_summary:
            state_text += f"\nVISION ANALYSIS:\n{vision_summary}\n"
        state_text += "\nWhat should I do next? Respond with JSON containing 'thought', 'action', and 'params'."

        content_blocks.append({"type": "text", "text": state_text})

        # Add image if available (resize to fit Anthropic's 5MB limit)
        if screenshot_path:
            img = Image.open(screenshot_path)

            # Calculate new dimensions to keep under 5MB (aim for ~4MB to be safe)
            # Start with max dimension of 1920 pixels (good balance of quality and size)
            max_dimension = 1920
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)

            # Save to bytes with optimization
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True, quality=85)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode("utf-8")

            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            })

        # Build messages array
        messages = [{"role": "user", "content": content_blocks}]

        # Call Anthropic API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.build_system_prompt(),
            messages=messages
        )

        # Parse response
        response_text = response.content[0].text

        # Parse JSON response (handle nested objects e.g. "params": {"x": 1, "y": 2})
        try:
            start = response_text.find("{")
            if start == -1:
                decision = {"thought": response_text, "action": None, "params": {}}
            else:
                depth = 0
                for i in range(start, len(response_text)):
                    if response_text[i] == "{":
                        depth += 1
                    elif response_text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = response_text[start : i + 1]
                            decision = json.loads(json_str)
                            break
                else:
                    decision = json.loads(response_text)
        except json.JSONDecodeError:
            decision = {
                "thought": response_text,
                "action": None,
                "params": {}
            }

        return decision

    def run(
        self,
        user_goal: str,
        max_steps: int = 20,
        delay_after_action: float = 0.5
    ) -> Dict[str, Any]:
        """Run the agent to accomplish the user's goal."""
        self.memory.add("user", user_goal, {"type": "goal"})

        print(f"\n{'='*60}")
        print(f"AGENT STARTED - Goal: {user_goal}")
        print(f"Conversation ID: {self.memory.conversation_id}")
        print(f"Model: {self.model}")
        print(f"Thinking: {'on' if self.thinking_enabled else 'off (faster)'}")
        print(f"{'='*60}\n")

        for step in range(max_steps):
            self.step_count = step + 1
            print(f"\n--- Step {self.step_count}/{max_steps} ---")

            # Determine what to show LLM
            screenshot_to_show = None
            vision_summary = None

            if self.last_vision_data:
                # We have vision data, show the annotated screenshot
                screenshot_to_show = self.last_vision_data["annotated"]
                vision_summary = self.get_ui_summary(self.last_vision_data["boxes"])
                print(f"Showing annotated screenshot with {len(self.last_vision_data['boxes'])} elements")
            else:
                # No vision data: run detect_elements first so LLM gets annotated image with element IDs
                # (Otherwise LLM would see plain screenshot and can't use click_element)
                print("Running vision analysis (vision.py) to get annotated screenshot with element IDs...")
                vision_data = self.run_vision_analysis()
                screenshot_to_show = vision_data["annotated"]
                vision_summary = self.get_ui_summary(vision_data["boxes"])
                print(f"Annotated screenshot: {vision_data['annotated']} ({len(vision_data['boxes'])} elements)")

            # Get LLM decision
            print("\nThinking...")
            decision = self.get_llm_decision(
                user_goal,
                screenshot_path=screenshot_to_show,
                vision_summary=vision_summary
            )

            thought = decision.get("thought", "")
            action = decision.get("action")
            params = decision.get("params", {})

            print(f"\nThought: {thought}")
            print(f"Action: {action}")
            if params:
                print(f"Params: {json.dumps(params, default=str)}")

            # Check if done
            if action == "done" or action is None:
                print("\n✓ Task complete!")
                self.memory.add("assistant", f"Completed: {thought}", {"step": step, "status": "done"})
                break

            # Execute action
            result = self.execute_action(action, params)
            normalized_action = result.get("_normalized_action", action) if isinstance(result, dict) else action

            # Record action for stuck detection
            self._record_action(normalized_action, params, result)

            # Handle result
            if result.get("done"):
                print("\n✓ Task complete!")
                self.memory.add("assistant", f"Completed: {thought}", {"step": step, "status": "done"})
                break

            if result.get("success"):
                msg = result.get("message", "Success")
                print(f"✓ {msg}")
                self.memory.add(
                    "assistant",
                    f"Executed {action}: {msg}",
                    {"step": step, "action": action, "result": "success"}
                )

                # Smart vision data clearing: only clear if UI likely changed
                # Take a quick screenshot to check if UI actually changed
                test_screenshot = self.take_screenshot()
                ui_changed = self._has_ui_changed(test_screenshot)

                # Clear vision data only for actions that change UI AND UI actually changed
                if normalized_action in ("click_element", "click_coords"):
                    # Click actions usually change UI - check with screenshot comparison
                    if ui_changed:
                        self.last_vision_data = None
                        print("  (UI changed - refreshing vision data)")
                    else:
                        print("  (UI unchanged - keeping vision data)")
                elif normalized_action in ("type", "press_key", "hotkey", "scroll"):
                    # These actions might change UI - clear to be safe
                    self.last_vision_data = None
                    print("  (Action may change UI - refreshing vision data)")
                # For 'wait' and 'see', keep vision data
            else:
                error = result.get("error", "Unknown error")
                print(f"✗ Error: {error}")
                self.memory.add(
                    "assistant",
                    f"Failed {action}: {error}",
                    {"step": step, "action": action, "result": "error"}
                )

            # Check if agent is stuck
            is_stuck, stuck_reason = self._is_stuck()
            if is_stuck:
                print(f"\n⚠ Agent appears stuck: {stuck_reason}")
                print("Trying alternative approach...")
                # Force vision refresh to get fresh data
                self.last_vision_data = None

            # Check if task appears complete
            if self.last_vision_data and self.last_vision_data.get("screenshot"):
                is_complete, complete_reason = self._check_task_completion(user_goal, self.last_vision_data["screenshot"])
                if is_complete:
                    print(f"\n✓ Task appears complete: {complete_reason}")
                    self.memory.add("assistant", f"Task completed: {complete_reason}", {"step": step, "status": "done"})
                    break

            # Wait for UI to update
            time.sleep(delay_after_action)

        else:
            print(f"\nReached maximum steps ({max_steps}). Stopping.")

        # Final screenshot
        final_screenshot = self.take_screenshot()
        print(f"\nFinal screenshot: {final_screenshot}")

        return {
            "conversation_id": self.memory.conversation_id,
            "steps_taken": self.step_count,
            "memory_file": str(self.memory.memory_file),
            "final_screenshot": final_screenshot
        }


def main():
    """CLI interface for the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Computer Use Agent with Claude")
    parser.add_argument("goal", help="The task to accomplish")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps")
    parser.add_argument("--conversation-id", help="Resume conversation")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Model to use")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay after actions")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking/reasoning (faster)")

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("claudekey")
    if not api_key:
        print("Error: 'claudekey' not found in .env file or environment")
        print("Make sure your .env file contains: claudekey=\"your_api_key\"")
        return

    agent = Agent(
        api_key=api_key,
        model=args.model,
        conversation_id=args.conversation_id,
        thinking_enabled=not args.no_thinking
    )

    result = agent.run(
        user_goal=args.goal,
        max_steps=args.max_steps,
        delay_after_action=args.delay
    )

    print(f"\n{'='*60}")
    print("AGENT SUMMARY")
    print(f"{'='*60}")
    print(f"Conversation ID: {result['conversation_id']}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Memory file: {result['memory_file']}")
    print(f"Final screenshot: {result['final_screenshot']}")


if __name__ == "__main__":
    main()
