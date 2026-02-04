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
try:
    from AppKit import NSApplication, NSActivationPolicyAccessory
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSActivationPolicyAccessory)  # No dock icon
except Exception:
    pass  # AppKit not available or failed - not critical

# Screen capture settings
SCREENSHOT_DIR = Path("screenshots")
MEMORY_DIR = Path("conversations")

# Create directories
SCREENSHOT_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)


class ConversationMemory:
    """Manages conversation memory as a text file."""

    def __init__(self, conversation_id: Optional[str] = None):
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_id = conversation_id
        self.memory_file = MEMORY_DIR / f"{conversation_id}.txt"
        self.history: List[Dict[str, str]] = []
        self._load_existing()

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

    def _infer_single_scale(self, shot_width: int, shot_height: int, screen_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer a single scale factor from screenshot-space to PyAutoGUI-space.
        Prefers explicit logical/physical matches; falls back to size ratio.
        """
        def _match_dims(a, b, tol=2):
            return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

        shot = (int(shot_width), int(shot_height))
        logical = (int(screen_info["logical_width"]), int(screen_info["logical_height"]))
        physical = (int(screen_info["physical_width"]), int(screen_info["physical_height"]))
        pyauto = (int(screen_info["pyauto_width"]), int(screen_info["pyauto_height"]))
        scale_factor = float(screen_info.get("scale_factor", 1.0) or 1.0)

        shot_space = "logical" if _match_dims(shot, logical) else "physical" if _match_dims(shot, physical) else "unknown"
        pyauto_space = "logical" if _match_dims(pyauto, logical) else "physical" if _match_dims(pyauto, physical) else "unknown"

        reason = "fallback_ratio"
        if shot_space != "unknown" and pyauto_space == shot_space:
            scale = 1.0
            reason = "shot=pyauto_space"
        elif shot_space == "logical" and pyauto_space == "physical":
            scale = scale_factor
            reason = "logical_to_physical"
        elif shot_space == "physical" and pyauto_space == "logical":
            scale = 1.0 / scale_factor if scale_factor != 0 else 1.0
            reason = "physical_to_logical"
        else:
            sx = pyauto[0] / shot[0] if shot[0] else 1.0
            sy = pyauto[1] / shot[1] if shot[1] else 1.0
            scale = (sx + sy) / 2.0

        return {
            "scale": float(scale),
            "reason": reason,
            "shot_space": shot_space,
            "pyauto_space": pyauto_space,
            "shot": shot,
            "pyauto": pyauto,
            "logical": logical,
            "physical": physical,
            "scale_factor": scale_factor,
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
            "screen_info": screen_info
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
        summary.append(f"Detected {len(boxes)} UI elements:\n")

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

        scale_info = self._infer_single_scale(shot_width, shot_height, screen_info)
        scale = scale_info["scale"]

        # Get click coordinates from vision data and scale to screen coordinates
        cx = box.get("click_x", box["cx"])
        cy = box.get("click_y", box["cy"])
        scaled_x = round(cx * scale)
        scaled_y = round(cy * scale)
        box_type = box.get("type", "unknown")
        text = box.get("text", "")

        if scale_info["reason"] != "shot=pyauto_space":
            print(
                "Scale mapping: "
                f"scale={scale_info['scale']:.3f} "
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
            result["message"] = f"Clicked element [{element_id}] ({box_type}) - original coords: ({cx}, {cy}) -> clicked at: ({scaled_x}, {scaled_y})"
            if text:
                result["message"] += f" '{text}'"

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
        scale_info = self._infer_single_scale(shot_width, shot_height, screen_info)
        scale = scale_info["scale"] if scale_info["scale"] != 0 else 1.0

        # Map screen coords -> screenshot coords
        shot_x = int(round(x / scale))
        shot_y = int(round(y / scale))
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

    def build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return f"""You are a computer use agent that can see and interact with a macOS computer screen.

SCREEN SIZE: {typeandclick.SCREEN_WIDTH}x{typeandclick.SCREEN_HEIGHT} pixels

AVAILABLE ACTIONS:

1. see - Take a regular screenshot and show it to you
   Use this when you just need to look at the screen

2. detect_elements - Find all clickable elements on screen
   VITAL: This shows you WHAT you can click with NUMBERED IDs
   Returns: Annotated screenshot with each element labeled 0, 1, 2, 3, etc.
   Use this BEFORE clicking anything - it tells you what element_id numbers are available
   Example: If you see "[5]" on a button in the annotated image, use {{"element_id": 5}} to click it

3. click_element(element_id, click_type) - Click on a UI element by its ID
   element_id: JUST the number (integer) shown on the annotated image - e.g., 5, 12, 0
   click_type: 'left' (default), 'right' (right-click), or 'double'
   SHORTCUTS: You can also use "left_click", "right_click", "double_click" as actions directly
   Examples:
   - Left click: {{"action": "click_element", "params": {{"element_id": 5}}}}
   - Right click: {{"action": "right_click", "params": {{"element_id": 5}}}}
   - Double click: {{"action": "double_click", "params": {{"element_id": 5}}}}
   CRITICAL: element_id MUST be a plain number like {{"element_id": 5}}
   WRONG: {{"element_id": "[5] ui_rect..."}} or {{"element": "[5]..."}}
   CORRECT: {{"element_id": 5}}

4. click_coords(x, y, click_type) - Click at exact coordinates
   x, y: Pixel coordinates from top-left
   click_type: 'left', 'right', or 'double' (default: 'left')

5. type(text) - Type text at current cursor position

6. press_key(key, presses) - Press a keyboard key
   key: e.g., 'enter', 'tab', 'escape', 'cmd', 'shift', 'space'
   presses: number of times (default: 1)

7. hotkey(*keys) - Press key combination
   Example: ["cmd", "c"] for copy, ["cmd", "v"] for paste

8. scroll(clicks) - Scroll mouse wheel
   clicks: positive=up, negative=down

9. wait(seconds) - Pause for N seconds (e.g. {{"seconds": 2}})
   Use when you need to let the UI settle before the next action

10. done - Task is complete

WORKFLOW:
- To see the screen: use "see"
- To click something you MUST:
  a) First use "detect_elements" - this returns an annotated screenshot with numbered IDs on each element
  b) Look at the annotated image to find the NUMBER (e.g. 5, 12, 0) on the element you want
  c) Use "click_element" with params {{"element_id": 5}} - JUST the number, nothing else
- CRITICAL: element_id is ALWAYS a plain integer like 0, 1, 2, 5, 10, etc.
- NEVER include the description text with element_id - just extract the number
- If unsure what to do, use "see" or "detect_elements" - do NOT use "wait" unless the UI needs time to load
- For typing/keyboard: use "type", "press_key", or "hotkey"

RESPONSE FORMAT:
Respond with JSON:
{{
    "thought": "Your reasoning",
    "action": "action_name",
    "params": {{"param": "value"}}  // omit for 'see', 'detect_elements', 'done'
}}

Example:
{{
    "thought": "I need to click on the Safari button. First let me use detect_elements to see all clickable items with their IDs.",
    "action": "detect_elements",
    "params": {{}}
}}

Then after seeing the annotated image:
{{
    "thought": "I can see the Safari button is element 5 at position (245, 12)",
    "action": "click_element",
    "params": {{"element_id": 5, "click_type": "left"}}
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

        # Build current state text
        state_text = f"USER GOAL: {user_goal}\n"
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

                # Clear vision data after any action that likely changes the UI
                if normalized_action in ("see", "click_element", "click_coords", "type", "press_key", "hotkey", "scroll"):
                    self.last_vision_data = None
            else:
                error = result.get("error", "Unknown error")
                print(f"✗ Error: {error}")
                self.memory.add(
                    "assistant",
                    f"Failed {action}: {error}",
                    {"step": step, "action": action, "result": "error"}
                )

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
