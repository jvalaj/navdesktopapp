"""
Mouse and Keyboard control functions for LLM agents on macOS.
These functions can be exposed as tools/actions for an LLM to interact with the system.
"""

import pyautogui
import time
from typing import Optional, Literal
from enum import Enum

# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small pause between actions

# Screen size for validation
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class ClickType(Enum):
    """Types of mouse clicks available."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


def click_at(
    x: int,
    y: int,
    click_type: str = "left",
    clicks: int = 1,
    duration: float = 0.0,
    move_delay: float = 0.0
) -> dict:
    """
    Click at specified x, y coordinates on the screen.

    Args:
        x: X coordinate (pixels from left edge)
        y: Y coordinate (pixels from top edge)
        click_type: Type of click - 'left', 'right', or 'middle'
        clicks: Number of clicks (1 for single, 2 for double, 3 for triple)
        duration: Duration of click in seconds (for drag effects)
        move_delay: Delay before clicking after moving (seconds)

    Returns:
        dict with status and message

    Example:
        click_at(100, 200, "left", 1)  # Single left click
        click_at(100, 200, "left", 2)  # Double click
        click_at(100, 200, "right", 1)  # Right click
    """
    try:
        # Validate coordinates
        if not (0 <= x <= SCREEN_WIDTH):
            return {"success": False, "error": f"X coordinate {x} out of bounds (0-{SCREEN_WIDTH})"}
        if not (0 <= y <= SCREEN_HEIGHT):
            return {"success": False, "error": f"Y coordinate {y} out of bounds (0-{SCREEN_HEIGHT})"}

        # Validate click type
        valid_click_types = ["left", "right", "middle"]
        if click_type not in valid_click_types:
            return {"success": False, "error": f"Invalid click_type '{click_type}'. Must be one of: {valid_click_types}"}

        # Move to position with optional delay
        pyautogui.moveTo(x, y, duration=move_delay if move_delay > 0 else 0)

        # Small pause to ensure position is reached
        if move_delay > 0:
            time.sleep(0.05)

        # Perform the click
        pyautogui.click(
            x=x,
            y=y,
            clicks=clicks,
            interval=0.1,
            button=click_type,
            duration=duration
        )

        return {
            "success": True,
            "message": f"Clicked {click_type} mouse button {clicks} time(s) at ({x}, {y})"
        }

    except pyautogui.FailSafeException:
        return {"success": False, "error": "Fail-safe triggered (mouse moved to corner)"}
    except Exception as e:
        return {"success": False, "error": f"Click failed: {str(e)}"}


def left_click(x: int, y: int, double: bool = False) -> dict:
    """
    Perform a left click at the specified coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        double: If True, perform double click

    Returns:
        dict with status and message
    """
    clicks = 2 if double else 1
    action = "double-clicked" if double else "clicked"
    result = click_at(x, y, "left", clicks)
    if result["success"]:
        result["message"] = f"Left {action} at ({x}, {y})"
    return result


def right_click(x: int, y: int) -> dict:
    """
    Perform a right click at the specified coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        dict with status and message
    """
    result = click_at(x, y, "right", 1)
    if result["success"]:
        result["message"] = f"Right-clicked at ({x}, {y})"
    return result


def double_click(x: int, y: int) -> dict:
    """
    Perform a double click at the specified coordinates.

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        dict with status and message
    """
    result = click_at(x, y, "left", 2)
    if result["success"]:
        result["message"] = f"Double-clicked at ({x}, {y})"
    return result


def triple_click(x: int, y: int) -> dict:
    """
    Perform a triple click at the specified coordinates (often selects entire line).

    Args:
        x: X coordinate
        y: Y coordinate

    Returns:
        dict with status and message
    """
    result = click_at(x, y, "left", 3)
    if result["success"]:
        result["message"] = f"Triple-clicked at ({x}, {y})"
    return result


def drag_to(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left"
) -> dict:
    """
    Drag from one position to another.

    Args:
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        end_x: Ending X coordinate
        end_y: Ending Y coordinate
        duration: Duration of drag in seconds
        button: Mouse button to hold ('left', 'right', 'middle')

    Returns:
        dict with status and message
    """
    try:
        # Validate coordinates
        if not (0 <= start_x <= SCREEN_WIDTH and 0 <= end_x <= SCREEN_WIDTH):
            return {"success": False, "error": "X coordinates out of bounds"}
        if not (0 <= start_y <= SCREEN_HEIGHT and 0 <= end_y <= SCREEN_HEIGHT):
            return {"success": False, "error": "Y coordinates out of bounds"}

        # Perform the drag
        pyautogui.dragTo(
            end_x,
            end_y,
            duration=duration,
            button=button
        )

        return {
            "success": True,
            "message": f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
        }

    except Exception as e:
        return {"success": False, "error": f"Drag failed: {str(e)}"}


def type_text(
    text: str,
    interval: float = 0.0,
    delay: float = 0.0
) -> dict:
    """
    Type text at the current cursor position.

    Args:
        text: The text to type
        interval: Delay between each keystroke (seconds)
        delay: Delay before starting to type (seconds)

    Returns:
        dict with status and message

    Example:
        type_text("Hello, World!")
        type_text("Hello", interval=0.1)  # Slower typing
    """
    try:
        # Wait before typing if delay specified
        if delay > 0:
            time.sleep(delay)

        # Type the text
        pyautogui.write(text, interval=interval)

        return {
            "success": True,
            "message": f"Typed text: '{text}'"
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to type text: {str(e)}"}


def press_key(key: str, presses: int = 1, interval: float = 0.0) -> dict:
    """
    Press a keyboard key.

    Args:
        key: The key to press (e.g., 'enter', 'tab', 'escape', 'command', 'shift')
        presses: Number of times to press the key
        interval: Delay between presses (seconds)

    Returns:
        dict with status and message

    Common keys:
        - Modifier keys: 'cmd', 'command', 'ctrl', 'control', 'shift', 'alt', 'option'
        - Navigation: 'up', 'down', 'left', 'right', 'home', 'end'
        - Action keys: 'enter', 'return', 'tab', 'escape', 'esc', 'space'
        - Function keys: 'f1', 'f2', ..., 'f12'
        - Special: 'backspace', 'delete', 'pageup', 'pagedown'

    Example:
        press_key("enter")
        press_key("tab", presses=3)
    """
    try:
        pyautogui.press(key, presses=presses, interval=interval)

        key_display = key if presses == 1 else f"{key} x{presses}"

        return {
            "success": True,
            "message": f"Pressed key: {key_display}"
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to press key: {str(e)}"}


def hotkey(*keys: str, interval: float = 0.0) -> dict:
    """
    Press a combination of keys simultaneously (hotkey/key combo).

    Args:
        *keys: Variable number of keys to press together
        interval: Delay between key presses (seconds)

    Returns:
        dict with status and message

    Common macOS hotkeys:
        - ('cmd', 'c') - Copy
        - ('cmd', 'v') - Paste
        - ('cmd', 'a') - Select all
        - ('cmd', 'x') - Cut
        - ('cmd', 'z') - Undo
        - ('cmd', 'shift', 'z') - Redo
        - ('cmd', 'w') - Close window
        - ('cmd', 'q') - Quit application
        - ('cmd', 'space') - Spotlight search
        - ('cmd', 'tab') - Switch applications

    Example:
        hotkey("cmd", "c")  # Copy
        hotkey("cmd", "shift", "4")  # Screenshot
    """
    try:
        pyautogui.hotkey(*keys, interval=interval)

        key_combo = " + ".join(keys)

        return {
            "success": True,
            "message": f"Pressed hotkey: {key_combo}"
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to press hotkey: {str(e)}"}


def scroll(
    clicks: int,
    x: Optional[int] = None,
    y: Optional[int] = None
) -> dict:
    """
    Scroll the mouse wheel.

    Args:
        clicks: Number of scroll clicks (positive = up, negative = down)
        x: X coordinate to scroll at (optional)
        y: Y coordinate to scroll at (optional)

    Returns:
        dict with status and message

    Example:
        scroll(10)  # Scroll up
        scroll(-5)  # Scroll down
        scroll(10, x=500, y=500)  # Scroll at specific position
    """
    try:
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x=x, y=y)
            position = f"at ({x}, {y})"
        else:
            pyautogui.scroll(clicks)
            position = "at current position"

        direction = "up" if clicks > 0 else "down"

        return {
            "success": True,
            "message": f"Scrolled {direction} {abs(clicks)} clicks {position}"
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to scroll: {str(e)}"}


def get_mouse_position() -> dict:
    """
    Get the current mouse position.

    Returns:
        dict with x, y coordinates
    """
    try:
        x, y = pyautogui.position()
        return {
            "success": True,
            "x": x,
            "y": y,
            "message": f"Current mouse position: ({x}, {y})"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get position: {str(e)}"}


def get_screen_size() -> dict:
    """
    Get the screen dimensions.

    Returns:
        dict with width and height
    """
    return {
        "success": True,
        "width": SCREEN_WIDTH,
        "height": SCREEN_HEIGHT,
        "message": f"Screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}"
    }


# ============================================================================
# FUNCTIONS TO EXPOSE TO LLM (as available actions/tools)
# ============================================================================

"""
Available actions to expose to LLM:

1. left_click(x, y, double=False) - Left click or double-click at coordinates
2. right_click(x, y) - Right click at coordinates
3. double_click(x, y) - Double click at coordinates
4. triple_click(x, y) - Triple click at coordinates
5. drag_to(start_x, start_y, end_x, end_y, duration=0.5) - Drag from one point to another
6. type_text(text, interval=0.0) - Type text at cursor position
7. press_key(key, presses=1) - Press a keyboard key
8. hotkey(*keys) - Press key combination (e.g., cmd+c)
9. scroll(clicks, x=None, y=None) - Scroll up/down
10. get_mouse_position() - Get current mouse position
11. get_screen_size() - Get screen dimensions
12. click_at(x, y, click_type, clicks) - Generic click function

Example tool definition for LLM:

{
    "name": "left_click",
    "description": "Perform a left click at specified screen coordinates",
    "parameters": {
        "x": {"type": "integer", "description": "X coordinate in pixels"},
        "y": {"type": "integer", "description": "Y coordinate in pixels"},
        "double": {"type": "boolean", "description": "True for double click, False for single click"}
    }
}
"""


if __name__ == "__main__":
    # Example usage and testing
    print("Testing mouse and keyboard functions...")
    print(f"Screen size: {get_screen_size()}")

    # Example: Get mouse position
    print(f"\n{get_mouse_position()}")

    # Example: Click
    # print(left_click(500, 500))

    # Example: Type text
    # print(type_text("Hello, World!"))

    # Example: Press key
    # print(press_key("enter"))

    print("\nAll functions loaded successfully!")
    print("\nNote: To use these functions, install pyautogui:")
    print("  pip install pyautogui")
