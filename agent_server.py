import asyncio
import json
import os
import re
import uuid
from datetime import datetime

try:
    import websockets
except Exception as exc:
    raise SystemExit(
        "Missing dependency 'websockets'. Install with: pip install websockets"
    ) from exc

from agent import Agent
from anthropic import Anthropic

PORT = int(os.environ.get("NAVAI_SERVER_PORT", "8765"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SCREENSHOTS_DIR = (os.environ.get("NAVAI_SCREENSHOTS_DIR") or "").strip()

clients = set()
current_task = None
stop_event = asyncio.Event()


def now_ts():
    return datetime.now().strftime("%H:%M:%S")


async def broadcast(payload):
    if not clients:
        return
    message = json.dumps(payload)
    await asyncio.gather(*[client.send(message) for client in clients])


async def emit_status(state):
    await broadcast({"type": "status", "state": state})


async def emit_step(step_id, title, caption=None, conversation_id=None):
    payload = {
        "type": "step",
        "stepId": step_id,
        "title": title,
        "caption": caption,
        "timestamp": now_ts(),
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    await broadcast(payload)


async def emit_screenshot(step_id, path, caption=None, conversation_id=None):
    normalized_path = _normalize_screenshot_path(path)
    payload = {
        "type": "screenshot",
        "stepId": step_id,
        "path": normalized_path,
        "caption": caption,
        "timestamp": now_ts(),
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    await broadcast(payload)

def _choose_one_screenshot_path(primary: str = None, secondary: str = None) -> str:
    """
    Enforce: max 1 screenshot emitted per action/step.
    Prefer annotated images when available.
    """
    if primary:
        return primary
    if secondary:
        return secondary
    return None


def _normalize_screenshot_path(path: str = None) -> str:
    """Return a stable absolute screenshot path for renderer consumption."""
    if not path:
        return ""
    p = os.path.expanduser(str(path).strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    if SCREENSHOTS_DIR:
        # Use basename to prevent duplicate/incorrect nested relative segments.
        return os.path.normpath(os.path.join(SCREENSHOTS_DIR, os.path.basename(p)))
    return os.path.normpath(os.path.abspath(p))


async def emit_tool(step_id, caption=None, conversation_id=None):
    payload = {
        "type": "tool",
        "stepId": step_id,
        "caption": caption,
        "timestamp": now_ts(),
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    await broadcast(payload)


async def emit_message_delta(message_id, role, text, conversation_id=None):
    payload = {
        "type": "message_delta",
        "messageId": message_id,
        "role": role,
        "delta": text,
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    await broadcast(payload)

def fallback_title(prompt: str) -> str:
    words = prompt.strip().split()[:4]
    if not words:
        return "New conversation"
    title = " ".join(words)
    return title[:40] + "..." if len(title) > 40 else title + "..."


async def generate_title(prompt: str, model: str) -> str:
    if not API_KEY:
        return fallback_title(prompt)

    def _call():
        client = Anthropic(api_key=API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=24,
            system="Create a concise 3-5 word title for this chat. Return only the title.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return text.strip("\"'“”")

    try:
        title = await asyncio.to_thread(_call)
        return title or fallback_title(prompt)
    except Exception:
        return fallback_title(prompt)


async def emit_title(conversation_id: str, title: str):
    await broadcast({"type": "title", "conversationId": conversation_id, "title": title})

def sanitize_title(title: str) -> str:
    """
    Normalize titles coming from prompts/LLMs (e.g. strip leading '# ' headings).
    """
    if title is None:
        return ""
    s = str(title).strip()
    s = s.strip("\"'“”").strip()
    s = re.sub(r"^\s*#{1,6}\s*", "", s)   # markdown headings
    s = re.sub(r"^\s*[-*]\s+", "", s)     # markdown bullets
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shorten_title(title: str, max_words: int = 4, max_len: int = 40) -> str:
    title = sanitize_title(title)
    words = title.strip().split()[:max_words]
    if not words:
        return "New conversation"
    s = " ".join(words)
    return (s[: max_len - 3] + "...") if len(s) > max_len else (s + "...")

async def handle_title(conversation_id: str, prompt: str, model: str):
    title = await generate_title(prompt, model)
    await emit_title(conversation_id, shorten_title(title))


async def run_agent(conversation_id, prompt, settings):
    stop_event.clear()
    await emit_status("running")

    agent = None
    assistant_message_id = str(uuid.uuid4())
    try:
        # Persist the conversation immediately, even if the run is stopped very early.
        agent = Agent(
            api_key=API_KEY,
            model=settings.get("model", "claude-sonnet-4-5-20250929"),
            conversation_id=conversation_id,
            thinking_enabled=True,
        )

        max_steps = settings.get("maxSteps", 20)
        delay_after_action = settings.get("delay", 0.5)
        allow_send_screenshots = settings.get("allowSendScreenshots", True)
        dry_run = settings.get("dryRun", False)

        # Save the user's prompt to conversation memory
        agent.memory.add("user", prompt, {"type": "goal"})

        for step in range(max_steps):
            if stop_event.is_set():
                await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
                agent.memory.add("assistant", "Stopped by user", {"step": step + 1, "status": "stopped"})
                break

            step_id = f"{conversation_id}-{step + 1}"
            await emit_step(step_id, "Thinking", conversation_id=conversation_id)

            screenshot_to_show = None
            vision_summary = None

            if agent.last_vision_data:
                screenshot_to_show = agent.last_vision_data.get("annotated")
                vision_summary = agent.get_ui_summary(agent.last_vision_data.get("boxes", []))
            else:
                await emit_step(step_id, "Taking screenshot and analyzing UI")
                vision_data = agent.run_vision_analysis()
                screenshot_to_show = vision_data.get("annotated")
                vision_summary = agent.get_ui_summary(vision_data.get("boxes", []))
                # Emit only ONE screenshot (prefer annotated)
                chosen = _choose_one_screenshot_path(
                    primary=vision_data.get("annotated"),
                    secondary=vision_data.get("screenshot"),
                )
                if chosen:
                    await emit_screenshot(step_id, chosen, "Screen snapshot", conversation_id)

            if not allow_send_screenshots:
                screenshot_to_show = None

            # LLM call is synchronous; run it in a worker thread so Stop can cancel this coroutine immediately.
            decision = await asyncio.to_thread(
                agent.get_llm_decision,
                prompt,
                screenshot_path=screenshot_to_show,
                vision_summary=vision_summary,
            )

            if stop_event.is_set():
                await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
                agent.memory.add("assistant", "Stopped by user", {"step": step + 1, "status": "stopped"})
                break

            thought = decision.get("thought", "")
            action = decision.get("action")
            params = decision.get("params", {})

            for chunk in [thought[i : i + 120] for i in range(0, len(thought), 120)]:
                if stop_event.is_set():
                    await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
                    agent.memory.add("assistant", "Stopped by user", {"step": step + 1, "status": "stopped"})
                    break
                await emit_message_delta(assistant_message_id, "assistant", chunk, conversation_id)
                await asyncio.sleep(0.02)

            if stop_event.is_set():
                break

            if action in (None, "done"):
                await emit_step(step_id, "Done", "Agent reported completion", conversation_id)
                agent.memory.add("assistant", f"Completed: {thought}", {"step": step + 1, "status": "done"})
                break

            await emit_step(step_id, f"{action}", conversation_id=conversation_id)

            if dry_run:
                await emit_tool(step_id, f"Dry run: would execute {action} with {params}", conversation_id)
                continue

            result = agent.execute_action(action, params)
            if result.get("success"):
                await emit_tool(step_id, result.get("message", "Executed"), conversation_id)
                agent.memory.add(
                    "assistant",
                    f"Executed {action}: {result.get('message', 'Success')}",
                    {"step": step + 1, "action": action, "result": "success"},
                )
            else:
                await emit_tool(step_id, result.get("error", "Error"), conversation_id)
                agent.memory.add(
                    "assistant",
                    f"Failed {action}: {result.get('error', 'Unknown error')}",
                    {"step": step + 1, "action": action, "result": "error"},
                )

            normalized_action = result.get("_normalized_action", action)

            # Emit at most ONE screenshot per action result (prefer annotated)
            chosen = _choose_one_screenshot_path(
                primary=result.get("annotated_screenshot"),
                secondary=result.get("screenshot"),
            )
            if chosen:
                await emit_screenshot(step_id, chosen, "Action snapshot", conversation_id)

            # Only clear vision data after actions that MODIFY the UI
            if normalized_action in (
                "click_element",
                "click_coords",
                "type",
                "press_key",
                "hotkey",
                "scroll",
            ):
                agent.last_vision_data = None

            await asyncio.sleep(delay_after_action)

        # Log final conversation save status
        if agent is not None:
            print(f"Conversation saved to: {agent.memory.memory_file}")
            print(f"Total entries in memory: {len(agent.memory.history)}")
    except asyncio.CancelledError:
        # If Stop cancels the running task, exit quickly.
        try:
            await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
        except Exception:
            pass
        if agent is not None:
            try:
                agent.memory.add("assistant", "Stopped by user", {"status": "stopped"})
            except Exception:
                pass
        raise
    finally:
        await emit_status("idle")


async def handler(websocket):
    global current_task
    clients.add(websocket)
    try:
        async for message in websocket:
            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "run":
                if current_task and not current_task.done():
                    continue
                conversation_id = payload.get("conversationId") or str(uuid.uuid4())
                prompt = payload.get("prompt", "")
                settings = payload.get("settings", {})
                if payload.get("requestTitle"):
                    model = settings.get("model", "claude-sonnet-4-5-20250929")
                    asyncio.create_task(handle_title(conversation_id, prompt, model))
                current_task = asyncio.create_task(run_agent(conversation_id, prompt, settings))

            elif msg_type == "stop":
                stop_event.set()
                # Cancel the running task so UI stops immediately (even mid-chunk / mid-wait).
                if current_task and not current_task.done():
                    current_task.cancel()
                await emit_status("idle")

    finally:
        clients.remove(websocket)


async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT):
        print(f"NavAI agent server running on ws://127.0.0.1:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
