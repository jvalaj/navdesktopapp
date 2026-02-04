import asyncio
import json
import os
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
    payload = {
        "type": "screenshot",
        "stepId": step_id,
        "path": path,
        "caption": caption,
        "timestamp": now_ts(),
    }
    if conversation_id is not None:
        payload["conversationId"] = conversation_id
    await broadcast(payload)


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


def shorten_title(title: str, max_words: int = 4, max_len: int = 40) -> str:
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

    assistant_message_id = str(uuid.uuid4())

    for step in range(max_steps):
        if stop_event.is_set():
            await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
            # Save to conversation memory
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
            # Emit both screenshots so user can see what the agent sees
            await emit_screenshot(step_id, vision_data.get("screenshot"), "Screenshot taken", conversation_id)
            await emit_screenshot(step_id, vision_data.get("annotated"), "Annotated screenshot with element IDs", conversation_id)

        if not allow_send_screenshots:
            screenshot_to_show = None

        decision = agent.get_llm_decision(
            prompt,
            screenshot_path=screenshot_to_show,
            vision_summary=vision_summary,
        )

        thought = decision.get("thought", "")
        action = decision.get("action")
        params = decision.get("params", {})

        for chunk in [thought[i : i + 120] for i in range(0, len(thought), 120)]:
            await emit_message_delta(assistant_message_id, "assistant", chunk, conversation_id)
            await asyncio.sleep(0.02)

        if action in (None, "done"):
            await emit_step(step_id, "Done", "Agent reported completion", conversation_id)
            # Save to conversation memory
            agent.memory.add("assistant", f"Completed: {thought}", {"step": step + 1, "status": "done"})
            break

        await emit_step(step_id, f"{action}", conversation_id=conversation_id)

        if dry_run:
            await emit_tool(step_id, f"Dry run: would execute {action} with {params}", conversation_id)
            continue

        result = agent.execute_action(action, params)
        if result.get("success"):
            await emit_tool(step_id, result.get("message", "Executed"), conversation_id)
            # Save to conversation memory
            agent.memory.add(
                "assistant",
                f"Executed {action}: {result.get('message', 'Success')}",
                {"step": step + 1, "action": action, "result": "success"}
            )
        else:
            await emit_tool(step_id, result.get("error", "Error"), conversation_id)
            # Save to conversation memory
            agent.memory.add(
                "assistant",
                f"Failed {action}: {result.get('error', 'Unknown error')}",
                {"step": step + 1, "action": action, "result": "error"}
            )

        # Emit screenshots for see and detect_elements actions so user can see what the agent is doing
        normalized_action = result.get("_normalized_action", action)

        # Only clear vision data after actions that MODIFY the UI
        # "see" and "detect_elements" just observe, so they should preserve the vision data
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
    print(f"Conversation saved to: {agent.memory.memory_file}")
    print(f"Total entries in memory: {len(agent.memory.history)}")

    await emit_status("idle")


async def handler(websocket):
    clients.add(websocket)
    try:
        async for message in websocket:
            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "run":
                global current_task
                if current_task and not current_task.done():
                    continue
                conversation_id = payload.get("conversationId") or str(uuid.uuid4())
                prompt = payload.get("prompt", "")
                settings = payload.get("settings", {})
                if payload.get("requestTitle"):
                    model = settings.get("model", "claude-sonnet-4-5-20250929")
                    asyncio.create_task(handle_title(conversation_id, prompt, model))
                current_task = asyncio.create_task(run_agent(conversation_id, prompt, settings))

            if msg_type == "stop":
                stop_event.set()

    finally:
        clients.remove(websocket)


async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT):
        print(f"NavAI agent server running on ws://127.0.0.1:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
