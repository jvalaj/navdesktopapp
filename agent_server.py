import asyncio
import json
import logging
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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("MODEL_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ZAI_API_KEY = os.environ.get("ZAI_API_KEY")
SCREENSHOTS_DIR = (os.environ.get("NAVAI_SCREENSHOTS_DIR") or "").strip()

clients = set()
current_task = None
stop_event = asyncio.Event()
LOGGER = logging.getLogger("navai.agent_server")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_event(event: str, **fields):
    payload = {
        "component": "agent_server",
        "event": event,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(fields)
    LOGGER.info(json.dumps(payload, default=str))


def now_ts():
    return datetime.now().strftime("%H:%M:%S")


async def broadcast(payload):
    if not clients:
        return
    message = json.dumps(payload)
    targets = list(clients)
    results = await asyncio.gather(
        *[client.send(message) for client in targets],
        return_exceptions=True,
    )
    for client, result in zip(targets, results):
        if isinstance(result, Exception):
            log_event(
                "websocket_send_failed",
                error=str(result),
                client=getattr(client, "id", None),
                payload_type=payload.get("type"),
            )
            clients.discard(client)


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


async def emit_decision_mode(mode, conversation_id=None):
    payload = {
        "type": "decision_mode",
        "mode": mode,
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
    log_event("run_started", conversation_id=conversation_id, max_steps=settings.get("maxSteps", 20))

    agent = None
    assistant_message_id = str(uuid.uuid4())
    try:
        # Persist the conversation immediately, even if the run is stopped very early.
        model_provider = settings.get("modelProvider", os.environ.get("NAVAI_MODEL_PROVIDER", "anthropic"))
        model_base_url = settings.get("modelBaseUrl", os.environ.get("MODEL_BASE_URL"))
        provider_name = str(model_provider or "anthropic").strip().lower()
        if provider_name in ("openai_compatible", "local"):
            provider_name = "openai"
        if provider_name == "openai":
            fallback_api_key = OPENAI_API_KEY
        elif provider_name == "gemini":
            fallback_api_key = GEMINI_API_KEY
        elif provider_name == "zai":
            fallback_api_key = ZAI_API_KEY
        else:
            fallback_api_key = API_KEY
        model_api_key = settings.get("modelApiKey") or fallback_api_key
        model_timeout_seconds = float(settings.get("llmTimeoutSeconds", 60))
        agent = Agent(
            api_key=model_api_key,
            model=settings.get("model", "claude-sonnet-4-5-20250929"),
            model_provider=model_provider,
            model_base_url=model_base_url,
            model_timeout_s=model_timeout_seconds,
            conversation_id=conversation_id,
            thinking_enabled=True,
        )

        max_steps = settings.get("maxSteps", 20)
        delay_after_action = settings.get("delay", 0.5)
        allow_send_screenshots = settings.get("allowSendScreenshots", True)
        dry_run = settings.get("dryRun", False)
        llm_timeout_seconds = float(settings.get("llmTimeoutSeconds", model_timeout_seconds))
        verification_every_n_steps = max(
            1,
            int(
                settings.get(
                    "verificationEveryNSteps",
                    settings.get("screenshotFrequency", 3),
                )
            ),
        )
        max_no_advance_checks = settings.get("maxStagnantSteps", 4)
        max_stuck_signals = settings.get("maxStuckSignals", 2)
        no_advance_checks = 0
        stuck_signals = 0
        fallback_steps_remaining = 0
        verifier_recovery_attempts = 0
        max_verifier_recovery_attempts = 3
        latest_verifier_feedback = None

        # Save the user's prompt to conversation memory
        agent.memory.add("user", prompt, {"type": "goal"})

        async def run_verifier_check(step_id: str, step_number: int, reason: str, force: bool = True):
            """Run verifier in worker thread with timeout protection."""
            if not agent.last_vision_data or not agent.last_vision_data.get("screenshot"):
                return None
            screenshot_path = agent.last_vision_data.get("screenshot")
            summary = agent.get_ui_summary(agent.last_vision_data.get("boxes", []))
            verifier_timeout = max(5.0, min(llm_timeout_seconds, 45.0))
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        agent.verify_task_state,
                        prompt,
                        screenshot_path,
                        summary,
                        force,
                        step_number,
                        reason,
                    ),
                    timeout=verifier_timeout,
                )
            except asyncio.TimeoutError:
                msg = f"Verifier timed out after {verifier_timeout:.0f}s."
                log_event(
                    "verifier_timeout",
                    conversation_id=conversation_id,
                    step=step_number,
                    timeout_seconds=verifier_timeout,
                    reason=reason,
                )
                await emit_tool(step_id, msg, conversation_id)
                return {
                    "ran": False,
                    "goal_satisfied": False,
                    "task_advanced": None,
                    "missing": "",
                    "recommended_recovery": "retry",
                    "reason": msg,
                }
            except Exception as exc:
                msg = f"Verifier error: {str(exc)}"
                log_event(
                    "verifier_failed",
                    conversation_id=conversation_id,
                    step=step_number,
                    reason=reason,
                    error=str(exc),
                )
                await emit_tool(step_id, msg, conversation_id)
                return {
                    "ran": False,
                    "goal_satisfied": False,
                    "task_advanced": None,
                    "missing": "",
                    "recommended_recovery": "retry",
                    "reason": msg,
                }

        for step in range(max_steps):
            if stop_event.is_set():
                await emit_message_delta(assistant_message_id, "assistant", "\nStopped by user.", conversation_id)
                agent.memory.add("assistant", "Stopped by user", {"step": step + 1, "status": "stopped"})
                break

            step_id = f"{conversation_id}-{step + 1}"
            await emit_step(step_id, "Thinking", conversation_id=conversation_id)

            screenshot_to_show = None
            vision_summary = None
            forced_recovery_decision = None

            # Always refresh vision per step so IDs and OCR reflect the current UI.
            await emit_step(step_id, "Taking screenshot and analyzing UI")
            vision_data = agent.run_vision_analysis()
            allow_element_fallback = fallback_steps_remaining > 0
            raw_screenshot = vision_data.get("screenshot")
            annotated_screenshot = vision_data.get("annotated")
            screenshot_to_show = annotated_screenshot if allow_element_fallback else raw_screenshot
            vision_summary = agent.get_ui_summary(vision_data.get("boxes", []))
            llm_vision_summary = vision_summary if allow_element_fallback else None
            await emit_decision_mode(
                "fallback_with_element_ids" if allow_element_fallback else "raw_screenshot_only",
                conversation_id,
            )
            # Emit only ONE screenshot (prefer annotated)
            chosen = _choose_one_screenshot_path(
                primary=screenshot_to_show,
                secondary=raw_screenshot,
            )
            if chosen:
                await emit_screenshot(step_id, chosen, "Screen snapshot", conversation_id)

            if (step + 1) % verification_every_n_steps == 0:
                await emit_step(step_id, "Verifying task status", conversation_id=conversation_id)
                periodic_verifier = await run_verifier_check(
                    step_id=step_id,
                    step_number=step + 1,
                    reason="periodic",
                    force=True,
                )
                if periodic_verifier:
                    latest_verifier_feedback = periodic_verifier
                    if periodic_verifier.get("goal_satisfied"):
                        reason = periodic_verifier.get("reason", "Verifier confirmed goal completion.")
                        await emit_step(step_id, "Done", reason, conversation_id)
                        agent.memory.add(
                            "assistant",
                            f"Completed (verified): {reason}",
                            {"step": step + 1, "status": "done", "reason": "verifier_satisfied"},
                        )
                        break

                    if periodic_verifier.get("task_advanced") is False:
                        no_advance_checks += 1
                    elif periodic_verifier.get("task_advanced") is True:
                        no_advance_checks = 0

                    if no_advance_checks >= max_no_advance_checks:
                        forced_recovery_decision = agent.choose_recovery_action(periodic_verifier)
                        if forced_recovery_decision and verifier_recovery_attempts < max_verifier_recovery_attempts:
                            verifier_recovery_attempts += 1
                            no_advance_checks = 0
                            fallback_steps_remaining = max(fallback_steps_remaining, 2)
                            await emit_tool(
                                step_id,
                                (
                                    "Verifier recovery selected: "
                                    f"{forced_recovery_decision.get('action')} "
                                    f"({periodic_verifier.get('recommended_recovery', 'retry')})"
                                ),
                                conversation_id,
                            )
                        else:
                            reason = (
                                "Blocked: verifier reports no task advancement "
                                f"for {max_no_advance_checks} checks. "
                                f"Missing: {periodic_verifier.get('missing', periodic_verifier.get('reason', 'unknown'))}"
                            )
                            log_event(
                                "run_blocked",
                                conversation_id=conversation_id,
                                step=step + 1,
                                reason_code="no_task_advance",
                                no_advance_checks=no_advance_checks,
                                verifier_reason=periodic_verifier.get("reason"),
                                verifier_missing=periodic_verifier.get("missing"),
                            )
                            await emit_step(step_id, "Blocked", reason, conversation_id)
                            agent.memory.add(
                                "assistant",
                                reason,
                                {"step": step + 1, "status": "blocked", "reason": "no_task_advance"},
                            )
                            break

            if not allow_send_screenshots:
                screenshot_to_show = None

            if forced_recovery_decision:
                decision = forced_recovery_decision
            else:
                # LLM call is synchronous; run it in a worker thread so Stop can cancel this coroutine immediately.
                try:
                    decision = await asyncio.wait_for(
                        asyncio.to_thread(
                            agent.get_llm_decision,
                            prompt,
                            screenshot_path=screenshot_to_show,
                            vision_summary=llm_vision_summary,
                            verifier_feedback=latest_verifier_feedback,
                            allow_element_fallback=allow_element_fallback,
                        ),
                        timeout=max(5.0, llm_timeout_seconds),
                    )
                except asyncio.TimeoutError:
                    reason = f"Blocked: model decision timed out after {llm_timeout_seconds:.0f}s."
                    log_event(
                        "run_blocked",
                        conversation_id=conversation_id,
                        step=step + 1,
                        reason_code="llm_timeout",
                        timeout_seconds=llm_timeout_seconds,
                    )
                    await emit_step(step_id, "Blocked", reason, conversation_id)
                    await emit_tool(step_id, reason, conversation_id)
                    agent.memory.add(
                        "assistant",
                        reason,
                        {"step": step + 1, "status": "blocked", "reason": "llm_timeout"},
                    )
                    break

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
                done_verifier = await run_verifier_check(
                    step_id=step_id,
                    step_number=step + 1,
                    reason="before_done",
                    force=True,
                )
                if done_verifier:
                    latest_verifier_feedback = done_verifier
                if done_verifier and done_verifier.get("goal_satisfied"):
                    done_reason = done_verifier.get("reason", "Verifier confirmed completion.")
                    await emit_step(step_id, "Done", done_reason, conversation_id)
                    agent.memory.add(
                        "assistant",
                        f"Completed (verified): {done_reason}",
                        {"step": step + 1, "status": "done", "reason": "verifier_satisfied"},
                    )
                    break

                recovery = agent.choose_recovery_action(done_verifier)
                if recovery:
                    done_verifier_meta = done_verifier or {}
                    await emit_tool(
                        step_id,
                        (
                            "Verifier rejected done; recovery action: "
                            f"{recovery.get('action')} ({done_verifier_meta.get('recommended_recovery', 'retry')})"
                        ),
                        conversation_id,
                    )
                    thought = recovery.get("thought", thought)
                    action = recovery.get("action")
                    params = recovery.get("params", {})
                else:
                    done_verifier_meta = done_verifier or {}
                    reason = (
                        "Blocked: completion not verified. "
                        f"Missing: {done_verifier_meta.get('missing', done_verifier_meta.get('reason', 'unknown'))}"
                    )
                    log_event(
                        "run_blocked",
                        conversation_id=conversation_id,
                        step=step + 1,
                        reason_code="completion_not_verified",
                        verifier_reason=(done_verifier or {}).get("reason"),
                        verifier_missing=(done_verifier or {}).get("missing"),
                    )
                    await emit_step(step_id, "Blocked", reason, conversation_id)
                    agent.memory.add(
                        "assistant",
                        reason,
                        {"step": step + 1, "status": "blocked", "reason": "completion_not_verified"},
                    )
                    break

            await emit_step(step_id, f"{action}", conversation_id=conversation_id)

            if dry_run:
                await emit_tool(step_id, f"Dry run: would execute {action} with {params}", conversation_id)
                continue

            result = agent.execute_action(action, params)
            normalized_action = result.get("_normalized_action", action)
            agent._record_action(
                normalized_action,
                params if isinstance(params, dict) else {"value": params},
                result,
            )
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

            is_stuck, stuck_reason = agent._is_stuck()
            if is_stuck:
                stuck_signals += 1
                await emit_tool(step_id, f"Potential loop detected: {stuck_reason}", conversation_id)
                agent.last_vision_data = None
                fallback_steps_remaining = max(fallback_steps_remaining, 2)
                if stuck_signals >= max_stuck_signals:
                    reason = f"Blocked: {stuck_reason}"
                    log_event(
                        "run_blocked",
                        conversation_id=conversation_id,
                        step=step + 1,
                        reason_code="stuck_loop",
                        details=stuck_reason,
                        stuck_signals=stuck_signals,
                    )
                    await emit_step(step_id, "Blocked", reason, conversation_id)
                    agent.memory.add(
                        "assistant",
                        reason,
                        {"step": step + 1, "status": "blocked", "reason": "stuck_loop"},
                    )
                    break
            else:
                stuck_signals = 0

            await asyncio.sleep(delay_after_action)
            if fallback_steps_remaining > 0:
                fallback_steps_remaining -= 1

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
                log_event("stop_requested", conversation_id=payload.get("conversationId"))
                # Cancel the running task so UI stops immediately (even mid-chunk / mid-wait).
                if current_task and not current_task.done():
                    current_task.cancel()
                await emit_status("idle")

    finally:
        clients.discard(websocket)


async def main():
    async with websockets.serve(handler, "127.0.0.1", PORT):
        print(f"NavAI agent server running on ws://127.0.0.1:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
