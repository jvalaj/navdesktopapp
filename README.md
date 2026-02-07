# Nav (v1)

## Project structure

The app uses **both** the `desktop/` folder and the **v1 root** (this folder):

| Location | Role |
|----------|------|
| **desktop/** | Electron app: main process, renderer (React UI), preload. This is what you run with `npm run dev` or the packaged app. |
| **v1 root** | Python backend: the Electron app spawns `agent_server.py` from here. The server uses `agent.py`, `vision.py`, and `typeandclick.py` (all in v1 root). |

When you launch the app, Electron starts the Python server via `path.resolve(app.getAppPath(), "..", "agent_server.py")`, so the agent and vision code in the v1 folder are required for the app to work.

## Vision precision on macOS (recommended)

`vision.py` now supports a **VLM-first** backend for UI element detection from screenshots. It uses a vision model to semantically detect UI controls, then applies deterministic anti-hallucination filters before returning clickable boxes.

The parse order in `auto` mode is:
1. VLM detection (`source=vlm`)
2. CV fallback/supplement (`source=cv`) when VLM is weak or supplement is enabled

### Controls

Environment variables:
- `VISION_BACKEND=auto|vlm|cv` (default `auto`)
- `VISION_USE_VLM=1` enables model-based screenshot detection (default `1`)
- `VISION_VLM_MODEL=claude-sonnet-4-5-20250929` model used for VLM detection
- `VISION_VLM_TIMEOUT=10.0` timeout (seconds) for VLM API call
- `VISION_STRICT=1` to prefer fewer, more-clickable elements (default `1`)
- `VISION_MAX_BOXES=220` cap output count (default `220`)
- `VISION_MIN_SCORE=0.70` CV threshold (default `0.70`)
- `VISION_CV_SUPPLEMENT=1` to add CV boxes even when VLM is available (default `0`)

CLI example:
- `python3 vision.py --image /path/to/screenshot.png --backend auto --strict`
- `python3 vision.py --image /path/to/screenshot.png --backend vlm --strict --vlm_model claude-sonnet-4-5-20250929`

Run tests:
- `python3 -m unittest discover -s tests -p "test*.py" -v`

## Clean architecture (simple)

Runtime flow:

`[Screen] -> [Screenshot] -> [User-selected model] -> [Structured action JSON] -> [Nav action engine] -> [Mouse/Keyboard/OS]`

- `agent.py` handles:
  - screenshot capture
  - model decision calls
  - action JSON parsing/normalization
  - dispatch to `typeandclick.py`
- `typeandclick.py` is model-agnostic execution (click/type/scroll/keys).
- `agent_server.py` passes model settings from UI/API to `Agent`.
- `vision.py` remains optional fallback/context and also supports provider selection for VLM detection.

### Model selection (decision model)

`agent_server.py` / websocket `settings` can now include:

- `model`: model name
- `modelProvider`: `anthropic` or `openai_compatible`
- `modelBaseUrl`: base URL for local OpenAI-compatible servers (examples: Ollama/vLLM/LM Studio)
- `modelApiKey`: optional API key for provider
- `llmTimeoutSeconds`: request timeout

Default behavior is unchanged: Anthropic model with existing env key.

### Local model examples

- Ollama compatible endpoint: `http://127.0.0.1:11434/v1`
- LM Studio compatible endpoint: `http://127.0.0.1:1234/v1`
- vLLM OpenAI endpoint: `http://127.0.0.1:8000/v1`

Use `modelProvider=openai_compatible` with your chosen `modelBaseUrl`.
