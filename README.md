# Nav (v1)

## Project structure

The app uses **both** the `desktop/` folder and the **v1 root** (this folder):

| Location | Role |
|----------|------|
| **desktop/** | Electron app: main process, renderer (React UI), preload. This is what you run with `npm run dev` or the packaged app. |
| **v1 root** | Python backend: the Electron app spawns `agent_server.py` from here. The server uses `agent.py`, `vision.py`, and `typeandclick.py` (all in v1 root). |

When you launch the app, Electron starts the Python server via `path.resolve(app.getAppPath(), "..", "agent_server.py")`, so the agent and vision code in the v1 folder are required for the app to work.
