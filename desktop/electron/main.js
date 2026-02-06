import { app, BrowserWindow, ipcMain, dialog, shell, screen, protocol, net } from "electron";
import path from "node:path";
import { spawn } from "node:child_process";
import fs from "node:fs";
import Store from "electron-store";
import { pathToFileURL, fileURLToPath } from "node:url";

const isDev = !app.isPackaged;
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const preloadPath = path.resolve(__dirname, "preload.cjs");
const store = new Store({ name: "nav-ai" });
const APIKEY_KEY = "anthropicApiKey";

protocol.registerSchemesAsPrivileged([
  {
    scheme: "navai",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true
    }
  }
]);

function getStorageDirs() {
  const userData = app.getPath("userData");
  const conversationsDir = path.join(userData, "storage", "conversations");
  const screenshotsDir = path.join(userData, "storage", "screenshots");
  return { conversationsDir, screenshotsDir };
}

function ensureStorageDirs() {
  const { conversationsDir, screenshotsDir } = getStorageDirs();
  fs.mkdirSync(conversationsDir, { recursive: true });
  fs.mkdirSync(screenshotsDir, { recursive: true });
}

let mainWindow;
let pythonProc;
let serverPort = 8765;
const iconPath = path.resolve(app.getAppPath(), "navlogo.png");
let restoreBounds = null;
let restoreOpacity = 1;
let animTimer = null;

function resolvePythonPath() {
  const stored = store.get("pythonPath");
  if (stored) return stored;
  return "python3";
}

function resolveServerScript() {
  return path.resolve(app.getAppPath(), "..", "agent_server.py");
}

async function startPythonServer() {
  if (pythonProc) return;
  const pythonPath = resolvePythonPath();
  const serverScript = resolveServerScript();
  const apiKey = store.get(APIKEY_KEY);
  const { conversationsDir, screenshotsDir } = getStorageDirs();
  const env = {
    ...process.env,
    NAVAI_SERVER_PORT: String(serverPort),
    ANTHROPIC_API_KEY: apiKey || "",
    NAVAI_CONVERSATIONS_DIR: conversationsDir,
    NAVAI_SCREENSHOTS_DIR: screenshotsDir
  };

  try {
    pythonProc = spawn(pythonPath, [serverScript], {
      env,
      stdio: "pipe"
    });
  } catch (err) {
    console.error("Failed to spawn Python:", err);
    pythonProc = null;
    return;
  }

  pythonProc.on("error", (err) => {
    console.error("Failed to start Python server:", err);
  });

  pythonProc.stdout.on("data", (data) => {
    if (isDev) console.log(`[py] ${data.toString().trim()}`);
  });

  pythonProc.stderr.on("data", (data) => {
    console.error(`[py] ${data.toString().trim()}`);
  });

  pythonProc.on("exit", () => {
    pythonProc = null;
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1220,
    height: 800,
    backgroundColor: "#F7FFF6",
    titleBarStyle: "hiddenInset",
    icon: iconPath,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    const indexPath = path.resolve(app.getAppPath(), "dist", "index.html");
    mainWindow.loadURL(pathToFileURL(indexPath).toString());
  }
}

function setAgentWindowMode(enabled) {
  if (!mainWindow) return;
  const animateTo = (target) => {
    if (!mainWindow) return;
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
    }
    const start = mainWindow.getBounds();
    const startTime = Date.now();
    const duration = 500; // Slower, more elegant
    // easeOutExpo: fast start, very smooth slow finish
    const easeOutExpo = (x) => (x === 1 ? 1 : 1 - Math.pow(2, -10 * x));

    animTimer = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const t = Math.min(1, elapsed / duration);
      const k = easeOutExpo(t);
      const next = {
        x: Math.round(start.x + (target.x - start.x) * k),
        y: Math.round(start.y + (target.y - start.y) * k),
        width: Math.round(start.width + (target.width - start.width) * k),
        height: Math.round(start.height + (target.height - start.height) * k)
      };
      mainWindow.setBounds(next, false);
      if (t >= 1) {
        clearInterval(animTimer);
        animTimer = null;
      }
    }, 1000 / 60); // Target 60fps
  };
  if (enabled) {
    restoreBounds = mainWindow.getBounds();
    restoreOpacity = mainWindow.getOpacity();
    const display = screen.getPrimaryDisplay();
    const workArea = display.workArea;
    const width = 360;
    const height = 620;
    const x = Math.max(workArea.x, workArea.x + 16);
    const y = Math.max(workArea.y, workArea.y + 16);
    animateTo({ x, y, width, height });
    mainWindow.setAlwaysOnTop(true, "floating");
    mainWindow.setVisibleOnAllWorkspaces(true);
    mainWindow.setContentProtection(true);
  } else {
    if (restoreBounds) {
      animateTo(restoreBounds);
    }
    mainWindow.setAlwaysOnTop(false);
    mainWindow.setVisibleOnAllWorkspaces(false);
    mainWindow.setContentProtection(false);
    mainWindow.setOpacity(restoreOpacity || 1);
    mainWindow.setIgnoreMouseEvents(false);
  }
}

function setCaptureMode(enabled) {
  if (!mainWindow) return;
  if (enabled) {
    restoreOpacity = mainWindow.getOpacity();
    mainWindow.setOpacity(0);
    mainWindow.setIgnoreMouseEvents(true);
  } else {
    mainWindow.setOpacity(restoreOpacity || 1);
    mainWindow.setIgnoreMouseEvents(false);
  }
}

app.whenReady().then(async () => {
  protocol.handle("navai", (request) => {
    const url = new URL(request.url);
    const decodedPath = decodeURIComponent(url.pathname || "");
    return net.fetch(pathToFileURL(decodedPath).toString());
  });

  ensureStorageDirs();
  createWindow();
  if (app.dock && iconPath) {
    try {
      await app.dock.setIcon(iconPath);
    } catch {
      // Ignore icon errors in dev.
    }
  }
  await startPythonServer();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (pythonProc) {
    pythonProc.kill();
  }
});

ipcMain.handle("settings:get", async () => {
  const model = store.get("model", "claude-sonnet-4-5-20250929");
  const screenshotFrequency = store.get("screenshotFrequency", 2);
  const verificationEveryNSteps = store.get("verificationEveryNSteps", screenshotFrequency);
  const saveScreenshots = store.get("saveScreenshots", true);
  const allowSendScreenshots = store.get("allowSendScreenshots", true);
  const dryRun = store.get("dryRun", false);
  const llmTimeoutSeconds = store.get("llmTimeoutSeconds", 60);
  const maxStagnantSteps = store.get("maxStagnantSteps", 4);
  const maxStuckSignals = store.get("maxStuckSignals", 2);
  return {
    model,
    screenshotFrequency,
    verificationEveryNSteps,
    saveScreenshots,
    allowSendScreenshots,
    dryRun,
    llmTimeoutSeconds,
    maxStagnantSteps,
    maxStuckSignals
  };
});

ipcMain.handle("settings:set", async (_event, updates) => {
  Object.entries(updates || {}).forEach(([key, value]) => {
    store.set(key, value);
  });
  return { ok: true };
});

ipcMain.handle("apikey:get-status", async () => {
  const key = store.get(APIKEY_KEY);
  return { connected: Boolean(key) };
});

ipcMain.handle("apikey:set", async (_event, apiKey) => {
  try {
    if (!apiKey) return { ok: false, error: "Missing key" };
    store.set(APIKEY_KEY, apiKey);
    return { ok: true };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
});

ipcMain.handle("apikey:clear", async () => {
  store.delete(APIKEY_KEY);
  return { ok: true };
});

ipcMain.handle("py:status", async () => {
  return { running: Boolean(pythonProc), port: serverPort };
});

ipcMain.handle("py:restart", async () => {
  if (pythonProc) pythonProc.kill();
  pythonProc = null;
  await startPythonServer();
  return { ok: true };
});

ipcMain.handle("file:open", async (_event, filePath) => {
  if (!filePath) return { ok: false };
  await shell.openPath(filePath);
  return { ok: true };
});

ipcMain.handle("dialog:select-python", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "Python", extensions: ["py", "exe"] }]
  });
  if (result.canceled || !result.filePaths[0]) return { ok: false };
  store.set("pythonPath", result.filePaths[0]);
  return { ok: true, path: result.filePaths[0] };
});

ipcMain.handle("window:agent-mode", async (_event, enabled) => {
  setAgentWindowMode(Boolean(enabled));
  return { ok: true };
});

ipcMain.handle("window:capture-mode", async (_event, enabled) => {
  setCaptureMode(Boolean(enabled));
  return { ok: true };
});

ipcMain.handle("storage:paths", async () => {
  return getStorageDirs();
});

ipcMain.handle("storage:open", async (_event, which) => {
  const dirs = getStorageDirs();
  const dir = which === "screenshots" ? dirs.screenshotsDir : dirs.conversationsDir;
  if (dir) await shell.openPath(dir);
  return { ok: true };
});
