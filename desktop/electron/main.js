import { app, BrowserWindow, ipcMain, dialog, shell, screen, protocol, net } from "electron";
import path from "node:path";
import fs from "node:fs";
import { pathToFileURL, fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import Store from "electron-store";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

protocol.registerSchemesAsPrivileged([
  { scheme: "navai", privileges: { standard: true, supportFetchAPI: true } }
]);

const isDev = !app.isPackaged;
const store = new Store({ name: "nav-ai" });
const APIKEY_KEY = "anthropicApiKey";

let mainWindow;
let pythonProc;
let serverPort = 8765;
const iconPath = path.resolve(app.getAppPath(), "navlogo.png");
let restoreBounds = null;
let restoreOpacity = 1;       // used by setAgentWindowMode
let captureRestoreOpacity = 1; // used by setCaptureMode (separate to avoid cross-talk)
let animTimer = null;

function getStorageDirs() {
  // Per-user, per-app directory (dynamic for whoever is running the app)
  const base = path.join(app.getPath("userData"), "storage");
  const conversationsDir = path.join(base, "conversations");
  const screenshotsDir = path.join(base, "screenshots");
  return { base, conversationsDir, screenshotsDir };
}

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
  try {
    fs.mkdirSync(conversationsDir, { recursive: true });
    fs.mkdirSync(screenshotsDir, { recursive: true });
  } catch (err) {
    console.error("Failed to create storage directories:", err);
  }
  const env = {
    ...process.env,
    NAVAI_SERVER_PORT: String(serverPort),
    ANTHROPIC_API_KEY: apiKey || "",
    NAVAI_CONVERSATIONS_DIR: conversationsDir,
    NAVAI_SCREENSHOTS_DIR: screenshotsDir,
    // Prevent Python from showing in dock
    PYTHONUNBUFFERED: "1"
  };

  try {
    pythonProc = spawn(pythonPath, [serverScript], {
      env,
      stdio: "pipe",
      detached: false
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
  // Resolve preload path - use absolute path in dev, app.getAppPath() in production
  let preloadPath;
  if (isDev) {
    // In dev, use the directory of this file (main.js) to find preload.cjs
    // __dirname is reliable in ES modules when used this way
    preloadPath = path.resolve(__dirname, "preload.cjs");
  } else {
    preloadPath = path.resolve(app.getAppPath(), "electron", "preload.cjs");
  }

  // Verify preload file exists
  if (!fs.existsSync(preloadPath)) {
    console.error(`Preload script not found at: ${preloadPath}`);
    console.error(`App path: ${app.getAppPath()}`);
    console.error(`__dirname: ${__dirname}`);
  } else {
    console.log(`Preload script found at: ${preloadPath}`);
  }

  mainWindow = new BrowserWindow({
    width: 1220,
    height: 800,
    backgroundColor: "#F7FFF6",
    titleBarStyle: "hiddenInset",
    icon: iconPath,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
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
    captureRestoreOpacity = mainWindow.getOpacity();
    mainWindow.setOpacity(0);
    mainWindow.setIgnoreMouseEvents(true);
  } else {
    mainWindow.setOpacity(captureRestoreOpacity || 1);
    mainWindow.setIgnoreMouseEvents(false);
  }
}

function resolveScreenshotPath(filePath) {
  if (!filePath || typeof filePath !== "string") return null;
  const appPath = app.getAppPath();
  const parentDir = path.resolve(appPath, "..");
  const { screenshotsDir } = getStorageDirs();
  // Allow screenshots from per-user storage (and keep legacy roots for dev/back-compat)
  const roots = [screenshotsDir, appPath, parentDir];
  const normalized = path.normalize(filePath).replace(/^(\.\.(\/|\\))+/g, "").replace(/^\/+/, "");
  let full = null;
  if (path.isAbsolute(filePath)) {
    full = path.normalize(filePath);
  } else {
    for (const r of roots) {
      const candidate = path.resolve(r, normalized);
      if (fs.existsSync(candidate)) {
        full = candidate;
        break;
      }
    }
    if (!full) full = path.resolve(parentDir, normalized);
  }
  try {
    const inRoot = roots.some((r) => {
      const rel = path.relative(r, full);
      return !rel.startsWith("..");
    });
    if (!inRoot || !fs.existsSync(full)) return null;
    return full;
  } catch (_) {
    return null;
  }
}

app.whenReady().then(async () => {
  protocol.handle("navai", (request) => {
    const urlPart = request.url.slice("navai://".length).replace(/^\/+/, "");
    const decoded = decodeURIComponent(urlPart);
    const full = resolveScreenshotPath(decoded);
    if (!full) return new Response("Not found", { status: 404 });
    return net.fetch(pathToFileURL(full).toString());
  });

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
  const saveScreenshots = store.get("saveScreenshots", true);
  const allowSendScreenshots = store.get("allowSendScreenshots", true);
  const dryRun = store.get("dryRun", false);
  return { model, screenshotFrequency, saveScreenshots, allowSendScreenshots, dryRun };
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

ipcMain.handle("storage:paths", async () => {
  const { conversationsDir, screenshotsDir } = getStorageDirs();
  return { conversationsDir, screenshotsDir };
});

ipcMain.handle("storage:open", async (_event, which) => {
  const { conversationsDir, screenshotsDir } = getStorageDirs();
  const target =
    which === "conversations" ? conversationsDir :
    which === "screenshots" ? screenshotsDir :
    null;
  if (!target) return { ok: false };
  await shell.openPath(target);
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
