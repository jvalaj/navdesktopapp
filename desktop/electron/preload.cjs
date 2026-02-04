const { contextBridge, ipcRenderer } = require("electron");
const path = require("node:path");

function toFileUrl(filePath) {
  if (!filePath) return "";
  try {
    const url = require("node:url");
    const fn = url.pathToFileURL;
    if (typeof fn === "function") return fn.call(url, filePath).toString();
  } catch (_) {}
  const normalized = path.resolve(filePath).replace(/\\/g, "/");
  const withLeading = normalized.startsWith("/") ? normalized : "/" + normalized;
  return "file://" + encodeURI(withLeading);
}

contextBridge.exposeInMainWorld("navai", {
  settingsGet: () => ipcRenderer.invoke("settings:get"),
  settingsSet: (updates) => ipcRenderer.invoke("settings:set", updates),
  apiKeyStatus: () => ipcRenderer.invoke("apikey:get-status"),
  apiKeySet: (key) => ipcRenderer.invoke("apikey:set", key),
  apiKeyClear: () => ipcRenderer.invoke("apikey:clear"),
  pythonStatus: () => ipcRenderer.invoke("py:status"),
  pythonRestart: () => ipcRenderer.invoke("py:restart"),
  selectPython: () => ipcRenderer.invoke("dialog:select-python"),
  setAgentWindowMode: (enabled) => ipcRenderer.invoke("window:agent-mode", enabled),
  setCaptureMode: (enabled) => ipcRenderer.invoke("window:capture-mode", enabled),
  openFile: (filePath) => ipcRenderer.invoke("file:open", filePath),
  fileUrl: toFileUrl
});
