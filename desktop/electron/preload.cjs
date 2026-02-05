const { contextBridge, ipcRenderer } = require("electron");

function toFileUrl(filePath) {
  if (!filePath) return "";
  const normalized = String(filePath).replace(/\\/g, "/");
  return "navai://" + encodeURI(normalized);
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
  storagePaths: () => ipcRenderer.invoke("storage:paths"),
  openStorageDir: (which) => ipcRenderer.invoke("storage:open", which),
  fileUrl: toFileUrl
});
