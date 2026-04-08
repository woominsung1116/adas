const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("adas", {
  // Python backend WebSocket URL
  backendUrl: "ws://localhost:8000/ws",

  // Listen for backend-ready event
  onBackendReady: (callback) => ipcRenderer.on("backend-ready", callback),

  // Direct Claude Code CLI call from renderer
  runClaude: (prompt) => ipcRenderer.invoke("run-claude", prompt),
});
