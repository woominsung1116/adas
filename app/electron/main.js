const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

let mainWindow;
let pythonProcess;

const isDev = !app.isPackaged;

/**
 * Resolve the Python executable.
 *
 * In development: look for a venv at the project root (../../.venv).
 * In packaged mode: fall back to a system `python3` / `python`.
 *
 * NOTE: A fully self-contained packaged Python runtime is not practical
 * without tools like PyInstaller or conda-pack. The packaged app assumes
 * Python 3.12+ is available on the system PATH with required packages
 * installed. This limitation is documented here intentionally.
 */
function resolvePython() {
  if (isDev) {
    // 1. Check repo-local venv (adas/.venv)
    const repoRoot = path.join(__dirname, "..", "..");
    const localVenv = path.join(repoRoot, ".venv", "bin", "python");
    if (fs.existsSync(localVenv)) return localVenv;

    // 2. Check parent project venv (캡스톤/.venv)
    const parentVenv = path.join(repoRoot, "..", ".venv", "bin", "python");
    if (fs.existsSync(parentVenv)) return parentVenv;
  }
  // Fallback to system python
  try {
    require("child_process").execSync("python3 --version", { stdio: "ignore" });
    return "python3";
  } catch { return "python"; }
}

function startPythonBackend() {
  const projectRoot = isDev
    ? path.join(__dirname, "..", "..")
    : path.join(process.resourcesPath, "python");

  // In dev mode, server.py lives alongside the app directory.
  // In packaged mode, backend/ must be included via extraResources or files.
  const serverPath = isDev
    ? path.join(__dirname, "..", "backend", "server.py")
    : path.join(process.resourcesPath, "backend", "server.py");

  if (!fs.existsSync(serverPath)) {
    console.error(
      `[Python] server.py not found at ${serverPath}. ` +
      `Ensure backend files are included in the build configuration.`
    );
    return;
  }

  const pythonCmd = resolvePython();
  console.log(`[Python] Using: ${pythonCmd}`);
  console.log(`[Python] Server: ${serverPath}`);
  console.log(`[Python] CWD: ${projectRoot}`);

  pythonProcess = spawn(pythonCmd, [serverPath], {
    cwd: projectRoot,
    env: { ...process.env, PYTHONPATH: projectRoot },
  });

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
    if (mainWindow && data.toString().includes("Uvicorn running")) {
      mainWindow.webContents.send("backend-ready");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    console.error(`[Python] ${msg}`);
    // Uvicorn logs startup to stderr
    if (mainWindow && msg.includes("Uvicorn running")) {
      mainWindow.webContents.send("backend-ready");
    }
  });

  pythonProcess.on("close", (code) => {
    console.log(`[Python] exited with code ${code}`);
  });

  pythonProcess.on("error", (err) => {
    console.error(`[Python] Failed to start: ${err.message}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: "ADAS - ADHD Classroom Simulation",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "renderer", "dist", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  startPythonBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// IPC: Claude Code CLI call
ipcMain.handle("run-claude", async (event, prompt) => {
  return new Promise((resolve, reject) => {
    const claude = spawn("claude", ["-p", prompt], { timeout: 180000 });
    let output = "";
    let error = "";

    claude.stdout.on("data", (data) => { output += data.toString(); });
    claude.stderr.on("data", (data) => { error += data.toString(); });
    claude.on("close", (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(error || `claude exited with code ${code}`));
    });
  });
});
