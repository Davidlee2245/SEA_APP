'use strict';

/**
 * SEA Electron Main Process
 *
 * Responsibilities:
 *  1. On first run → show settings window so the user can pick their Python exe
 *  2. On subsequent runs → spawn the Python backend as a child process,
 *     wait for "BACKEND_READY" on stdout, then open the main app window
 *  3. Pass the resolved backend port to the renderer via IPC so React
 *     can construct absolute API URLs (Phase 3)
 *  4. Kill the Python process cleanly on quit
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn }  = require('child_process');
const path       = require('path');
const fs         = require('fs');
const net        = require('net');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** JSON file that stores user configuration (Python path, data root, etc.) */
const CONFIG_FILE = path.join(app.getPath('userData'), 'sea-config.json');

/**
 * When running packaged, Python script lives under process.resourcesPath/app/.
 * When running in dev (electron . from the electron/ dir), it's one level up.
 */
const APP_DIR = app.isPackaged
  ? path.join(process.resourcesPath, 'app')
  : path.join(__dirname, '..');

/**
 * Where the user's data lives (input images, outputs, cache).
 * In dev mode we default to the project's own data/ folder so existing
 * sample data is immediately available. In production we use userData so
 * data survives app updates.
 */
const DEFAULT_DATA_ROOT = app.isPackaged
  ? path.join(app.getPath('userData'), 'SEA', 'data')
  : path.join(APP_DIR, 'data');

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    }
  } catch (e) {
    console.error('[Config] Failed to read config:', e.message);
  }
  return null;
}

function saveConfig(config) {
  try {
    fs.mkdirSync(path.dirname(CONFIG_FILE), { recursive: true });
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), 'utf8');
  } catch (e) {
    console.error('[Config] Failed to save config:', e.message);
  }
}

// ---------------------------------------------------------------------------
// Port finder — tries `start`, increments if busy
// ---------------------------------------------------------------------------

function findFreePort(start = 5000) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.listen(start, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
    server.on('error', () => {
      findFreePort(start + 1).then(resolve).catch(reject);
    });
  });
}

// ---------------------------------------------------------------------------
// Python backend process
// ---------------------------------------------------------------------------

let pythonProcess = null;

/**
 * Spawn the Python Flask backend and resolve when it prints "BACKEND_READY".
 *
 * @param {string} pythonExe  - Absolute path to python / python.exe
 * @param {string} dataRoot   - Root directory for data (passed as --data-root)
 * @param {number} port       - Port Flask will listen on
 * @returns {Promise<void>}
 */
function startPythonBackend(pythonExe, dataRoot, port) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(APP_DIR, 'api_server_extended.py');

    console.log('[Backend] Spawning:', pythonExe, scriptPath);
    console.log('[Backend] Port:', port, '| Data root:', dataRoot);

    // Build environment — inherit parent env then layer in API keys from config
    const childEnv = { ...process.env };
    const config   = loadConfig() || {};
    if (config.openaiKey)    childEnv['OPENAI_API_KEY']    = config.openaiKey;
    if (config.anthropicKey) childEnv['ANTHROPIC_API_KEY'] = config.anthropicKey;

    pythonProcess = spawn(pythonExe, [
      scriptPath,
      '--port', String(port),
      '--data-root', dataRoot,
    ], {
      cwd: APP_DIR,
      env: childEnv,
    });

    let ready = false;
    const stderrLines = [];   // collect for error reporting

    pythonProcess.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      process.stdout.write('[Python] ' + text);
      if (!ready && text.includes('BACKEND_READY')) {
        ready = true;
        resolve();
      }
    });

    pythonProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      process.stderr.write('[Python ERR] ' + text);
      // Keep the first 20 lines of stderr for the error dialog
      if (stderrLines.length < 20) {
        stderrLines.push(...text.split('\n').filter(l => l.trim()));
      }
    });

    function buildErrorMessage(prefix) {
      const excerpt = stderrLines.slice(0, 10).join('\n');
      return excerpt
        ? `${prefix}\n\nPython stderr:\n${excerpt}`
        : prefix;
    }

    pythonProcess.on('exit', (code, signal) => {
      console.log(`[Backend] Process exited — code: ${code}, signal: ${signal}`);
      if (!ready) {
        reject(new Error(buildErrorMessage(
          `Python backend exited (code ${code}) before signalling BACKEND_READY.\n` +
          `Executable: ${pythonExe}\n` +
          'Check that the Python executable is correct and all dependencies are installed.'
        )));
      }
    });

    pythonProcess.on('error', (err) => {
      if (!ready) {
        reject(new Error(buildErrorMessage(
          `Failed to start Python process: ${err.message}\n` +
          `Executable tried: ${pythonExe}`
        )));
      }
    });

    // Safety timeout — 90 s is generous for ML model loading
    setTimeout(() => {
      if (!ready) {
        reject(new Error(
          'Python backend did not become ready within 90 seconds.\n' +
          'It may still be loading ML models. Try again, or check the logs.'
        ));
      }
    }, 90_000);
  });
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

let mainWindow     = null;
let settingsWindow = null;
let splashWindow   = null;

/** True while we are between "settings closed" and "main window opened". */
let isLaunching = false;

// ---------------------------------------------------------------------------
// Splash / loading window shown while the Python backend starts up
// ---------------------------------------------------------------------------

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width:       400,
    height:      200,
    resizable:   false,
    frame:       false,       // borderless for a clean splash look
    transparent: false,
    show:        true,
    title:       'SEA — Starting…',
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });

  // Inline HTML — no extra file needed
  splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(`
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8"/>
      <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
          background:#1a1a2e; color:#e0e0e0;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          display:flex; flex-direction:column;
          align-items:center; justify-content:center;
          height:100vh; gap:18px;
        }
        h2 { color:#e94560; font-size:20px; letter-spacing:.5px; }
        p  { font-size:13px; color:#8899aa; }
        .bar-wrap { width:260px; height:4px; background:#1e2e3e; border-radius:2px; overflow:hidden; }
        .bar { height:100%; background:#e94560; border-radius:2px;
               animation: grow 2s ease-in-out infinite alternate; }
        @keyframes grow { from{width:10%} to{width:90%} }
      </style>
    </head>
    <body>
      <h2>🔬 SEA</h2>
      <div class="bar-wrap"><div class="bar"></div></div>
      <p>Starting Python backend — this may take up to 90 s on first run…</p>
    </body>
    </html>
  `)}`);

  splashWindow.on('closed', () => { splashWindow = null; });
}

function closeSplash() {
  if (splashWindow) {
    splashWindow.close();
    splashWindow = null;
  }
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width:    1440,
    height:   900,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    title: 'SEA — Exosome Analysis',
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
      // Allow loading local file:// images served by the Python backend
      webSecurity: true,
    },
  });

  const indexPath = app.isPackaged
    ? path.join(process.resourcesPath, 'app', 'frontend', 'dist', 'index.html')
    : path.join(__dirname, '..', 'frontend', 'dist', 'index.html');

  mainWindow.loadFile(indexPath);

  // Send backend port to renderer after page loads
  // (renderer reads window.electronAPI.onBackendPort)
  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.webContents.send('backend-port', port);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
    mainWindow.webContents.openDevTools();   // TODO: remove before release
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

function createSettingsWindow() {
  settingsWindow = new BrowserWindow({
    width:     580,
    height:    620,
    resizable: false,
    show:      true,
    title:     'SEA — First-Time Setup',
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
    },
  });

  settingsWindow.setMenuBarVisibility(false);
  settingsWindow.loadFile(path.join(__dirname, 'settings.html'));
  settingsWindow.on('closed', () => { settingsWindow = null; });
}

// ---------------------------------------------------------------------------
// App launch flow
// ---------------------------------------------------------------------------

async function launch() {
  const config = loadConfig();

  if (!config || !config.pythonExe) {
    // First run — ask user to configure Python
    createSettingsWindow();
    return;
  }

  await launchWithConfig(config);
}

async function launchWithConfig(config) {
  isLaunching = true;
  createSplashWindow();

  try {
    const port     = await findFreePort(5000);
    const dataRoot = config.dataRoot || DEFAULT_DATA_ROOT;

    await startPythonBackend(config.pythonExe, dataRoot, port);

    closeSplash();
    isLaunching = false;
    createMainWindow(port);
  } catch (err) {
    closeSplash();
    isLaunching = false;

    const choice = dialog.showMessageBoxSync({
      type:    'error',
      title:   'Failed to Start Backend',
      message: 'The Python backend could not be started.',
      detail:  err.message + '\n\nWould you like to open Settings to fix the configuration?',
      buttons: ['Open Settings', 'Quit'],
    });
    if (choice === 0) {
      createSettingsWindow();
    } else {
      app.quit();
    }
  }
}

// ---------------------------------------------------------------------------
// IPC handlers (called from renderer via contextBridge)
// ---------------------------------------------------------------------------

/** Open a file picker and return the selected Python executable path. */
ipcMain.handle('select-python', async () => {
  const result = await dialog.showOpenDialog(settingsWindow, {
    title:   'Select Python Executable',
    message: 'Choose the python or python.exe inside your SEA conda environment',
    filters: [
      { name: 'Python Executable', extensions: ['exe', ''] },
      { name: 'All Files',         extensions: ['*'] },
    ],
    properties: ['openFile'],
  });
  return result.canceled ? null : result.filePaths[0];
});

/** Open a folder picker for the data root directory. */
ipcMain.handle('select-data-root', async () => {
  const result = await dialog.showOpenDialog(settingsWindow, {
    title:      'Select Data Root Folder',
    message:    'Choose the folder where SEA will store input images and results',
    properties: ['openDirectory', 'createDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

/** Save config to disk and launch the app. Closes settings window. */
ipcMain.handle('launch-after-settings', async (_event, config) => {
  saveConfig(config);

  if (settingsWindow) {
    settingsWindow.close();
  }

  await launchWithConfig(config);
});

/** Return currently saved config (so settings window can pre-fill fields). */
ipcMain.handle('get-config', async () => {
  return loadConfig() || {};
});

/** Return the default data root so settings window can show it as placeholder. */
ipcMain.handle('get-default-data-root', async () => {
  return DEFAULT_DATA_ROOT;
});

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(launch);

app.on('window-all-closed', () => {
  // Don't quit while we are in the middle of starting the backend —
  // the splash window may have just closed and the main window isn't open yet.
  if (isLaunching) return;
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  // macOS: re-open window on dock icon click
  if (mainWindow === null && settingsWindow === null) launch();
});

function killPython() {
  if (pythonProcess) {
    console.log('[Main] Killing Python backend...');
    try {
      // On Windows, SIGTERM is not supported — taskkill is more reliable
      if (process.platform === 'win32') {
        require('child_process').spawnSync('taskkill', ['/PID', String(pythonProcess.pid), '/F', '/T']);
      } else {
        pythonProcess.kill('SIGTERM');
      }
    } catch (e) {
      console.error('[Main] Failed to kill Python process:', e.message);
    }
    pythonProcess = null;
  }
}

// Fires on clean quit (Cmd+Q, window close, app.quit())
app.on('will-quit', killPython);

// Belt-and-suspenders: also fires on process.exit() and uncaught crashes
// so Python doesn't become an orphan if Electron hard-crashes.
process.on('exit',           killPython);
process.on('SIGINT',         () => { killPython(); process.exit(0); });
process.on('SIGTERM',        () => { killPython(); process.exit(0); });
process.on('uncaughtException', (err) => {
  console.error('[Main] Uncaught exception:', err);
  killPython();
  process.exit(1);
});
