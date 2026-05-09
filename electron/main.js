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

const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const { spawn, spawnSync } = require('child_process');
const os           = require('os');
const path       = require('path');
const fs         = require('fs');
const net        = require('net');
const { pathToFileURL } = require('url');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** JSON file that stores user configuration (Python path, data root, etc.) */
const CONFIG_FILE = path.join(app.getPath('userData'), 'sea-config.json');
const STORAGE_FILE = path.join(app.getPath('userData'), 'sea-storage.json');

/**
 * When running packaged, Python script lives under process.resourcesPath/app/.
 * When running in dev (electron . from the electron/ dir), it's one level up.
 */
const APP_DIR = app.isPackaged
  ? path.join(process.resourcesPath, 'app')
  : path.join(__dirname, '..');
const APP_ICON_PATH = path.join(__dirname, 'exosome.png');

/**
 * Default data root when Settings has no dataRoot (input images, outputs, cache).
 * Uses the OS Documents folder so the path is writable and portable across platforms.
 */
const DEFAULT_DATA_ROOT = path.join(app.getPath('documents'), 'SEA', 'data');

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
// Automatic conda + pip setup (first launch)
// ---------------------------------------------------------------------------

const CONDA_SPAWN_MAX_BUFFER = 64 * 1024 * 1024;

/** Send a setup status line to the settings window progress log. */
function sendAutoSetupProgress(webContents, message) {
  if (webContents && !webContents.isDestroyed()) {
    webContents.send('auto-setup-progress', message);
  }
}

/** Common conda install roots (no PATH search). */
function getCondaInstallRoots() {
  const home = os.homedir();
  const roots = [];

  if (process.platform === 'win32') {
    const profile = process.env.USERPROFILE || home;
    const localApp = process.env.LOCALAPPDATA || '';
    roots.push(path.join(profile, 'anaconda3'));
    roots.push(path.join(profile, 'miniconda3'));
    if (localApp) {
      roots.push(path.join(localApp, 'anaconda3'));
    }
  } else {
    roots.push(path.join(home, 'anaconda3'));
    roots.push(path.join(home, 'miniconda3'));
    roots.push('/opt/anaconda3');
    roots.push('/opt/miniconda3');
    roots.push(path.join(home, 'opt', 'anaconda3'));
    roots.push(path.join(home, 'opt', 'miniconda3'));
  }

  return roots;
}

/** Find conda executable in known install locations. */
function findCondaExecutable() {
  for (const root of getCondaInstallRoots()) {
    if (!root || !fs.existsSync(root)) continue;
    if (process.platform === 'win32') {
      const condaExe = path.join(root, 'Scripts', 'conda.exe');
      if (fs.existsSync(condaExe)) return condaExe;
      const condaBat = path.join(root, 'condabin', 'conda.bat');
      if (fs.existsSync(condaBat)) return condaBat;
    } else {
      const condaBin = path.join(root, 'bin', 'conda');
      if (fs.existsSync(condaBin)) return condaBin;
    }
  }
  return null;
}

function runConda(condaExe, args) {
  const needsShell =
    process.platform === 'win32' && condaExe.toLowerCase().endsWith('.bat');
  const opts = {
    encoding: 'utf8',
    maxBuffer: CONDA_SPAWN_MAX_BUFFER,
    env: process.env,
    shell: needsShell,
  };
  return spawnSync(condaExe, args, opts);
}

function getCondaBase(condaExe) {
  const r = runConda(condaExe, ['info', '--base']);
  if (r.error || r.status !== 0) return null;
  const lines = String(r.stdout || '')
    .trim()
    .split(/\r?\n/)
    .filter((l) => l.trim());
  const base = lines.length ? lines[lines.length - 1].trim() : '';
  return base || null;
}

/** Candidate interpreter paths for conda env SEA (conda uses env root python.exe on Windows; Scripts is fallback). */
function seaPythonCandidatesFromBase(condaBase) {
  if (process.platform === 'win32') {
    return [
      path.join(condaBase, 'envs', 'SEA', 'python.exe'),
      path.join(condaBase, 'envs', 'SEA', 'Scripts', 'python.exe'),
    ];
  }
  return [
    path.join(condaBase, 'envs', 'SEA', 'bin', 'python'),
    path.join(condaBase, 'envs', 'SEA', 'bin', 'python3'),
  ];
}

/** First preferred path whether or not env exists yet (used after create before files appear). */
function primarySeaPythonPathFromBase(condaBase) {
  return seaPythonCandidatesFromBase(condaBase)[0];
}

function seaEnvironmentExistsAtBase(condaBase) {
  return seaPythonCandidatesFromBase(condaBase).some((p) => fs.existsSync(p));
}

/** Resolve installed SEA interpreter, or primary expected path if none found. */
function resolveSeaPythonExe(condaBase) {
  for (const candidate of seaPythonCandidatesFromBase(condaBase)) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return primarySeaPythonPathFromBase(condaBase);
}

/**
 * Create SEA conda env + pip install requirements bundled with the app.
 * @param {Electron.WebContents} webContents
 * @returns {Promise<{ success: true, pythonExe: string } | { success: false, reason: 'conda-not-found' | 'error', message?: string }>}
 */
async function autoSetupConda(webContents) {
  sendAutoSetupProgress(webContents, 'Looking for Anaconda...');

  const condaExe = findCondaExecutable();
  if (!condaExe) {
    return { success: false, reason: 'conda-not-found' };
  }

  const condaBase = getCondaBase(condaExe);
  if (!condaBase) {
    return {
      success: false,
      reason: 'error',
      message: 'Found conda but could not read conda base path (conda info --base failed).',
    };
  }

  let seaPython = resolveSeaPythonExe(condaBase);

  if (!seaEnvironmentExistsAtBase(condaBase)) {
    sendAutoSetupProgress(webContents, 'Creating SEA Python environment (this may take a few minutes)...');
    const createRes = runConda(condaExe, ['create', '-n', 'SEA', 'python=3.10.19', '-y']);
    const createErr = (
      createRes.stderr ||
      createRes.stdout ||
      (createRes.error && createRes.error.message) ||
      ''
    ).trim();
    if (createRes.status !== 0 || createRes.error) {
      return {
        success: false,
        reason: 'error',
        message: createErr.slice(0, 8000) || 'conda create failed.',
      };
    }
    seaPython = resolveSeaPythonExe(condaBase);
    if (!fs.existsSync(seaPython)) {
      return {
        success: false,
        reason: 'error',
        message: 'SEA environment was created but the Python executable was not found at the expected path.',
      };
    }
  }

  const reqPath = path.join(APP_DIR, 'requirements.txt');
  if (!fs.existsSync(reqPath)) {
    return {
      success: false,
      reason: 'error',
      message: `requirements.txt not found at ${reqPath}`,
    };
  }

  sendAutoSetupProgress(webContents, 'Installing dependencies...');
  const pipRes = spawnSync(seaPython, ['-m', 'pip', 'install', '-r', reqPath], {
    encoding: 'utf8',
    maxBuffer: CONDA_SPAWN_MAX_BUFFER,
    env: process.env,
    shell: false,
  });
  const pipErr = (
    pipRes.stderr ||
    pipRes.stdout ||
    (pipRes.error && pipRes.error.message) ||
    ''
  ).trim();
  if (pipRes.status !== 0 || pipRes.error) {
    return {
      success: false,
      reason: 'error',
      message: pipErr.slice(0, 8000) || 'pip install failed.',
    };
  }

  sendAutoSetupProgress(webContents, 'Setup complete! Launching SEA...');
  return { success: true, pythonExe: seaPython };
}

// ---------------------------------------------------------------------------
// Port finder — tries `start`, increments if busy
// ---------------------------------------------------------------------------

function findFreePort(start = 8765) {
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
let agentProcess = null;

function loadStore() {
  try {
    if (!fs.existsSync(STORAGE_FILE)) return {};
    const raw = fs.readFileSync(STORAGE_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    console.error('[Store] Failed to read store file:', e.message);
    return {};
  }
}

function saveStore(store) {
  try {
    fs.mkdirSync(path.dirname(STORAGE_FILE), { recursive: true });
    fs.writeFileSync(STORAGE_FILE, JSON.stringify(store, null, 2), 'utf8');
    return true;
  } catch (e) {
    console.error('[Store] Failed to write store file:', e.message);
    return false;
  }
}

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

function startAgentBackend(pythonExe, port) {
  const scriptPath = path.join(APP_DIR, 'api_agent_chat.py');
  const childEnv = { ...process.env };
  const config = loadConfig() || {};
  if (config.openaiKey) childEnv['OPENAI_API_KEY'] = config.openaiKey;
  if (config.anthropicKey) childEnv['ANTHROPIC_API_KEY'] = config.anthropicKey;

  console.log('[Agent] Spawning:', pythonExe, scriptPath);
  console.log('[Agent] Port:', port);

  agentProcess = spawn(pythonExe, [scriptPath, '--port', String(port)], {
    cwd: APP_DIR,
    env: childEnv,
  });

  agentProcess.stdout.on('data', (chunk) => {
    process.stdout.write('[Agent] ' + chunk.toString());
  });

  agentProcess.stderr.on('data', (chunk) => {
    process.stderr.write('[Agent ERR] ' + chunk.toString());
  });

  agentProcess.on('exit', (code, signal) => {
    console.log(`[Agent] Process exited — code: ${code}, signal: ${signal}`);
    agentProcess = null;
  });

  agentProcess.on('error', (err) => {
    console.error('[Agent] Failed to start agent backend:', err.message);
    agentProcess = null;
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
    icon:        APP_ICON_PATH,
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
    icon: APP_ICON_PATH,
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
      additionalArguments: [`--sea-backend-port=${port}`],
      // Allow loading local file:// images served by the Python backend
      webSecurity: true,
    },
  });

  const indexPath = app.isPackaged
    ? path.join(process.resourcesPath, 'app', 'frontend', 'dist', 'index.html')
    : path.join(__dirname, '..', 'frontend', 'dist', 'index.html');

  mainWindow.loadFile(indexPath);

  const allowedFilePrefix = pathToFileURL(path.dirname(indexPath) + path.sep).toString();

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url && !url.startsWith(allowedFilePrefix)) {
      shell.openExternal(url).catch((err) => {
        console.error('[Main] Failed to open external URL:', err.message);
      });
    }
    return { action: 'deny' };
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (url && !url.startsWith(allowedFilePrefix)) {
      event.preventDefault();
      shell.openExternal(url).catch((err) => {
        console.error('[Main] Failed to open external navigation URL:', err.message);
      });
    }
  });

  // Send backend port to renderer after page loads
  // (renderer reads window.electronAPI.onBackendPort)
  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.webContents.send('backend-port', port);
    // Temporary debugging aid: keep DevTools open in AppImage builds.
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
    // Temporary debugging aid: keep DevTools open in AppImage builds.
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

function createSettingsWindow(options = {}) {
  const { editMode = false, autoSetup = false, condaNotFound = false } = options;

  let modeParam = 'setup';
  if (editMode) modeParam = 'edit';
  else if (condaNotFound) modeParam = 'conda-missing';
  else if (autoSetup) modeParam = 'autosetup';

  settingsWindow = new BrowserWindow({
    width:     580,
    height:    autoSetup ? 700 : 620,
    resizable: false,
    show:      true,
    title:     'SEA — First-Time Setup',
    icon:      APP_ICON_PATH,
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
    },
  });

  settingsWindow.setMenuBarVisibility(false);
  settingsWindow.loadFile(path.join(__dirname, 'settings.html'), {
    query: { mode: modeParam },
  });
  settingsWindow.on('closed', () => { settingsWindow = null; });
}

function openSettingsWindow(editMode = true) {
  if (settingsWindow) {
    settingsWindow.focus();
    return;
  }
  createSettingsWindow({ editMode });
}

function createAppMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Settings',
          accelerator: 'CmdOrCtrl+,',
          click: () => openSettingsWindow(true),
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        {
          label: 'SEA Repository',
          click: () => shell.openExternal('https://github.com/Davidlee2245/SEA_APP'),
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---------------------------------------------------------------------------
// App launch flow
// ---------------------------------------------------------------------------

async function launch() {
  const config = loadConfig();

  if (!config || !config.pythonExe) {
    if (!findCondaExecutable()) {
      createSettingsWindow({ condaNotFound: true });
      return;
    }
    createSettingsWindow({ autoSetup: true });
    return;
  }

  await launchWithConfig(config);
}

async function launchWithConfig(config) {
  isLaunching = true;
  createSplashWindow();

  try {
    const port     = await findFreePort(8765);
    const dataRoot = config.dataRoot || DEFAULT_DATA_ROOT;

    await startPythonBackend(config.pythonExe, dataRoot, port);
    startAgentBackend(config.pythonExe, port + 1);

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

/** Open a file picker for microscopy/image files from the main process. */
ipcMain.handle('select-image-file', async () => {
  const ownerWindow = mainWindow || settingsWindow || null;
  const result = await dialog.showOpenDialog(ownerWindow, {
    title: 'Select Image File',
    properties: ['openFile'],
    filters: [
      { name: 'Microscopy Images', extensions: ['tif', 'tiff', 'png', 'jpg', 'jpeg'] },
      { name: 'TIFF Files', extensions: ['tif', 'tiff'] },
      { name: 'All Files', extensions: ['*'] },
    ],
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

/** Save config from Settings opened while app is already running. */
ipcMain.handle('save-settings', async (_event, config) => {
  saveConfig(config);
  return true;
});

/** Return currently saved config (so settings window can pre-fill fields). */
ipcMain.handle('get-config', async () => {
  return loadConfig() || {};
});

/** Return the default data root so settings window can show it as placeholder. */
ipcMain.handle('get-default-data-root', async () => {
  return DEFAULT_DATA_ROOT;
});

/** Automated conda env + pip install; on success persists pythonExe and launches main app. */
ipcMain.handle('start-auto-setup', async (event) => {
  const wc = event.sender;
  try {
    const result = await autoSetupConda(wc);
    if (result.success) {
      const prev = loadConfig() || {};
      saveConfig({
        ...prev,
        pythonExe: result.pythonExe,
      });
      const nextCfg = loadConfig();
      if (settingsWindow && !settingsWindow.isDestroyed()) {
        settingsWindow.close();
      }
      await launchWithConfig(nextCfg || { pythonExe: result.pythonExe });
      return { success: true };
    }
    return result;
  } catch (e) {
    return { success: false, reason: 'error', message: e.message || String(e) };
  }
});

ipcMain.handle('open-external', async (_event, url) => {
  if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) return false;
  await shell.openExternal(url);
  return true;
});

ipcMain.handle('store:get', async (_event, key) => {
  if (typeof key !== 'string' || !key) return null;
  const store = loadStore();
  return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null;
});

ipcMain.handle('store:set', async (_event, key, value) => {
  if (typeof key !== 'string' || !key) return false;
  const store = loadStore();
  store[key] = value;
  return saveStore(store);
});

ipcMain.handle('store:remove', async (_event, key) => {
  if (typeof key !== 'string' || !key) return false;
  const store = loadStore();
  delete store[key];
  return saveStore(store);
});

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(() => {
  createAppMenu();
  launch();
});

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
  if (agentProcess) {
    console.log('[Main] Killing Agent backend...');
    try {
      if (process.platform === 'win32') {
        require('child_process').spawnSync('taskkill', ['/PID', String(agentProcess.pid), '/F', '/T']);
      } else {
        agentProcess.kill('SIGTERM');
      }
    } catch (e) {
      console.error('[Main] Failed to kill Agent process:', e.message);
    }
    agentProcess = null;
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
