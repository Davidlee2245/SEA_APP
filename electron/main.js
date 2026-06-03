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
const { spawn, spawnSync, execSync } = require('child_process');
const os           = require('os');
const path       = require('path');
const fs         = require('fs');
const net        = require('net');
const https      = require('https');
const http       = require('http');
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

const OLLAMA_API_URL = 'http://127.0.0.1:11434';
const OLLAMA_DEFAULT_MODEL = 'llava';
const OLLAMA_LINUX_USER_BIN = path.join(os.homedir(), '.local', 'bin', 'ollama');

let ollamaProcess = null;
/** Resolved absolute path to ollama binary for spawn() calls. */
let ollamaBinaryPath = null;

function getLlmDefaults() {
  return { llmProvider: 'openai', ollamaModel: OLLAMA_DEFAULT_MODEL };
}

/** Layer LLM provider settings into a child-process environment. */
function applyLlmEnv(childEnv, config) {
  const cfg = config || loadConfig() || {};
  const defaults = getLlmDefaults();
  childEnv.LLM_PROVIDER = cfg.llmProvider || defaults.llmProvider;
  childEnv.OLLAMA_MODEL = cfg.ollamaModel || defaults.ollamaModel;
  if (cfg.openaiKey) childEnv.OPENAI_API_KEY = cfg.openaiKey;
  if (cfg.anthropicKey) childEnv.ANTHROPIC_API_KEY = cfg.anthropicKey;
}

function updateSplashStatus(message) {
  if (!splashWindow || splashWindow.isDestroyed()) return;
  splashWindow.webContents
    .executeJavaScript(
      `(function(){ var p=document.querySelector('p'); if(p) p.textContent=${JSON.stringify(message)}; })();`
    )
    .catch(() => {});
}

function runCommand(cmd, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, {
      env: process.env,
      shell: false,
      ...options,
    });
    let stdout = '';
    let stderr = '';
    child.stdout?.on('data', (d) => { stdout += d.toString(); });
    child.stderr?.on('data', (d) => { stderr += d.toString(); });
    child.on('close', (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
    child.on('error', (err) => {
      resolve({ code: 1, stdout, stderr: err.message });
    });
  });
}

function checkOllamaVersion(binary) {
  return new Promise((resolve) => {
    const child = spawn(binary, ['--version'], {
      env: process.env,
      shell: false,
      windowsHide: true,
    });
    child.on('error', () => resolve(false));
    child.on('close', (code) => resolve(code === 0));
  });
}

function resolveOllamaBinary(config) {
  const cfg = config || loadConfig() || {};
  if (cfg.ollamaBinaryPath && fs.existsSync(cfg.ollamaBinaryPath)) {
    return cfg.ollamaBinaryPath;
  }
  if (fs.existsSync(OLLAMA_LINUX_USER_BIN)) {
    return OLLAMA_LINUX_USER_BIN;
  }
  return 'ollama';
}

function downloadFile(url, destPath, onProgress) {
  return new Promise((resolve, reject) => {
    const follow = (requestUrl) => {
      const lib = requestUrl.startsWith('https') ? https : http;
      lib.get(requestUrl, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          follow(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`Download failed: HTTP ${res.statusCode}`));
          return;
        }
        const total = parseInt(res.headers['content-length'], 10) || 0;
        let downloaded = 0;
        const file = fs.createWriteStream(destPath);
        res.on('data', (chunk) => {
          downloaded += chunk.length;
          if (onProgress && total > 0) onProgress(downloaded / total);
        });
        res.pipe(file);
        file.on('finish', () => {
          file.close(() => resolve(destPath));
        });
        file.on('error', reject);
      }).on('error', reject);
    };
    follow(url);
  });
}

function appendPathToBashrcIfNeeded() {
  const bashrc = path.join(os.homedir(), '.bashrc');
  const line = 'export PATH="$HOME/.local/bin:$PATH"';
  try {
    const existing = fs.existsSync(bashrc) ? fs.readFileSync(bashrc, 'utf8') : '';
    if (!existing.includes('.local/bin')) {
      fs.appendFileSync(bashrc, `\n# Added by SEA for Ollama\n${line}\n`);
    }
  } catch (e) {
    console.warn('[Ollama] Could not update ~/.bashrc:', e.message);
  }
  const localBin = path.dirname(OLLAMA_LINUX_USER_BIN);
  if (!process.env.PATH.split(path.delimiter).includes(localBin)) {
    process.env.PATH = `${localBin}${path.delimiter}${process.env.PATH}`;
  }
}

async function installOllamaLinux(statusFn) {
  statusFn('Installing Ollama (system installer)…');
  const official = await runCommand('sh', [
    '-c',
    'curl -fsSL https://ollama.com/install.sh | sh',
  ], { shell: false });

  if (official.code === 0) {
    const binary = resolveOllamaBinary({});
    if (await checkOllamaVersion(binary)) {
      return { success: true, binaryPath: binary };
    }
  }

  statusFn('System install unavailable — installing to ~/.local/bin…');
  fs.mkdirSync(path.dirname(OLLAMA_LINUX_USER_BIN), { recursive: true });
  const dl = await runCommand('curl', [
    '-L', 'https://ollama.com/download/ollama-linux-amd64',
    '-o', OLLAMA_LINUX_USER_BIN,
  ]);
  if (dl.code !== 0) {
    return { success: false, message: dl.stderr || 'Failed to download Ollama binary.' };
  }
  fs.chmodSync(OLLAMA_LINUX_USER_BIN, 0o755);
  appendPathToBashrcIfNeeded();
  return { success: true, binaryPath: OLLAMA_LINUX_USER_BIN };
}

async function installOllamaWindows(statusFn) {
  const dest = path.join(os.tmpdir(), 'OllamaSetup.exe');
  statusFn('Downloading Ollama installer…');
  try {
    await downloadFile(
      'https://ollama.com/download/OllamaSetup.exe',
      dest,
      (pct) => statusFn(`Downloading Ollama… ${Math.round(pct * 100)}%`),
    );
  } catch (e) {
    return { success: false, message: e.message };
  }
  statusFn('Running Ollama installer (silent)…');
  const install = spawnSync(dest, ['/S'], {
    encoding: 'utf8',
    windowsHide: true,
    timeout: 600_000,
  });
  if (install.status !== 0) {
    return {
      success: false,
      message: (install.stderr || install.stdout || 'Installer exited with an error.').trim(),
    };
  }
  return { success: true, binaryPath: 'ollama' };
}

async function installOllamaDarwin(statusFn) {
  statusFn('Opening Ollama download page…');
  await shell.openExternal('https://ollama.com/download/Ollama-darwin.dmg');
  const choice = dialog.showMessageBoxSync({
    type: 'info',
    title: 'Install Ollama',
    message: 'After installing Ollama from the downloaded disk image, click Continue.',
    buttons: ['Continue', 'Cancel'],
    defaultId: 0,
    cancelId: 1,
  });
  if (choice !== 0) {
    return { success: false, message: 'Installation cancelled.' };
  }
  const binary = resolveOllamaBinary({});
  if (!(await checkOllamaVersion(binary))) {
    return { success: false, message: 'Ollama is still not available. Install it and try again.' };
  }
  return { success: true, binaryPath: binary };
}

async function installOllama(statusFn) {
  const status = statusFn || (() => {});
  if (process.platform === 'linux') return installOllamaLinux(status);
  if (process.platform === 'win32') return installOllamaWindows(status);
  if (process.platform === 'darwin') return installOllamaDarwin(status);
  return { success: false, message: `Ollama auto-install is not supported on ${process.platform}.` };
}

async function isOllamaApiUp() {
  try {
    const res = await fetch(`${OLLAMA_API_URL}/api/tags`);
    return res.ok;
  } catch {
    return false;
  }
}

function waitForOllamaApi(timeoutMs = 60_000, intervalMs = 500) {
  const started = Date.now();
  return new Promise((resolve) => {
    const tick = async () => {
      if (await isOllamaApiUp()) {
        resolve(true);
        return;
      }
      if (Date.now() - started >= timeoutMs) {
        resolve(false);
        return;
      }
      setTimeout(tick, intervalMs);
    };
    tick();
  });
}

function startOllamaPull(binary, model) {
  console.log('[Ollama] Pulling model in background:', model);
  const pull = spawn(binary, ['pull', model], {
    env: process.env,
    shell: false,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  pull.stdout.on('data', (chunk) => {
    process.stdout.write('[Ollama pull] ' + chunk.toString());
  });
  pull.stderr.on('data', (chunk) => {
    process.stderr.write('[Ollama pull] ' + chunk.toString());
  });
  pull.on('close', (code) => {
    console.log(`[Ollama] pull ${model} finished with code ${code}`);
  });
}

async function ensureOllamaServeRunning(binary, model, statusFn) {
  const status = statusFn || (() => {});
  ollamaBinaryPath = binary;

  if (await isOllamaApiUp()) {
    status('Ollama is running.');
    startOllamaPull(binary, model);
    return true;
  }

  status('Starting Ollama server…');
  ollamaProcess = spawn(binary, ['serve'], {
    env: process.env,
    shell: false,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  ollamaProcess.stdout?.on('data', (chunk) => {
    process.stdout.write('[Ollama] ' + chunk.toString());
  });
  ollamaProcess.stderr?.on('data', (chunk) => {
    process.stderr.write('[Ollama] ' + chunk.toString());
  });
  ollamaProcess.on('exit', (code, signal) => {
    console.log(`[Ollama] serve exited — code: ${code}, signal: ${signal}`);
    ollamaProcess = null;
  });
  ollamaProcess.on('error', (err) => {
    console.warn('[Ollama] serve failed to start:', err.message);
    ollamaProcess = null;
  });

  const ready = await waitForOllamaApi(90_000);
  if (!ready) {
    console.warn('[Ollama] API did not become ready within timeout.');
    return false;
  }

  status(`Ollama ready — downloading model "${model}" if needed…`);
  startOllamaPull(binary, model);
  return true;
}

function fallbackConfigToOpenAI(config, reason) {
  const next = { ...config, llmProvider: 'openai' };
  saveConfig(next);
  dialog.showMessageBoxSync({
    type: 'warning',
    title: 'Ollama Unavailable',
    message: 'Could not use local Ollama — falling back to OpenAI.',
    detail: reason || 'Install Ollama later from Settings and choose the Ollama provider.',
    buttons: ['OK'],
  });
  return next;
}

/**
 * When llmProvider is ollama: detect/install binary, start serve, pull model.
 * On failure or user skip, switches config to openai.
 */
async function ensureOllamaReady(config, statusFn) {
  const cfg = { ...getLlmDefaults(), ...config };
  if (cfg.llmProvider !== 'ollama') {
    return cfg;
  }

  const status = statusFn || (() => {});
  const model = cfg.ollamaModel || OLLAMA_DEFAULT_MODEL;
  let binary = resolveOllamaBinary(cfg);

  if (!(await checkOllamaVersion(binary))) {
    const install = dialog.showMessageBoxSync({
      type: 'question',
      title: 'Install Ollama',
      message: 'Ollama (local AI) is not installed. Install it now? (~500MB)',
      buttons: ['Install', 'Skip - Use OpenAI instead'],
      defaultId: 0,
      cancelId: 1,
    });
    if (install !== 0) {
      return fallbackConfigToOpenAI(cfg, 'Ollama installation was skipped.');
    }

    status('Installing Ollama…');
    const result = await installOllama(status);
    if (!result.success) {
      return fallbackConfigToOpenAI(
        cfg,
        result.message || 'Ollama installation failed.'
      );
    }
    binary = result.binaryPath;
    cfg.ollamaBinaryPath = binary;
    saveConfig(cfg);

    if (!(await checkOllamaVersion(binary))) {
      return fallbackConfigToOpenAI(cfg, 'Ollama was installed but `ollama --version` still fails.');
    }
    status('Ollama installed successfully.');
  } else {
    ollamaBinaryPath = binary;
  }

  const running = await ensureOllamaServeRunning(binary, model, status);
  if (!running) {
    return fallbackConfigToOpenAI(cfg, 'Ollama server did not start on localhost:11434.');
  }

  cfg.ollamaBinaryPath = binary;
  return cfg;
}

function killOllama() {
  if (ollamaProcess && ollamaProcess.pid) {
    const pid = ollamaProcess.pid;
    console.log('[Ollama] Terminating tracked serve process PID', pid);
    terminatePidTree(pid, 'Ollama serve (tracked)');
    try {
      ollamaProcess.removeAllListeners?.();
    } catch (_) {}
    ollamaProcess = null;
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

/** Flask backend prefers this port; must match api_server_extended.py default. */
const BACKEND_PREFERRED_PORT = 8765;

/** True when `:` + port appears as its own numeric suffix (avoid :187659 false matches). */
function lineLooksLikeTcpListenOnPort(line, port) {
  const mark = ':' + port;
  let i = 0;
  while ((i = line.indexOf(mark, i)) !== -1) {
    const prevOk = i === 0 || !/\d/.test(line[i - 1]);
    const tail = line.slice(i + mark.length);
    const nextOk = !tail || !/^\d/.test(tail);
    if (prevOk && nextOk) return true;
    i += 1;
  }
  return false;
}

/**
 * Collect PIDs that have a TCP LISTEN socket on localhost for `port`.
 * Windows: parse `netstat -ano`; macOS/Linux: `lsof`, then optional `ss` fallback.
 */
function getListeningPidsOnPort(port) {
  const pids = [];
  const seen = new Set();

  function add(pid) {
    const n = parseInt(pid, 10);
    if (!Number.isFinite(n) || n <= 0 || n === process.pid) return;
    if (!seen.has(n)) {
      seen.add(n);
      pids.push(n);
    }
  }

  if (process.platform === 'win32') {
    try {
      const out = execSync('netstat -ano', { encoding: 'utf8', windowsHide: true, timeout: 15000 });
      for (const line of out.split(/\r?\n/)) {
        if (!/\bLISTENING\b/i.test(line)) continue;
        if (!/^TCP/i.test(line.trim())) continue;
        if (!lineLooksLikeTcpListenOnPort(line, port)) continue;
        const m = line.match(/\s(\d+)\s*$/);
        if (m) add(m[1]);
      }
    } catch (err) {
      console.warn('[Backend Lifecycle] netstat -ano failed:', err.message);
    }
    return pids;
  }

  try {
    const out = execSync(`lsof -nP -iTCP:${port} -sTCP:LISTEN -t`, {
      encoding: 'utf8',
      timeout: 15000,
    });
    for (const line of out.trim().split(/\s*\n+/).map((s) => s.trim()).filter(Boolean)) {
      add(line);
    }
    if (pids.length > 0) return pids;
  } catch (_) {
    /* lsof missing or no listeners */
  }

  try {
    const out = execSync(`sh -c 'ss -lptn 2>/dev/null || true'`, { encoding: 'utf8', timeout: 15000 });
    for (const line of out.split(/\r?\n/)) {
      if (!lineLooksLikeTcpListenOnPort(line, port)) continue;
      const m = line.match(/pid=(\d+)/i);
      if (m) add(m[1]);
    }
  } catch (_) {
    /* ss optional */
  }

  return pids;
}

/** Force-terminate process tree / PID (used for orphans on our preferred port). */
function terminatePidTree(pid, description) {
  if (!pid) return;
  const label = description || 'PID ' + pid;
  try {
    if (process.platform === 'win32') {
      console.log('[Backend Lifecycle]', label + ': taskkill /F /T for PID', pid);
      spawnSync('taskkill', ['/PID', String(pid), '/F', '/T'], {
        encoding: 'utf8',
        windowsHide: true,
        timeout: 20000,
      });
    } else {
      console.log('[Backend Lifecycle]', label + ': SIGKILL for PID', pid);
      try {
        process.kill(pid, 'SIGKILL');
      } catch (sigErr) {
        console.warn('[Backend Lifecycle] process.kill SIGKILL:', sigErr.message, '— trying /bin/sh kill -9');
        spawnSync('/bin/sh', ['-c', `kill -9 ${pid} 2>/dev/null || true`], { encoding: 'utf8', timeout: 10000 });
      }
    }
  } catch (e) {
    console.warn('[Backend Lifecycle] terminatePidTree PID', pid, ':', e.message);
  }
}

/**
 * Before starting Flask: release BACKEND_PREFERRED_PORT if something else holds it.
 */
async function ensurePreferredBackendPortFree() {
  const port = BACKEND_PREFERRED_PORT;
  const pids = getListeningPidsOnPort(port);
  if (!pids.length) {
    console.log(`[Backend Lifecycle] Preferred port ${port} is available (no listener PIDs detected).`);
    return;
  }
  console.warn(
    `[Backend Lifecycle] Preferred port ${port} is busy — terminating listener PID(s): ${pids.join(', ')}`
  );
  for (const pid of pids) {
    terminatePidTree(pid, `Stale listener occupying port ${port}`);
  }
  await new Promise((resolve) => setTimeout(resolve, 400));
}

function findFreePort(start = BACKEND_PREFERRED_PORT) {
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

    const childEnv = { ...process.env };
    const config   = loadConfig() || {};
    applyLlmEnv(childEnv, config);

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
  applyLlmEnv(childEnv, config);

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
    updateSplashStatus('Checking local AI (Ollama)…');
    const resolvedConfig = await ensureOllamaReady(config, updateSplashStatus);

    await ensurePreferredBackendPortFree();
    const port     = await findFreePort(BACKEND_PREFERRED_PORT);
    const dataRoot = resolvedConfig.dataRoot || DEFAULT_DATA_ROOT;

    updateSplashStatus('Starting Python backend — this may take up to 90 s on first run…');
    await startPythonBackend(resolvedConfig.pythonExe, dataRoot, port);
    startAgentBackend(resolvedConfig.pythonExe, port + 1);

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

/** Ping Ollama HTTP API (settings "Test Connection"). */
ipcMain.handle('test-ollama-connection', async () => {
  try {
    const res = await fetch(`${OLLAMA_API_URL}/api/tags`);
    if (!res.ok) {
      return { ok: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    const count = Array.isArray(data.models) ? data.models.length : 0;
    return { ok: true, modelCount: count };
  } catch (e) {
    return { ok: false, error: e.message || String(e) };
  }
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
  if (process.platform !== 'darwin') {
    console.log('[Backend Lifecycle] window-all-closed (non-macOS): stopping backends before quit');
    killPython();
    killOllama();
    app.quit();
  }
});

app.on('activate', () => {
  // macOS: re-open window on dock icon click
  if (mainWindow === null && settingsWindow === null) launch();
});

function killPython() {
  if (pythonProcess && pythonProcess.pid) {
    const pid = pythonProcess.pid;
    console.log('[Backend Lifecycle] Terminating tracked Flask backend PID', pid);
    terminatePidTree(pid, 'Flask backend (tracked)');
    try {
      pythonProcess.removeAllListeners?.();
    } catch (_) {}
    pythonProcess = null;
  }
  if (agentProcess && agentProcess.pid) {
    const pid = agentProcess.pid;
    console.log('[Backend Lifecycle] Terminating tracked Agent backend PID', pid);
    terminatePidTree(pid, 'Agent backend (tracked)');
    try {
      agentProcess.removeAllListeners?.();
    } catch (_) {}
    agentProcess = null;
  }
}

app.on('before-quit', () => {
  console.log('[Backend Lifecycle] before-quit: stopping backends');
  killPython();
  killOllama();
});

// Fires on clean quit (Cmd+Q, window close, app.quit())
app.on('will-quit', () => {
  console.log('[Backend Lifecycle] will-quit: ensuring backends are stopped');
  killPython();
  killOllama();
});

// Belt-and-suspenders: also fires on process.exit() and uncaught crashes
// so child processes don't become orphans if Electron hard-crashes.
function killAllChildProcesses() {
  killPython();
  killOllama();
}

process.on('exit',           killAllChildProcesses);
process.on('SIGINT',         () => { killAllChildProcesses(); process.exit(0); });
process.on('SIGTERM',        () => { killAllChildProcesses(); process.exit(0); });
process.on('uncaughtException', (err) => {
  console.error('[Main] Uncaught exception:', err);
  killAllChildProcesses();
  process.exit(1);
});
