'use strict';

/**
 * Electron Preload Script
 *
 * Runs in a privileged context but exposes only a narrow API surface to the
 * renderer (React app) via contextBridge.  This keeps nodeIntegration OFF
 * while still letting the renderer call Electron/Node features through IPC.
 *
 * Race-condition fix for backend-port:
 *   The main process sends 'backend-port' on 'did-finish-load', which fires
 *   before React's first useEffect runs.  We buffer the port here at the
 *   preload level so that onBackendPort() can fire the callback immediately
 *   even if the message already arrived by the time React registers its handler.
 */

const { contextBridge, ipcRenderer } = require('electron');

// ---------------------------------------------------------------------------
// Backend port — buffered so React never misses it regardless of timing
// ---------------------------------------------------------------------------

let _bufferedPort   = null;   // set as soon as the IPC message arrives
let _portCallback   = null;   // set when React calls onBackendPort()

// Prefer synchronous bootstrap from mainWindow.webPreferences.additionalArguments.
// This prevents renderer startup races where components fetch before async IPC.
const portArg = process.argv.find((arg) => arg.startsWith('--sea-backend-port='));
if (portArg) {
  const parsed = Number(portArg.split('=')[1]);
  if (!Number.isNaN(parsed) && parsed > 0) {
    _bufferedPort = parsed;
  }
}

ipcRenderer.on('backend-port', (_event, port) => {
  _bufferedPort = port;
  if (_portCallback) {
    _portCallback(port);
    _portCallback = null;
  }
});

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

contextBridge.exposeInMainWorld('electronAPI', {
  // ── Settings / Config ────────────────────────────────────────────────────

  /** Open a native file picker, return selected Python exe path (or null). */
  selectPython: () => ipcRenderer.invoke('select-python'),

  /** Open a native folder picker, return selected data root path (or null). */
  selectDataRoot: () => ipcRenderer.invoke('select-data-root'),

  /** Open a native image file picker, return selected path (or null). */
  selectImageFile: () => ipcRenderer.invoke('select-image-file'),

  /** Save config and launch the main app window. */
  launchAfterSettings: (config) => ipcRenderer.invoke('launch-after-settings', config),

  /** Save config while app is already running (no relaunch). */
  saveSettings: (config) => ipcRenderer.invoke('save-settings', config),

  /** Read current saved config. */
  getConfig: () => ipcRenderer.invoke('get-config'),

  /** Get the default data root path (shown as placeholder in settings). */
  getDefaultDataRoot: () => ipcRenderer.invoke('get-default-data-root'),

  /** Run automatic conda + pip setup (first-launch). Returns result object. */
  startAutoSetup: () => ipcRenderer.invoke('start-auto-setup'),

  /** Subscribe to auto-setup status lines (main → settings). */
  onAutoSetupProgress: (callback) => {
    const handler = (_event, message) => {
      if (typeof message === 'string') callback(message);
    };
    ipcRenderer.on('auto-setup-progress', handler);
  },

  /** Open a URL in the system browser (settings page). */
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  /** Ping localhost Ollama API (settings test button). */
  testOllamaConnection: () => ipcRenderer.invoke('test-ollama-connection'),

  // ── Runtime ──────────────────────────────────────────────────────────────

  /**
   * Register a callback that receives the resolved backend port.
   * Safe to call at any point — if the port already arrived it fires immediately.
   *
   * Usage in React:
   *   window.electronAPI?.onBackendPort((port) => setBackendPort(port));
   */
  onBackendPort: (callback) => {
    if (_bufferedPort !== null) {
      // Port already received before React mounted — fire immediately
      callback(_bufferedPort);
    } else {
      // Port not yet received — store callback, fired when IPC arrives
      _portCallback = callback;
    }
  },

  /** Get backend port synchronously if already known, otherwise null. */
  getBackendPort: () => _bufferedPort,

  // ── Persistent key/value storage (Electron userData) ─────────────────────
  storeGet: (key) => ipcRenderer.invoke('store:get', key),
  storeSet: (key, value) => ipcRenderer.invoke('store:set', key, value),
  storeRemove: (key) => ipcRenderer.invoke('store:remove', key),
});
