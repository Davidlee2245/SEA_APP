/**
 * Runtime API base URL resolution.
 *
 * Browser dev workflow (window.electronAPI is absent):
 *   getApiBase()   → ''                        (relative → Vite proxy → localhost:8765)
 *   getWsBase()    → 'ws://localhost:8765'      (direct WS to backend; must match api_server_extended --port)
 *   getAgentBase() → 'http://localhost:8766'    (agent chat; main backend port + 1, same as Electron)
 *
 * Electron production (window.electronAPI present):
 *   App.tsx calls setBackendPort(port) once when it receives the IPC message.
 *   After that, all getters return absolute URLs using the injected port.
 *
 * Usage in components:
 *   import { getApiBase, getWsBase, getAgentBase } from '../lib/apiBase';
 *
 *   fetch(`${getApiBase()}/api/health`)
 *   `${getWsBase()}/ws/pipeline`
 *   fetch(`${getAgentBase()}/api/agent/chat`)
 */

let _httpBase  = '';                       // empty → relative URLs for browser dev
let _wsBase    = 'ws://localhost:8765';    // default WS for browser dev (matches api_server_extended.py default)
let _agentBase = 'http://localhost:8766';  // agent server: next port after main API (see electron/main.js)

// Bootstrap from Electron synchronously when available to avoid startup races.
try {
  const initialPort = window.electronAPI?.getBackendPort?.();
  if (typeof initialPort === 'number' && Number.isFinite(initialPort) && initialPort > 0) {
    _httpBase = `http://localhost:${initialPort}`;
    _wsBase = `ws://localhost:${initialPort}`;
    _agentBase = `http://localhost:${initialPort + 1}`;
  }
} catch {
  // Non-Electron/browser contexts safely ignore bootstrap failures.
}

/** HTTP base for the main Flask API.  Empty string in browser dev mode. */
export function getApiBase(): string   { return _httpBase;  }

/** WebSocket base for the pipeline stream. */
export function getWsBase(): string    { return _wsBase;    }

/** HTTP base for the agent chat server (main API port + 1 in dev / Electron). */
export function getAgentBase(): string { return _agentBase; }

/**
 * Called once by App.tsx when Electron sends the resolved backend port.
 * Has no effect in browser dev mode (window.electronAPI is undefined there,
 * so App.tsx never calls this).
 */
export function setBackendPort(port: number): void {
  _httpBase  = `http://localhost:${port}`;
  _wsBase    = `ws://localhost:${port}`;
  // Agent server runs on the next port by convention
  _agentBase = `http://localhost:${port + 1}`;
}

// ── Type augmentation so TypeScript knows about window.electronAPI ──────────

declare global {
  interface Window {
    electronAPI?: {
      selectPython:        () => Promise<string | null>;
      selectDataRoot:      () => Promise<string | null>;
      selectImageFile:     () => Promise<string | null>;
      launchAfterSettings: (config: Record<string, unknown>) => Promise<void>;
      saveSettings:        (config: Record<string, unknown>) => Promise<boolean>;
      getConfig:           () => Promise<Record<string, unknown>>;
      getDefaultDataRoot:  () => Promise<string>;
      testOllamaConnection: () => Promise<{ ok: boolean; error?: string; modelCount?: number }>;
      onBackendPort:       (callback: (port: number) => void) => void;
      getBackendPort:      () => number | null;
      storeGet:            (key: string) => Promise<string | null>;
      storeSet:            (key: string, value: string) => Promise<boolean>;
      storeRemove:         (key: string) => Promise<boolean>;
    };
  }
}
