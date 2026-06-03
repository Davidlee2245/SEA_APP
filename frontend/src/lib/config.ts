/**
 * SEA user configuration — Electron sea-config.json, agent API, or browser localStorage.
 */

import { getAgentBase } from './apiBase';

export const SEA_CONFIG_STORAGE_KEY = 'sea-config';

export type LlmProvider = 'openai' | 'anthropic' | 'ollama';

export interface SeaConfig {
  pythonExe?: string;
  dataRoot?: string | null;
  openaiKey?: string | null;
  anthropicKey?: string | null;
  llmProvider?: LlmProvider;
  ollamaModel?: string;
  ollamaBinaryPath?: string;
}

/** Agent GET /api/agent/config response (keys may be masked). */
export interface AgentConfigResponse {
  ok?: boolean;
  llmProvider?: LlmProvider;
  ollamaModel?: string;
  openaiKey?: string | null;
  anthropicKey?: string | null;
  openaiKeySet?: boolean;
  anthropicKeySet?: boolean;
  configPath?: string | null;
}

export function isElectron(): boolean {
  return typeof window !== 'undefined' && !!window.electronAPI?.getConfig;
}

function readLocalStorageConfig(): SeaConfig {
  try {
    const raw = localStorage.getItem(SEA_CONFIG_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as SeaConfig;
      return parsed && typeof parsed === 'object' ? parsed : {};
    }
  } catch (e) {
    console.warn('[config] Failed to read localStorage:', e);
  }
  return {};
}

/** Fetch LLM config from the running agent backend (web dev). */
export async function fetchAgentConfig(): Promise<AgentConfigResponse | null> {
  try {
    const res = await fetch(`${getAgentBase()}/api/agent/config`);
    if (!res.ok) return null;
    const data = (await res.json()) as AgentConfigResponse;
    return data?.ok !== false ? data : null;
  } catch {
    return null;
  }
}

function isMaskedKey(value: string | null | undefined): boolean {
  return typeof value === 'string' && value.endsWith('...');
}

/** Merge agent API config over local/Electron config (prefer backend for LLM fields). */
export function mergeAgentConfigInto(
  base: SeaConfig,
  agent: AgentConfigResponse | null
): SeaConfig {
  if (!agent) return base;
  const merged: SeaConfig = { ...base };
  if (agent.llmProvider) merged.llmProvider = agent.llmProvider;
  if (agent.ollamaModel) merged.ollamaModel = agent.ollamaModel;
  if (agent.openaiKey && !isMaskedKey(agent.openaiKey)) {
    merged.openaiKey = agent.openaiKey;
  }
  if (agent.anthropicKey && !isMaskedKey(agent.anthropicKey)) {
    merged.anthropicKey = agent.anthropicKey;
  }
  return merged;
}

/** Load config: Electron IPC, else agent API + localStorage (web). */
export async function getConfig(): Promise<SeaConfig> {
  if (window.electronAPI?.getConfig) {
    return (await window.electronAPI.getConfig()) as SeaConfig;
  }
  const local = readLocalStorageConfig();
  const agent = await fetchAgentConfig();
  return mergeAgentConfigInto(local, agent);
}

/** LLM fields pushed to agent backend for live reload (web dev). */
export function agentConfigPayload(config: SeaConfig): Record<string, string | null> {
  return {
    llmProvider: config.llmProvider || 'openai',
    ollamaModel: config.ollamaModel || 'llava',
    openaiKey: config.openaiKey ?? null,
    anthropicKey: config.anthropicKey ?? null,
  };
}

/** POST LLM settings to agent backend; returns true if applied. */
export async function pushAgentConfig(config: SeaConfig): Promise<boolean> {
  try {
    const res = await fetch(`${getAgentBase()}/api/agent/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(agentConfigPayload(config)),
    });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.ok === true;
  } catch {
    return false;
  }
}

export type SaveConfigResult = {
  backendApplied: boolean;
};

/** Persist config: Electron IPC, or localStorage + agent API (web). */
export async function saveConfig(config: SeaConfig): Promise<SaveConfigResult> {
  if (window.electronAPI?.saveSettings) {
    await window.electronAPI.saveSettings(config as Record<string, unknown>);
    return { backendApplied: false };
  }
  localStorage.setItem(SEA_CONFIG_STORAGE_KEY, JSON.stringify(config, null, 2));
  const backendApplied = await pushAgentConfig(config);
  return { backendApplied };
}
