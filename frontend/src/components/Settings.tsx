/**
 * In-app Settings — Electron (duplicate of settings window) and web dev (localStorage).
 */

import React, { useCallback, useEffect, useState } from 'react';
import '../styles/Settings.css';
import {
  getConfig,
  saveConfig,
  isElectron,
  type LlmProvider,
  type SeaConfig,
} from '../lib/config';

const DEFAULT_LLM_PROVIDER: LlmProvider = 'openai';
const DEFAULT_OLLAMA_MODEL = 'llama3.2-vision:90b';
const OLLAMA_TAGS_URL = 'http://localhost:11434/api/tags';

async function fetchOllamaTags(): Promise<
  { ok: true; models: string[] } | { ok: false; error: string }
> {
  try {
    const res = await fetch(OLLAMA_TAGS_URL);
    if (!res.ok) {
      return { ok: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    const models = Array.isArray(data.models)
      ? data.models
          .map((m: { name?: string }) => m?.name)
          .filter((name): name is string => typeof name === 'string' && name.length > 0)
          .sort((a, b) => a.localeCompare(b))
      : [];
    return { ok: true, models };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { ok: false, error: msg };
  }
}

async function testOllamaConnection(): Promise<{ ok: boolean; error?: string; modelCount?: number }> {
  if (window.electronAPI?.testOllamaConnection) {
    return window.electronAPI.testOllamaConnection();
  }
  const result = await fetchOllamaTags();
  if (!result.ok) {
    return { ok: false, error: result.error };
  }
  return { ok: true, modelCount: result.models.length };
}

const Settings: React.FC = () => {
  const [pythonExe, setPythonExe] = useState('');
  const [dataRoot, setDataRoot] = useState('');
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [llmProvider, setLlmProvider] = useState<LlmProvider>(DEFAULT_LLM_PROVIDER);
  const [ollamaModel, setOllamaModel] = useState(DEFAULT_OLLAMA_MODEL);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [ollamaTestMsg, setOllamaTestMsg] = useState('');
  const [ollamaTestOk, setOllamaTestOk] = useState<boolean | null>(null);
  const [testingOllama, setTestingOllama] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false);
  const [ollamaModelsFetchFailed, setOllamaModelsFetchFailed] = useState(false);

  const inElectron = isElectron();

  const loadOllamaModels = useCallback(async () => {
    setOllamaModelsLoading(true);
    setOllamaModelsFetchFailed(false);
    const result = await fetchOllamaTags();
    if (result.ok) {
      setOllamaModels(result.models);
      setOllamaModelsFetchFailed(result.models.length === 0);
    } else {
      setOllamaModels([]);
      setOllamaModelsFetchFailed(true);
    }
    setOllamaModelsLoading(false);
  }, []);

  useEffect(() => {
    if (llmProvider !== 'ollama') return;
    loadOllamaModels();
  }, [llmProvider, loadOllamaModels]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const cfg = await getConfig();
        if (cancelled) return;
        if (cfg.pythonExe) setPythonExe(cfg.pythonExe);
        if (cfg.dataRoot) setDataRoot(cfg.dataRoot);
        if (cfg.openaiKey) setOpenaiKey(cfg.openaiKey);
        if (cfg.anthropicKey) setAnthropicKey(cfg.anthropicKey);
        if (cfg.llmProvider) setLlmProvider(cfg.llmProvider);
        if (cfg.ollamaModel) setOllamaModel(cfg.ollamaModel);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const buildConfig = useCallback((): SeaConfig => ({
    pythonExe: pythonExe.trim() || undefined,
    dataRoot: dataRoot.trim() || null,
    openaiKey: openaiKey.trim() || null,
    anthropicKey: anthropicKey.trim() || null,
    llmProvider: llmProvider || DEFAULT_LLM_PROVIDER,
    ollamaModel: ollamaModel.trim() || DEFAULT_OLLAMA_MODEL,
  }), [pythonExe, dataRoot, openaiKey, anthropicKey, llmProvider, ollamaModel]);

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage('');
    try {
      const { backendApplied } = await saveConfig(buildConfig());
      if (inElectron) {
        setSaveMessage('✓ Settings saved');
      } else if (backendApplied) {
        setSaveMessage('✓ Settings saved and applied to backend');
      } else {
        setSaveMessage('✓ Settings saved (backend not running — start it to apply)');
      }
    } catch (e) {
      setSaveMessage('✗ ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setSaving(false);
    }
  };

  const handleTestOllama = async () => {
    setTestingOllama(true);
    setOllamaTestMsg('Testing…');
    setOllamaTestOk(null);
    try {
      const result = await testOllamaConnection();
      if (result.ok) {
        const n = result.modelCount != null ? ` (${result.modelCount} model(s))` : '';
        setOllamaTestMsg('✓ Ollama is reachable at localhost:11434' + n);
        setOllamaTestOk(true);
      } else {
        setOllamaTestMsg('✗ ' + (result.error || 'Could not reach Ollama'));
        setOllamaTestOk(false);
      }
    } catch (e) {
      setOllamaTestMsg('✗ ' + (e instanceof Error ? e.message : String(e)));
      setOllamaTestOk(false);
    } finally {
      setTestingOllama(false);
    }
  };

  const showOllamaFields = llmProvider === 'ollama';
  const ollamaModelOptions =
    ollamaModel && !ollamaModels.includes(ollamaModel)
      ? [ollamaModel, ...ollamaModels]
      : ollamaModels;
  const showOllamaModelDropdown =
    showOllamaFields &&
    !ollamaModelsLoading &&
    !ollamaModelsFetchFailed &&
    ollamaModelOptions.length > 0;

  if (loading) {
    return (
      <div className="settings-panel">
        <p className="settings-status">Loading settings…</p>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      <div className="settings-panel-header">
        <h2>⚙ SEA Settings</h2>
        <p>
          {inElectron
            ? 'Changes are saved to your profile. Restart the app to apply Python path and data root changes to backends.'
            : 'Configure API keys and LLM options for local development. Python paths apply when using Electron.'}
        </p>
      </div>

      {!inElectron && (
        <div className="settings-banner-web" role="note">
          Browser mode: LLM settings are saved to <code>localStorage</code> and pushed to the agent API
          (<code>npm run server</code> on port 8766), which writes{' '}
          <code>~/.config/sea-exosome-analysis/sea-config.json</code>.
        </div>
      )}

      <div className="settings-field">
        <label htmlFor="settings-pythonExe">Python Executable</label>
        <div className="settings-hint">
          Path to the <code>python</code> binary in your SEA conda environment.
        </div>
        <div className="settings-input-row">
          <input
            id="settings-pythonExe"
            type="text"
            value={pythonExe}
            onChange={(e) => setPythonExe(e.target.value)}
            placeholder="/path/to/anaconda3/envs/SEA/bin/python"
            spellCheck={false}
          />
        </div>
      </div>

      <hr className="settings-divider" />

      <div className="settings-field">
        <label htmlFor="settings-dataRoot">Data Root</label>
        <div className="settings-hint">Folder for input images and analysis results.</div>
        <div className="settings-input-row">
          <input
            id="settings-dataRoot"
            type="text"
            value={dataRoot}
            onChange={(e) => setDataRoot(e.target.value)}
            placeholder="Leave blank for default (Electron: Documents/SEA/data)"
            spellCheck={false}
          />
        </div>
      </div>

      <hr className="settings-divider" />

      <div className="settings-field">
        <label htmlFor="settings-openaiKey">OpenAI API Key</label>
        <div className="settings-hint">Optional — required for cloud OpenAI LLM features.</div>
        <div className="settings-input-row">
          <input
            id="settings-openaiKey"
            type="password"
            value={openaiKey}
            onChange={(e) => setOpenaiKey(e.target.value)}
            placeholder="sk-..."
            autoComplete="off"
          />
        </div>
      </div>

      <div className="settings-field">
        <label htmlFor="settings-anthropicKey">Anthropic API Key</label>
        <div className="settings-hint">Optional.</div>
        <div className="settings-input-row">
          <input
            id="settings-anthropicKey"
            type="password"
            value={anthropicKey}
            onChange={(e) => setAnthropicKey(e.target.value)}
            placeholder="sk-ant-..."
            autoComplete="off"
          />
        </div>
      </div>

      <hr className="settings-divider" />

      <div className="settings-field">
        <label htmlFor="settings-llmProvider">LLM Provider</label>
        <div className="settings-hint">
          OpenAI or Anthropic (cloud), or Ollama (local at localhost:11434).
        </div>
        <div className="settings-input-row">
          <select
            id="settings-llmProvider"
            value={llmProvider}
            onChange={(e) => {
              setLlmProvider(e.target.value as LlmProvider);
              setOllamaTestMsg('');
              setOllamaTestOk(null);
            }}
          >
            <option value="openai">OpenAI (cloud)</option>
            <option value="anthropic">Anthropic (cloud)</option>
            <option value="ollama">Ollama (local)</option>
          </select>
        </div>
      </div>

      {showOllamaFields && (
        <div className="settings-field">
          <label htmlFor="settings-ollamaModel">Ollama Model</label>
          <div className="settings-hint">Vision model recommended for image analysis.</div>
          {ollamaModelsLoading && (
            <p className="settings-status settings-models-loading">Loading models…</p>
          )}
          <div className="settings-input-row">
            {showOllamaModelDropdown ? (
              <select
                id="settings-ollamaModel"
                value={ollamaModelOptions.includes(ollamaModel) ? ollamaModel : ollamaModelOptions[0]}
                onChange={(e) => setOllamaModel(e.target.value)}
              >
                {ollamaModelOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              !ollamaModelsLoading && (
                <input
                  id="settings-ollamaModel"
                  type="text"
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                  placeholder={DEFAULT_OLLAMA_MODEL}
                  spellCheck={false}
                />
              )
            )}
            <button
              type="button"
              className="settings-btn-secondary"
              onClick={loadOllamaModels}
              disabled={ollamaModelsLoading}
              title="Refresh model list"
              aria-label="Refresh model list"
            >
              🔄
            </button>
            <button
              type="button"
              className="settings-btn-secondary"
              onClick={handleTestOllama}
              disabled={testingOllama || ollamaModelsLoading}
            >
              Test Connection
            </button>
          </div>
          {!ollamaModelsLoading && ollamaModelsFetchFailed && (
            <div className="settings-validation err">
              Could not load models — is Ollama running?
            </div>
          )}
          {ollamaTestMsg && (
            <div
              className={`settings-validation ${ollamaTestOk === true ? 'ok' : ollamaTestOk === false ? 'err' : ''}`}
            >
              {ollamaTestMsg}
            </div>
          )}
        </div>
      )}

      <div className="settings-footer">
        <button
          type="button"
          className="settings-btn-save"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving…' : 'Save Settings'}
        </button>
        {saveMessage && (
          <span className={`settings-status ${saveMessage.startsWith('✓') ? 'ok' : ''}`}>
            {saveMessage}
          </span>
        )}
      </div>
    </div>
  );
};

export default Settings;
