import { getApiBase } from './apiBase';

/** Short-lived cache so multiple tabs don't hammer GET /api/state/<sample>/load */
const diskStateCache = new Map<string, { t: number; data: Record<string, unknown> }>();
const DISK_CACHE_MS = 4000;

export function invalidateSampleStateDiskCache(sample: string): void {
  if (sample) diskStateCache.delete(sample);
}

export async function get(key: string): Promise<string | null> {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export async function set(key: string, value: string): Promise<void> {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Ignore localStorage write failures.
  }
}

export async function remove(key: string): Promise<void> {
  try {
    localStorage.removeItem(key);
  } catch {
    // Ignore localStorage remove failures.
  }
}

export type SeaStateKey = 'preprocess' | 'alignment' | 'exosome' | 'ui';

/**
 * Persist one slice of sample-local state under <dataRoot>/input/<sample>/.sea_state/
 * (Flask POST /api/state/<sample>/save). Invalidates short-lived disk cache for that sample.
 */
export async function saveStateToDisk(
  sample: string,
  stateKey: SeaStateKey,
  data: object,
): Promise<boolean> {
  if (!sample) return false;
  invalidateSampleStateDiskCache(sample);
  try {
    const res = await fetch(
      `${getApiBase()}/api/state/${encodeURIComponent(sample)}/save`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [stateKey]: data }),
      },
    );
    if (!res.ok) {
      console.warn('[sea_state] save failed', res.status, await res.text().catch(() => ''));
      return false;
    }
    const j = await res.json().catch(() => ({}));
    return !!j.success;
  } catch (e) {
    console.warn('[sea_state] save error', e);
    return false;
  }
}

/**
 * Load all .sea_state/*.json for this sample (GET /api/state/<sample>/load).
 * @param force bypass short-lived in-memory cache
 */
export async function loadStateFromDisk(
  sample: string,
  force = false,
): Promise<Record<string, unknown>> {
  if (!sample) return {};
  if (!force) {
    const hit = diskStateCache.get(sample);
    if (hit && Date.now() - hit.t < DISK_CACHE_MS) {
      return { ...hit.data };
    }
  }
  try {
    const res = await fetch(`${getApiBase()}/api/state/${encodeURIComponent(sample)}/load`);
    if (!res.ok) {
      console.warn('[sea_state] load failed', res.status);
      return {};
    }
    const j = await res.json().catch(() => ({}));
    const data = (j && j.success && j.data && typeof j.data === 'object') ? j.data : {};
    diskStateCache.set(sample, { t: Date.now(), data: data as Record<string, unknown> });
    return { ...(data as Record<string, unknown>) };
  } catch (e) {
    console.warn('[sea_state] load error', e);
    return {};
  }
}

/** Merge one tab's UI slice into ui_state.json (position / channel / lastActiveTab). */
export async function saveUiTabSlice(
  sample: string,
  tabId: 'alignment' | 'exosome' | 'imageProcessing',
  slice: { position?: string; channel?: string; lastActiveTab?: string },
): Promise<boolean> {
  if (!sample) return false;
  const tabSlice: Record<string, string> = {};
  if (slice.position !== undefined && slice.position !== '') tabSlice.position = slice.position;
  if (slice.channel !== undefined && slice.channel !== '') tabSlice.channel = slice.channel;
  if (slice.lastActiveTab !== undefined && slice.lastActiveTab !== '') {
    tabSlice.lastActiveTab = slice.lastActiveTab;
  }
  const uiPatch: Record<string, unknown> = { [tabId]: tabSlice };
  if (slice.lastActiveTab !== undefined && slice.lastActiveTab !== '') {
    uiPatch.lastActiveTab = slice.lastActiveTab;
  }
  return saveStateToDisk(sample, 'ui', uiPatch);
}

/** Merge key → JSON string entries into exosome_state.json (filtered detections, click history, etc.). */
export async function mergeExosomeStorageKeysOnDisk(
  sample: string,
  kv: Record<string, string>,
): Promise<boolean> {
  if (!sample || !Object.keys(kv).length) return false;
  const cur = await loadStateFromDisk(sample, true);
  const ex = (cur.exosome && typeof cur.exosome === 'object')
    ? (cur.exosome as { storage?: Record<string, string> })
    : {};
  const storageMap = { ...(ex.storage || {}), ...kv };
  return saveStateToDisk(sample, 'exosome', { storage: storageMap });
}
