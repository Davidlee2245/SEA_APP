const isElectron = (): boolean => window.electronAPI !== undefined;

export async function get(key: string): Promise<string | null> {
  try {
    if (isElectron() && window.electronAPI?.storeGet) {
      const value = await window.electronAPI.storeGet(key);
      if (value !== null) return value;
    }
    return localStorage.getItem(key);
  } catch {
    return localStorage.getItem(key);
  }
}

export async function set(key: string, value: string): Promise<void> {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Ignore localStorage write failures.
  }

  if (isElectron() && window.electronAPI?.storeSet) {
    try {
      await window.electronAPI.storeSet(key, value);
    } catch {
      // Never throw to keep caller flow stable.
    }
  }
}

export async function remove(key: string): Promise<void> {
  try {
    localStorage.removeItem(key);
  } catch {
    // Ignore localStorage remove failures.
  }

  if (isElectron() && window.electronAPI?.storeRemove) {
    try {
      await window.electronAPI.storeRemove(key);
    } catch {
      // Never throw to keep caller flow stable.
    }
  }
}
