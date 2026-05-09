import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/index.css';
import App from './App';
import { setBackendPort } from './lib/apiBase';

// Electron boot path: apply backend port before initial render if already known.
const bootPort = window.electronAPI?.getBackendPort?.();
if (typeof bootPort === 'number' && Number.isFinite(bootPort) && bootPort > 0) {
  setBackendPort(bootPort);
}

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);


