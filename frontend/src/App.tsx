/**
 * Main App Component
 */

import React, { useState, useEffect } from 'react';
import { setBackendPort } from './lib/apiBase';
import AlignmentViewer from './components/AlignmentViewer';
import ImageProcessing from './components/ImageProcessing';
import Alignment from './components/Alignment';
import AgentChat from './components/AgentChat';
import ExosomeDetection from './components/ExosomeDetection';
import './styles/App.css';

const App: React.FC = () => {
  // In Electron, receive the backend port from the main process and update
  // the shared API base so all components use the correct absolute URL.
  // In a browser dev environment window.electronAPI is undefined, so this
  // effect does nothing and relative URLs / Vite proxy continue to work.
  useEffect(() => {
    window.electronAPI?.onBackendPort((port: number) => {
      setBackendPort(port);
    });
  }, []);

  const [sampleName, setSampleName] = useState<string>('1');
  const [useMockData, setUseMockData] = useState<boolean>(false);  // Default to real data
  const [inputValue, setInputValue] = useState<string>('1');
  const [activeTab, setActiveTab] = useState<'agent' | 'processing' | 'alignment' | 'exosome' | 'viewer'>('agent');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSampleName(inputValue);
  };

  return (
    <div className="app">
      {/* Tab Navigation */}
      <nav className="app-nav">
        <button
          className={`nav-tab ${activeTab === 'agent' ? 'active' : ''}`}
          onClick={() => setActiveTab('agent')}
        >
          🤖 Agent Chat
        </button>
        <button
          className={`nav-tab ${activeTab === 'processing' ? 'active' : ''}`}
          onClick={() => setActiveTab('processing')}
        >
          🖼️ Image Processing
        </button>
        <button
          className={`nav-tab ${activeTab === 'alignment' ? 'active' : ''}`}
          onClick={() => setActiveTab('alignment')}
        >
          🎯 Alignment
        </button>
        <button
          className={`nav-tab ${activeTab === 'exosome' ? 'active' : ''}`}
          onClick={() => setActiveTab('exosome')}
        >
          🔬 Exosome Detection
        </button>
        <button
          className={`nav-tab ${activeTab === 'viewer' ? 'active' : ''}`}
          onClick={() => setActiveTab('viewer')}
        >
          📊 Results Viewer
        </button>
      </nav>

      {/* All tabs are always mounted (hidden via CSS) to preserve state across tab switches */}

      <div className="tab-content" style={{ display: activeTab === 'agent' ? 'block' : 'none' }}>
        <AgentChat />
      </div>

      <div className="tab-content" style={{ display: activeTab === 'processing' ? 'block' : 'none' }}>
        <React.Suspense fallback={<div>Loading Image Processing...</div>}>
          <ImageProcessing />
        </React.Suspense>
      </div>

      <div className="tab-content" style={{ display: activeTab === 'alignment' ? 'block' : 'none' }}>
        <React.Suspense fallback={<div>Loading Alignment...</div>}>
          <Alignment isActive={activeTab === 'alignment'} />
        </React.Suspense>
      </div>

      <div className="tab-content" style={{ display: activeTab === 'exosome' ? 'block' : 'none' }}>
        <React.Suspense fallback={<div>Loading Exosome Detection...</div>}>
          <ExosomeDetection />
        </React.Suspense>
      </div>

      <div className="tab-content" style={{ display: activeTab === 'viewer' ? 'block' : 'none' }}>
        <div className="app-controls">
          <form onSubmit={handleSubmit} className="sample-selector">
            <label>
              Sample Name:
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter sample name (e.g., 1, 2, Sample_01)"
              />
            </label>
            <button type="submit">Load Sample</button>
          </form>

          <label className="mock-data-toggle">
            <input
              type="checkbox"
              checked={useMockData}
              onChange={(e) => setUseMockData(e.target.checked)}
            />
            Use Mock Data (enable for development without backend)
          </label>
        </div>

        <AlignmentViewer sampleName={sampleName} useMockData={useMockData} />
      </div>
    </div>
  );
};

export default App;

