/**
 * Main App Component
 */

import React, { useState, useEffect, useRef } from 'react';
import { setBackendPort } from './lib/apiBase';
import AlignmentViewer from './components/AlignmentViewer';
import ImageProcessing from './components/ImageProcessing';
import Alignment from './components/Alignment';
import AgentChat from './components/AgentChat';
import ExosomeDetection, { type ExosomeDetectionImperativeHandle } from './components/ExosomeDetection';
import DataAnalysis from './components/DataAnalysis';
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

  const [activeTab, setActiveTab] = useState<
    'agent' | 'processing' | 'alignment' | 'exosome' | 'dataAnalysis' | 'viewer'
  >('agent');

  const exosomeDetectionRef = useRef<ExosomeDetectionImperativeHandle>(null);

  useEffect(() => {
    if (activeTab !== 'exosome') return;
    requestAnimationFrame(() => {
      console.log('[FIT TRIGGER]', { source: 'App.tsx:activeTab-exosome-rAF' });
      exosomeDetectionRef.current?.fitToViewport();
    });
  }, [activeTab]);

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
        <button
          className={`nav-tab ${activeTab === 'dataAnalysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('dataAnalysis')}
        >
          📉 Data Analysis
        </button>
      </nav>

      {/* All tabs are always mounted (hidden via CSS) to preserve state across tab switches */}

      <div
        className="tab-content"
        data-tab="agent"
        data-active={activeTab === 'agent' ? 'true' : 'false'}
      >
        <AgentChat />
      </div>

      <div
        className="tab-content"
        data-tab="processing"
        data-active={activeTab === 'processing' ? 'true' : 'false'}
      >
        <React.Suspense fallback={<div>Loading Image Processing...</div>}>
          <ImageProcessing isActive={activeTab === 'processing'} />
        </React.Suspense>
      </div>

      <div
        className="tab-content"
        data-tab="alignment"
        data-active={activeTab === 'alignment' ? 'true' : 'false'}
      >
        <React.Suspense fallback={<div>Loading Alignment...</div>}>
          <Alignment isActive={activeTab === 'alignment'} />
        </React.Suspense>
      </div>

      <div
        className="tab-content"
        data-tab="exosome"
        data-active={activeTab === 'exosome' ? 'true' : 'false'}
      >
        <React.Suspense fallback={<div>Loading Exosome Detection...</div>}>
          <ExosomeDetection ref={exosomeDetectionRef} isActive={activeTab === 'exosome'} />
        </React.Suspense>
      </div>

      <div
        className="tab-content"
        data-tab="viewer"
        data-active={activeTab === 'viewer' ? 'true' : 'false'}
      >
        <AlignmentViewer />
      </div>

      <div
        className="tab-content"
        data-tab="dataAnalysis"
        data-active={activeTab === 'dataAnalysis' ? 'true' : 'false'}
      >
        <DataAnalysis />
      </div>
    </div>
  );
};

export default App;

