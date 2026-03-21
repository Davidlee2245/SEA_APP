/**
 * Main Alignment Viewer Component
 * Orchestrates the 3-step visualization flow
 */

import React, { useState, useEffect } from 'react';
import { AlignmentResult } from '../types/alignment';
import BeforeAfterView from './BeforeAfterView';
import MovementQuantification from './MovementQuantification';
import ResultsVisualization from './ResultsVisualization';
import '../styles/AlignmentViewer.css';
import { getApiBase } from '../lib/apiBase';

interface AlignmentViewerProps {
  sampleName: string;
  useMockData?: boolean;
}

const AlignmentViewer: React.FC<AlignmentViewerProps> = ({ 
  sampleName, 
  useMockData = false 
}) => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [alignmentData, setAlignmentData] = useState<AlignmentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAlignmentData();
  }, [sampleName, useMockData]);

  const loadAlignmentData = async () => {
    setLoading(true);
    setError(null);

    try {
      let data: AlignmentResult;
      
      if (useMockData) {
        const { getMockAlignmentData } = await import('../mocks/mockData');
        data = getMockAlignmentData(sampleName);
      } else {
        const response = await fetch(`${getApiBase()}/api/alignment/${sampleName}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch alignment data: ${response.statusText}`);
        }
        const result = await response.json();
        data = result.data;
      }

      setAlignmentData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleStepChange = (step: number) => {
    setCurrentStep(step);
  };

  if (loading) {
    return (
      <div className="alignment-viewer loading">
        <div className="spinner"></div>
        <p>Loading alignment data for {sampleName}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alignment-viewer error">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <button onClick={loadAlignmentData}>Retry</button>
      </div>
    );
  }

  if (!alignmentData) {
    return (
      <div className="alignment-viewer error">
        <p>No alignment data available</p>
      </div>
    );
  }

  return (
    <div className="alignment-viewer">
      <header className="viewer-header">
        <h1>SEA Alignment Visualization</h1>
        <h2>Sample: {alignmentData.sampleName}</h2>
        {alignmentData.metadata && (
          <div className="metadata">
            <span>Anchor: {alignmentData.metadata.anchorChannel}</span>
            <span>Channels: {alignmentData.metadata.totalChannels}</span>
          </div>
        )}
      </header>

      <nav className="step-navigation">
        {[1, 2, 3].map((step) => (
          <button
            key={step}
            className={`step-button ${currentStep === step ? 'active' : ''} ${
              currentStep > step ? 'completed' : ''
            }`}
            onClick={() => handleStepChange(step)}
          >
            <span className="step-number">{step}</span>
            <span className="step-label">
              {step === 1 && 'Before / After'}
              {step === 2 && 'Movement Analysis'}
              {step === 3 && 'Final Results'}
            </span>
          </button>
        ))}
      </nav>

      <main className="step-content">
        {currentStep === 1 && (
          <BeforeAfterView frames={alignmentData.frames} />
        )}
        {currentStep === 2 && (
          <MovementQuantification frames={alignmentData.frames} />
        )}
        {currentStep === 3 && (
          <ResultsVisualization 
            results={alignmentData.finalResults}
            sampleName={alignmentData.sampleName}
            colocalization={alignmentData.colocalization}
            comboAnalysis={alignmentData.comboAnalysis}
          />
        )}
      </main>

      <footer className="viewer-footer">
        <button
          onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
          disabled={currentStep === 1}
        >
          Previous
        </button>
        <button
          onClick={() => setCurrentStep(Math.min(3, currentStep + 1))}
          disabled={currentStep === 3}
        >
          Next
        </button>
      </footer>
    </div>
  );
};

export default AlignmentViewer;

