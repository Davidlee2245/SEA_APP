/**
 * Step 3: Final Results Visualization
 * Displays final outputs from the alignment pipeline
 */

import React, { useState } from 'react';
import { ResultItem, ColocalizationStats, ComboAnalysis } from '../types/alignment';
import '../styles/ResultsVisualization.css';

interface ResultsVisualizationProps {
  results: ResultItem[];
  sampleName: string;
  colocalization?: ColocalizationStats;
  comboAnalysis?: ComboAnalysis;  // NEW: Multi-channel combination analysis
}

const ResultsVisualization: React.FC<ResultsVisualizationProps> = ({ 
  results, 
  sampleName,
  colocalization,
  comboAnalysis
}) => {
  const [selectedResult, setSelectedResult] = useState<ResultItem | null>(
    results.length > 0 ? results[0] : null
  );
  const [imageError, setImageError] = useState<boolean>(false);

  const getResultIcon = (type: string): string => {
    switch (type) {
      case 'overlay':
        return '🎨';
      case 'csv':
        return '📊';
      case 'plot':
        return '📈';
      case 'label':
        return '🏷️';
      case 'registered':
        return '✓';
      default:
        return '📄';
    }
  };

  const isImageResult = (type: string): boolean => {
    return ['overlay', 'plot', 'label', 'registered'].includes(type);
  };

  const handleDownload = (result: ResultItem) => {
    const link = document.createElement('a');
    link.href = result.url;
    link.download = result.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const groupedResults = results.reduce((acc, result) => {
    if (!acc[result.type]) {
      acc[result.type] = [];
    }
    acc[result.type].push(result);
    return acc;
  }, {} as Record<string, ResultItem[]>);

  return (
    <div className="results-visualization">
      <div className="view-header">
        <h3>Final Results</h3>
        <p className="description">
          View and download all output files generated from the aligned images. 
          Results include overlays, quantification data, and processed images.
        </p>
      </div>

      {/* Colocalization Statistics */}
      {colocalization && colocalization.totalDetections > 0 && (
        <div className="colocalization-stats">
          <h4>🔬 Colocalization Analysis</h4>
          <div className="coloc-summary">
            <div className="coloc-stat-card highlight">
              <div className="stat-value">{colocalization.totalColocalized}</div>
              <div className="stat-label">Colocalized Objects</div>
              <div className="stat-rate">{colocalization.colocalizationRate}% of total</div>
            </div>
            <div className="coloc-stat-card">
              <div className="stat-value">{colocalization.totalDetections}</div>
              <div className="stat-label">Total Detections</div>
            </div>
          </div>
          
          <div className="coloc-per-channel">
            <h5>Colocalization by Channel:</h5>
            <div className="channel-grid">
              {colocalization.channels.map((channel, idx) => (
                <div key={idx} className="channel-card">
                  <div className="channel-name">{channel.channel}</div>
                  <div className="channel-stats">
                    <div className="stat-row">
                      <span className="label">Total Detections:</span>
                      <span className="value">{channel.totalDetections}</span>
                    </div>
                    <div className="stat-row">
                      <span className="label">Colocalized:</span>
                      <span className="value coloc-count">{channel.colocalizedCount}</span>
                    </div>
                    <div className="stat-row">
                      <span className="label">Rate:</span>
                      <span className="value coloc-rate">{channel.colocalizedPercent}%</span>
                    </div>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${channel.colocalizedPercent}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Multi-Channel Combination Analysis */}
      {comboAnalysis && comboAnalysis.combo_table && comboAnalysis.combo_table.length > 0 && (
        <div className="combo-analysis">
          <h4>🧬 Multi-Channel Combination Analysis</h4>
          
          <div className="combo-table-section">
            <h5>Channel Combinations:</h5>
            <div className="combo-table-container">
              <table className="combo-table">
                <thead>
                  <tr>
                    <th>Combination</th>
                    <th>Count</th>
                    <th>Rate / Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {comboAnalysis.combo_table
                    .sort((a, b) => {
                      // Sort by: 1) number of channels (desc), 2) count (desc)
                      if (b.combo.length !== a.combo.length) {
                        return b.combo.length - a.combo.length;
                      }
                      return b.count - a.count;
                    })
                    .map((entry, idx) => {
                      // Calculate percentage from rate
                      const percentage = (entry.rate * 100);
                      // Format combination label (already normalized from backend)
                      const comboLabel = entry.combo.join(' + ');
                      
                      return (
                        <tr key={idx} className="combo-row">
                          <td className="combo-cell">
                            <span className="combo-badge">
                              {comboLabel}
                            </span>
                          </td>
                          <td className="count-cell">{entry.count}</td>
                          <td className="percentage-cell">
                            <div className="percentage-bar-container">
                              <div className="percentage-bar">
                                <div 
                                  className="percentage-fill" 
                                  style={{ width: `${percentage}%` }}
                                ></div>
                              </div>
                              <span className="percentage-text">
                                {entry.rate.toFixed(4)} ({percentage.toFixed(1)}%)
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <div className="results-layout">
        <aside className="results-sidebar">
          <h4>Output Files</h4>
          {Object.entries(groupedResults).map(([type, items]) => (
            <div key={type} className="result-group">
              <h5 className="result-type-header">
                {getResultIcon(type)} {type.toUpperCase()}
              </h5>
              <ul className="result-list">
                {items.map((result, index) => (
                  <li
                    key={`${type}-${index}`}
                    className={`result-item ${
                      selectedResult?.url === result.url ? 'selected' : ''
                    }`}
                    onClick={() => {
                      setSelectedResult(result);
                      setImageError(false);
                    }}
                  >
                    <span className="result-name">{result.name}</span>
                    <button
                      className="download-icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDownload(result);
                      }}
                      title="Download"
                    >
                      ⬇️
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </aside>

        <main className="results-content">
          {selectedResult ? (
            <div className="result-viewer">
              <div className="result-header">
                <h4>
                  {getResultIcon(selectedResult.type)} {selectedResult.name}
                </h4>
                <button
                  className="download-button"
                  onClick={() => handleDownload(selectedResult)}
                >
                  Download
                </button>
              </div>

              {selectedResult.description && (
                <p className="result-description">{selectedResult.description}</p>
              )}

              <div className="result-display">
                {isImageResult(selectedResult.type) ? (
                  imageError ? (
                    <div className="result-placeholder error">
                      <p>Image not available</p>
                      <code>{selectedResult.url}</code>
                    </div>
                  ) : (
                    <img
                      src={selectedResult.url}
                      alt={selectedResult.name}
                      onError={() => setImageError(true)}
                    />
                  )
                ) : selectedResult.type === 'csv' ? (
                  <div className="csv-preview">
                    <p>CSV data file - click Download to view full contents</p>
                    <code>{selectedResult.url}</code>
                  </div>
                ) : (
                  <div className="generic-file">
                    <p>File: {selectedResult.name}</p>
                    <code>{selectedResult.url}</code>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="result-viewer empty">
              <p>No results available</p>
            </div>
          )}
        </main>
      </div>

      <div className="results-summary">
        <h4>Summary</h4>
        <div className="summary-grid">
          <div className="summary-item">
            <strong>Sample:</strong> {sampleName}
          </div>
          <div className="summary-item">
            <strong>Total Files:</strong> {results.length}
          </div>
          <div className="summary-item">
            <strong>Output Types:</strong> {Object.keys(groupedResults).length}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsVisualization;

