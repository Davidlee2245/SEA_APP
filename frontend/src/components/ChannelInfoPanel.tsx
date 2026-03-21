/**
 * Channel Info Panel - Intensity Histograms and Statistics
 * Computes histogram and stats from preview PNG images
 */

import React, { useState, useEffect, useRef } from 'react';
import '../styles/ChannelInfoPanel.css';
import { getApiBase } from '../lib/apiBase';

interface ChannelStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  p1: number;
  p99: number;
  histogram: number[];
  backend_accurate?: boolean; // Flag to indicate if these are 16-bit backend stats
}

interface ChannelItem {
  key: string;
  cycle: string;
  channel: string;
  marker: string | null;
  display_label: string;
  tiff_path: string;
  preview_url: string | null;
}

interface ChannelInfoPanelProps {
  stages: {
    raw?: Record<string, string>;
    processed?: Record<string, string>;  // Final processed output
    // Legacy support for intermediate stages
    contrast_enhance?: Record<string, string>;
    aligned?: Record<string, string>;
    step1?: Record<string, string>;
    step2?: Record<string, string>;
    step3?: Record<string, string>;
    step4?: Record<string, string>;
  };
  currentStage: 'raw' | 'processed' | 'contrast_enhance' | 'aligned' | 'step1' | 'step2' | 'step3' | 'step4';
  availableItems: ChannelItem[];
  isLoaded: boolean;
  selectedSample: string;
  selectedPosition: string;
  backendStats?: Record<string, any>; // Optional 16-bit stats from backend
}

// Cache for computed histograms
const histogramCache = new Map<string, ChannelStats>();

const computeHistogramFromPng = async (url: string): Promise<ChannelStats> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      try {
        // Create offscreen canvas
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }

        // Draw image
        ctx.drawImage(img, 0, 0);

        // Get pixel data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Initialize histogram bins (256 for 8-bit)
        const histogram = new Array(256).fill(0);
        const intensities: number[] = [];

        // Count intensities (grayscale, so R = G = B)
        for (let i = 0; i < data.length; i += 4) {
          const intensity = data[i]; // Red channel (same as G and B for grayscale)
          histogram[intensity]++;
          intensities.push(intensity);
        }

        // Sort for percentile calculations
        intensities.sort((a, b) => a - b);

        // Compute statistics
        const min = intensities[0];
        const max = intensities[intensities.length - 1];
        const mean = intensities.reduce((sum, val) => sum + val, 0) / intensities.length;
        const median = intensities[Math.floor(intensities.length / 2)];
        const p1 = intensities[Math.floor(intensities.length * 0.01)];
        const p99 = intensities[Math.floor(intensities.length * 0.99)];

        resolve({
          min,
          max,
          mean,
          median,
          p1,
          p99,
          histogram,
        });
      } catch (error) {
        reject(error);
      }
    };

    img.onerror = () => reject(new Error('Failed to load image'));

    // Ensure URL is absolute (add backend server if relative)
    const absoluteUrl = url.startsWith('http') ? url : `${getApiBase()}${url}`;
    // Add cache-busting param (use & if URL already has query params, otherwise ?)
    const separator = absoluteUrl.includes('?') ? '&' : '?';
    img.src = `${absoluteUrl}${separator}t=${Date.now()}`;
  });
};

const ChannelInfoPanel: React.FC<ChannelInfoPanelProps> = ({
  stages,
  currentStage,
  availableItems,
  isLoaded,
  selectedSample,
  selectedPosition,
  backendStats,
}) => {
  const [channelStats, setChannelStats] = useState<Record<string, ChannelStats>>({});
  const [isComputing, setIsComputing] = useState(false);
  const computeCountRef = useRef(0);

  // Compute histograms when stage, channels, or preview URLs change
  useEffect(() => {
    if (!isLoaded) return;
    
    // Get the current stage data (handle both old and new stage structure)
    const stageData = stages[currentStage];
    if (!stageData) {
      console.log(`[ChannelInfo] No data for stage: ${currentStage}`);
      return;
    }

    const computeAllHistograms = async () => {
      const currentCount = ++computeCountRef.current;
      setIsComputing(true);

      try {
        console.log(`[ChannelInfo] Updating statistics for stage: ${currentStage}`);
        const newStats: Record<string, ChannelStats> = {};

        for (const item of availableItems) {
          // If we have accurate backend stats for this channel AND stage, use them!
          const hasBackendStats = backendStats && backendStats[item.key] && 
                                (currentStage === 'processed' || currentStage === 'aligned' || currentStage === 'raw');

          const url = stageData[item.key];
          if (!url) continue;

          // Generate cache key
          const cacheKey = `${selectedSample}/${selectedPosition}/${currentStage}/${item.key}/${url}`;

          if (hasBackendStats) {
            console.log(`[ChannelInfo] Using accurate 16-bit backend stats for ${item.key}`);
            const bStats = backendStats![item.key];
            
            // SCALE 16-BIT TO 8-BIT for display consistency (Requested: 25.8 vs 6899.4)
            // If the values are in 16-bit range (max > 255), scale them down to 0-255
            const scaleFactor = (bStats.max > 255 || bStats.mean > 255) ? 255.0 / 65535.0 : 1.0;
            
            // We still need a histogram for visualization. 
            // If the cache doesn't have it, we compute it from the 8-bit preview.
            let histogram: number[] = new Array(256).fill(0);
            if (histogramCache.has(cacheKey)) {
              histogram = histogramCache.get(cacheKey)!.histogram;
            } else {
              try {
                const previewStats = await computeHistogramFromPng(url);
                histogram = previewStats.histogram;
                // Cache it so we don't re-compute
                histogramCache.set(cacheKey, { ...previewStats });
              } catch (e) {
                console.warn(`[ChannelInfo] Could not compute preview histogram for ${item.key}`, e);
              }
            }

            newStats[item.key] = {
              min: bStats.min * scaleFactor,
              max: bStats.max * scaleFactor,
              mean: bStats.mean * scaleFactor,
              median: bStats.median * scaleFactor,
              p1: bStats.p1 * scaleFactor,
              p99: bStats.p99 * scaleFactor,
              histogram: histogram,
              backend_accurate: true
            };
            continue;
          }

          // Fallback to computing from PNG
          if (histogramCache.has(cacheKey)) {
            newStats[item.key] = histogramCache.get(cacheKey)!;
            continue;
          }

          try {
            const stats = await computeHistogramFromPng(url);
            histogramCache.set(cacheKey, stats);
            newStats[item.key] = stats;
          } catch (error) {
            console.error(`[ChannelInfo] Failed to compute histogram for ${item.key}:`, error);
          }
        }

        if (currentCount === computeCountRef.current) {
          setChannelStats(newStats);
        }
      } finally {
        if (currentCount === computeCountRef.current) {
          setIsComputing(false);
        }
      }
    };

    computeAllHistograms();
  }, [stages, currentStage, availableItems, isLoaded, selectedSample, selectedPosition, backendStats]);

  const renderHistogram = (histogram: number[], channel: string, mean: number, isAccurate: boolean, min: number, max: number) => {
    const maxCount = Math.max(...histogram);
    const barCount = 64; // Bin 256 values into 64 bars for display
    const binSize = 256 / barCount;

    // Aggregate bins
    const displayBins: number[] = [];
    for (let i = 0; i < barCount; i++) {
      let sum = 0;
      for (let j = 0; j < binSize; j++) {
        sum += histogram[Math.floor(i * binSize + j)] || 0;
      }
      displayBins.push(sum);
    }

    const maxDisplayCount = Math.max(...displayBins);
    
    // Calculate x position for mean line
    // If accurate (16-bit), we need to map the mean from [min, max] to [0, 255] for display
    let meanX: number;
    if (isAccurate && max > min) {
      const normalizedMean = ((mean - min) / (max - min)) * 255;
      meanX = (Math.max(0, Math.min(255, normalizedMean)) / 255) * 256;
    } else {
      meanX = (mean / 255) * 256;
    }

    return (
      <svg className="histogram-svg" viewBox="0 0 256 80" preserveAspectRatio="none">
        {displayBins.map((count, index) => {
          const height = maxDisplayCount > 0 ? (count / maxDisplayCount) * 70 : 0;
          const x = (index / barCount) * 256;
          const barWidth = 256 / barCount;

          return (
            <rect
              key={index}
              x={x}
              y={80 - height}
              width={barWidth}
              height={height}
              fill={isAccurate ? "#2ecc71" : "#3498db"}
              opacity={0.8}
            />
          );
        })}
        {/* Baseline */}
        <line x1="0" y1="80" x2="256" y2="80" stroke="#ccc" strokeWidth="0.5" />
        
        {/* Mean line (red dashed) */}
        <line 
          x1={meanX} 
          y1="5" 
          x2={meanX} 
          y2="80" 
          stroke="#e74c3c" 
          strokeWidth="1.5" 
          strokeDasharray="3,3"
          opacity="0.9"
        />
        {/* Mean dot at top */}
        <circle 
          cx={meanX} 
          cy="5" 
          r="2.5" 
          fill="#e74c3c"
          opacity="0.9"
        />
      </svg>
    );
  };

  if (!isLoaded) {
    return (
      <div className="channel-info-panel">
        <div className="panel-header">
          <h3>📊 Channel Info</h3>
        </div>
        <div className="panel-placeholder">
          <p>Load a position to view channel statistics</p>
        </div>
      </div>
    );
  }

  return (
    <div className="channel-info-panel">
      <div className="panel-header">
        <h3>📊 Channel Info</h3>
        <div className="stage-indicator">Stage: {currentStage}</div>
      </div>

      {isComputing && (
        <div className="computing-indicator">Updating statistics...</div>
      )}

      <div className="channel-cards">
        {availableItems.map((item) => {
          const stats = channelStats[item.key];

          if (!stats) {
            return (
              <div key={item.key} className="channel-card">
                <div className="channel-card-header">{item.display_label}</div>
                <div className="channel-card-body">
                  <p className="missing-channel">Loading...</p>
                </div>
              </div>
            );
          }

          return (
            <div key={item.key} className="channel-card">
              <div className="channel-card-header">
                {item.display_label}
                {stats.backend_accurate && (
                  <span className="accurate-badge" title="16-bit Backend Accurate Stats" style={{ marginLeft: '8px', color: '#2ecc71', fontSize: '0.8rem' }}>✓</span>
                )}
              </div>
              <div className="channel-card-body">
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Min:</span>
                    <span className="stat-value">{stats.min.toFixed(0)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Max:</span>
                    <span className="stat-value">{stats.max.toFixed(0)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Mean:</span>
                    <span className="stat-value">{stats.mean.toFixed(1)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Median:</span>
                    <span className="stat-value">{stats.median.toFixed(0)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">P1:</span>
                    <span className="stat-value">{stats.p1.toFixed(0)}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">P99:</span>
                    <span className="stat-value">{stats.p99.toFixed(0)}</span>
                  </div>
                </div>

                <div className="histogram-container">
                  <div className="histogram-label">
                    Intensity Distribution {stats.backend_accurate ? '(16-bit scaled)' : '(8-bit)'}
                    <span style={{ marginLeft: '8px', color: '#e74c3c', fontSize: '8px' }}>
                      ⬤ Mean
                    </span>
                  </div>
                  {renderHistogram(stats.histogram, item.key, stats.mean, !!stats.backend_accurate, stats.min, stats.max)}
                  <div className="histogram-axis">
                    <span>{stats.backend_accurate ? stats.min.toFixed(0) : '0'}</span>
                    <span>{stats.backend_accurate ? ((stats.min + stats.max) / 2).toFixed(0) : '128'}</span>
                    <span>{stats.backend_accurate ? stats.max.toFixed(0) : '255'}</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ChannelInfoPanel;

