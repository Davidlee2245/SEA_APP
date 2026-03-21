/**
 * Exosome Detection Component
 * Uses SAM (Segment Anything Model) for exosome segmentation
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import '../styles/ExosomeDetection.css';
import { getApiBase } from '../lib/apiBase';

interface ChannelItem {
  key: string;
  display_label: string;
  preview_url?: string;
}

interface DetectionResult {
  id: number;
  area: number;
  centroid: [number, number];
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score?: number;
}

interface ExosomeDetectionState {
  selectedSample: string;
  selectedPosition: string;
  selectedChannel: string;
  availableSamples: string[];
  availablePositions: string[];
  availableItems: ChannelItem[];
  loaded: boolean;
  currentImageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  
  // Detection method
  detectionMethod: 'sam' | 'blob' | 'random_forest';
  
  // Detection settings
  modelType: string;
  checkpointPath: string;
  device: string;
  confidenceThreshold: number;
  minArea: number;
  maxArea: number;
  removeSmallObjects: boolean;
  fillHoles: boolean;
  
  // Detection mode (for SAM only)
  detectionMode: 'box' | 'point' | 'auto';
  
  // Blob detection parameters
  blobThreshold: number; // Threshold for blob detection
  blobMinCircularity: number; // Minimum circularity (0-1)
  blobMaxCircularity: number; // Maximum circularity (0-1)
  blobMinInertiaRatio: number; // Minimum inertia ratio (0-1)
  
  // Prompts
  boxPrompt: [number, number, number, number] | null; // [x1, y1, x2, y2]
  pointPrompts: Array<{ x: number; y: number; label: number }>; // label: 1=positive, 0=negative
  
  // Results
  masks: boolean[][][] | null; // (N, H, W) boolean array
  scores: number[] | null;
  detections: DetectionResult[];
  isDetecting: boolean;
  
  // Display settings
  maskOpacity: number;
  showMaskOutlines: boolean;
  showConfidenceMap: boolean; // For Random Forest
  
  // Random Forest annotations
  annotations: Array<{ 
    id: string; // Unique ID for removal
    points: Array<[number, number]>, 
    label: number 
  }>; // label: 1=exosome, 0=background
  brushSize: number;
  annotationMode: 'exosome' | 'background'; // Current annotation mode
  
  // Pixel Inspector
  pixelInspector: {
    x: number | null;
    y: number | null;
    intensity: number | null;
    confidence: number | null;
    predictedClass: string | null;
  };
  
  // Click History
  clickHistory: Array<{
    id: string; // Unique ID for removal
    timestamp: number;
    imageX: number;
    imageY: number;
    intensity: number | null;
    confidence: number | null;
    predictedClass: string | null;
    scale: number;
    offsetX: number;
    offsetY: number;
    annotationId?: string; // Link to annotation group for Random Forest
  }>;
  
  // Brush preview (for Random Forest annotation)
  brushPreview: {
    x: number | null;
    y: number | null;
  };
  
  // Debug logging
  debugLogging: boolean;
  debugLogs: Array<{
    timestamp: number;
    type: 'click' | 'annotation' | 'zoom' | 'pan';
    data: any;
  }>;
  
  // Calibration mode
  calibrationMode: boolean;
  
  // Random Forest results
  probabilityMap: number[][] | null;
  
  // Export
  exportStatus: string | null;
  exportPaths: string[];
}

// Helper component for slider + numeric input pair
interface SliderInputProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  isInteger?: boolean;
  precision?: number; // Number of decimal places to display
}

const SliderInput: React.FC<SliderInputProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  isInteger = false,
  precision = 2,
}) => {
  const [inputValue, setInputValue] = useState<string>(value.toString());

  // Update input value when prop value changes (from slider)
  useEffect(() => {
    setInputValue(value.toString());
  }, [value]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    onChange(newValue);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputVal = e.target.value;
    setInputValue(inputVal);

    // Allow empty input while typing
    if (inputVal === '' || inputVal === '-') {
      return;
    }

    const numValue = isInteger ? parseInt(inputVal, 10) : parseFloat(inputVal);
    
    if (!isNaN(numValue)) {
      // Clamp value to min/max
      const clampedValue = Math.max(min, Math.min(max, numValue));
      onChange(clampedValue);
    }
  };

  const handleInputBlur = () => {
    // On blur, ensure input is valid and clamped
    const numValue = isInteger ? parseInt(inputValue, 10) : parseFloat(inputValue);
    if (isNaN(numValue)) {
      setInputValue(value.toString());
    } else {
      const clampedValue = Math.max(min, Math.min(max, numValue));
      setInputValue(clampedValue.toString());
      onChange(clampedValue);
    }
  };

  const displayValue = isInteger ? Math.round(value) : value.toFixed(precision);

  return (
    <div className="input-group">
      <label>{label}: {displayValue}</label>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleSliderChange}
          style={{ flex: 1 }}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          style={{
            width: '80px',
            padding: '0.25rem 0.5rem',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '0.9rem',
          }}
        />
      </div>
    </div>
  );
};

// Area Distribution Histogram with auto log-scale for skewed distributions
const AreaHistogram: React.FC<{ detections: DetectionResult[] }> = ({ detections }) => {
  const histCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || detections.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);

    const areas = detections.map(d => d.area);
    const minArea = Math.min(...areas);
    const maxArea = Math.max(...areas);
    const meanArea = areas.reduce((s, a) => s + a, 0) / areas.length;
    const sortedAreas = [...areas].sort((a, b) => a - b);
    const medianArea = sortedAreas.length % 2 === 0
      ? (sortedAreas[sortedAreas.length / 2 - 1] + sortedAreas[sortedAreas.length / 2]) / 2
      : sortedAreas[Math.floor(sortedAreas.length / 2)];

    // Decide whether to use log scale: if max/median ratio > 10, distribution is skewed
    const useLog = minArea > 0 && maxArea / Math.max(medianArea, 1) > 10;

    const numBins = Math.min(30, Math.max(8, Math.ceil(Math.sqrt(areas.length))));

    // Build bin edges
    let binEdges: number[];
    if (useLog) {
      const logMin = Math.log10(Math.max(minArea, 0.5));
      const logMax = Math.log10(maxArea);
      const logStep = (logMax - logMin) / numBins;
      binEdges = Array.from({ length: numBins + 1 }, (_, i) => Math.pow(10, logMin + i * logStep));
    } else {
      const step = (maxArea - minArea || 1) / numBins;
      binEdges = Array.from({ length: numBins + 1 }, (_, i) => minArea + i * step);
    }

    // Count per bin
    const bins = new Array(numBins).fill(0);
    areas.forEach(a => {
      for (let i = 0; i < numBins; i++) {
        if (a >= binEdges[i] && (i === numBins - 1 || a < binEdges[i + 1])) {
          bins[i]++;
          break;
        }
      }
    });

    const maxCount = Math.max(...bins);

    // Layout
    const pad = { top: 20, right: 20, bottom: 48, left: 50 };
    const plotW = cssW - pad.left - pad.right;
    const plotH = cssH - pad.top - pad.bottom;

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, cssW, cssH);

    // Helper: map an area value to x pixel position
    const areaToX = (val: number): number => {
      if (useLog) {
        const logMin = Math.log10(binEdges[0]);
        const logMax = Math.log10(binEdges[numBins]);
        return pad.left + ((Math.log10(Math.max(val, binEdges[0])) - logMin) / (logMax - logMin)) * plotW;
      }
      return pad.left + ((val - binEdges[0]) / (binEdges[numBins] - binEdges[0])) * plotW;
    };

    // Grid lines
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 1;
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.top + plotH - (i / yTicks) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Bars (variable width for log scale)
    bins.forEach((count, i) => {
      const x1 = areaToX(binEdges[i]);
      const x2 = areaToX(binEdges[i + 1]);
      const barW = x2 - x1;
      const barH = maxCount > 0 ? (count / maxCount) * plotH : 0;
      const y = pad.top + plotH - barH;

      const grad = ctx.createLinearGradient(x1, y, x1, pad.top + plotH);
      grad.addColorStop(0, '#3498db');
      grad.addColorStop(1, '#2471a3');
      ctx.fillStyle = grad;
      ctx.fillRect(x1 + 0.5, y, Math.max(barW - 1, 1), barH);

      if (count > 0 && barH > 14 && barW > 18) {
        ctx.fillStyle = '#fff';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(count), x1 + barW / 2, y + 12);
      } else if (count > 0 && barW > 12) {
        ctx.fillStyle = '#555';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(count), x1 + barW / 2, y - 3);
      }
    });

    // Mean / Median dashed lines
    const drawStatLine = (value: number, color: string, label: string, yOff: number) => {
      const xPos = areaToX(value);
      if (xPos >= pad.left && xPos <= pad.left + plotW) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(xPos, pad.top);
        ctx.lineTo(xPos, pad.top + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = color;
        ctx.font = 'bold 10px sans-serif';
        const labelText = `${label}: ${value.toFixed(1)}`;
        // Flip label to left side if too close to right edge
        const textW = ctx.measureText(labelText).width;
        if (xPos + textW + 8 > pad.left + plotW) {
          ctx.textAlign = 'right';
          ctx.fillText(labelText, xPos - 4, pad.top + yOff);
        } else {
          ctx.textAlign = 'left';
          ctx.fillText(labelText, xPos + 4, pad.top + yOff);
        }
      }
    };

    drawStatLine(meanArea, '#e74c3c', 'Mean', 12);
    drawStatLine(medianArea, '#27ae60', 'Median', 24);

    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#555';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= yTicks; i++) {
      const val = Math.round((i / yTicks) * maxCount);
      const y = pad.top + plotH - (i / yTicks) * plotH;
      ctx.fillText(String(val), pad.left - 5, y + 3);
    }

    // X-axis labels: show bin edge values
    ctx.textAlign = 'center';
    ctx.fillStyle = '#555';
    ctx.font = '10px sans-serif';
    const maxLabels = 7;
    const labelStep = Math.max(1, Math.ceil(numBins / maxLabels));
    for (let i = 0; i <= numBins; i += labelStep) {
      const val = binEdges[i];
      const x = areaToX(val);
      const label = val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toFixed(0);
      ctx.fillText(label, x, pad.top + plotH + 14);
    }

    // Axis titles
    ctx.fillStyle = '#333';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(useLog ? 'Area (px², log scale)' : 'Area (px²)', pad.left + plotW / 2, cssH - 4);

    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Count', 0, 0);
    ctx.restore();

    // Scale mode indicator
    if (useLog) {
      ctx.fillStyle = '#888';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('log₁₀ scale', cssW - pad.right, pad.top - 5);
    }
  }, [detections]);

  if (detections.length === 0) return null;

  return (
    <div className="histogram-container">
      <h3>Area Distribution</h3>
      <canvas
        ref={histCanvasRef}
        style={{ width: '100%', height: '220px', borderRadius: '4px' }}
      />
    </div>
  );
};

const ExosomeDetection: React.FC = () => {
  const [state, setState] = useState<ExosomeDetectionState>({
    selectedSample: '',
    selectedPosition: '',
    selectedChannel: '',
    availableSamples: [],
    availablePositions: [],
    availableItems: [],
    loaded: false,
    currentImageUrl: null,
    imageWidth: 0,
    imageHeight: 0,
    
    detectionMethod: 'sam',
    
    // Random Forest settings
    annotations: [],
    brushSize: 5,
    annotationMode: 'exosome',
    showConfidenceMap: false,
    probabilityMap: null,
    pixelInspector: {
      x: null,
      y: null,
      intensity: null,
      confidence: null,
      predictedClass: null,
    },
    
    modelType: 'sam_vit_h',
    checkpointPath: 'checkpoints/pretrained/sam_vit_h_4b8939.pth',
    device: 'auto',
    confidenceThreshold: 0.5,
    minArea: 10,
    maxArea: 10000,
    removeSmallObjects: true,
    fillHoles: false,
    
    detectionMode: 'box',
    
    blobThreshold: 0.5,
    blobMinCircularity: 0.3,
    blobMaxCircularity: 1.0,
    blobMinInertiaRatio: 0.3,
    
    boxPrompt: null,
    pointPrompts: [],
    
    masks: null,
    scores: null,
    detections: [],
    isDetecting: false,
    
    maskOpacity: 0.5,
    showMaskOutlines: true,
    
    // Click History
    clickHistory: [],
    brushPreview: { x: null, y: null },
    
    // Debug logging
    debugLogging: false,
    debugLogs: [],
    calibrationMode: false,
    
    exportStatus: null,
    exportPaths: [],
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const isDrawingRef = useRef<boolean>(false);
  const startPosRef = useRef<{ x: number; y: number } | null>(null);
  const selectedDetectionRef = useRef<number | null>(null);
  const isAnnotatingRef = useRef<boolean>(false);
  const currentAnnotationPointsRef = useRef<Array<[number, number]>>([]);
  const currentAnnotationIdRef = useRef<string | null>(null);
  
  // Zoom and pan state
  const [zoomState, setZoomState] = useState({
    scale: 1.0,
    offsetX: 0,
    offsetY: 0,
  });
  const isPanningRef = useRef<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number } | null>(null);
  
  // Viewport and content refs (single source of truth)
  const viewportRef = useRef<HTMLDivElement>(null); // Captures pointer events
  const contentRef = useRef<HTMLDivElement>(null); // Gets transform (translate + scale)

  // Fetch available samples on mount
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await fetch('${getApiBase()}/api/input/samples');
        const data = await response.json();
        if (data.success) {
          setState(prev => ({ ...prev, availableSamples: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch samples:', err);
      }
    };
    fetchSamples();
  }, []);

  // Fetch positions when sample changes
  useEffect(() => {
    if (!state.selectedSample) {
      setState(prev => ({
        ...prev,
        availablePositions: [],
        selectedPosition: '',
        loaded: false,
        availableItems: [],
        currentImageUrl: null,
      }));
      return;
    }

    const fetchPositions = async () => {
      try {
        const response = await fetch(
          `${getApiBase()}/api/input/samples/${state.selectedSample}/positions`
        );
        const data = await response.json();
        if (data.success) {
          setState(prev => ({ ...prev, availablePositions: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch positions:', err);
      }
    };
    fetchPositions();
  }, [state.selectedSample]);

  // Load position and get channel info
  const handleLoadPosition = async () => {
    if (!state.selectedSample || !state.selectedPosition) {
      alert('Please select both sample and position');
      return;
    }

    try {
      const response = await fetch('${getApiBase()}/api/input/load_position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: state.selectedSample,
          position: state.selectedPosition,
        }),
      });

      const data = await response.json();
      if (!data.success) {
        alert(data.error || 'Failed to load position');
        return;
      }

      const items: ChannelItem[] = data.data.items || [];
      const firstItem = items[0];
      
      setState(prev => ({
        ...prev,
        availableItems: items,
        selectedChannel: firstItem?.key || '',
        loaded: items.length > 0,
        currentImageUrl: firstItem?.preview_url || null,
        boxPrompt: null,
        pointPrompts: [],
        masks: null,
        scores: null,
        detections: [],
        clickHistory: [], // Reset click history when loading new image
        debugLogs: [], // Reset debug logs when loading new image
      }));

      // Load image to get dimensions
      if (firstItem?.preview_url) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          setState(prev => ({
            ...prev,
            imageWidth: img.width,
            imageHeight: img.height,
          }));
          imageRef.current = img;
          drawCanvas();
        };
        const absoluteUrl = firstItem.preview_url.startsWith('http')
          ? firstItem.preview_url
          : `${getApiBase()}${firstItem.preview_url}`;
        img.src = absoluteUrl;
      }
    } catch (err) {
      console.error('Failed to load position:', err);
      alert('Failed to load position: ' + err);
    }
  };

  // Update image when channel changes
  useEffect(() => {
    if (!state.loaded || !state.selectedChannel) return;

    const item = state.availableItems.find(i => i.key === state.selectedChannel);
    if (item?.preview_url) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        setState(prev => ({
          ...prev,
          currentImageUrl: item.preview_url || null,
          imageWidth: img.width,
          imageHeight: img.height,
        }));
        imageRef.current = img;
        drawCanvas();
      };
      const absoluteUrl = item.preview_url.startsWith('http')
        ? item.preview_url
        : `${getApiBase()}${item.preview_url}`;
      img.src = absoluteUrl;
    }
  }, [state.selectedChannel, state.loaded]);

  // Draw canvas with image, prompts, and masks
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = imageRef.current;
    canvas.width = img.width;
    canvas.height = img.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw box prompt (SAM only)
    if (state.detectionMethod === 'sam' && state.boxPrompt) {
      const [x1, y1, x2, y2] = state.boxPrompt;
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    // Draw point prompts (SAM only)
    if (state.detectionMethod === 'sam') {
      state.pointPrompts.forEach(prompt => {
      ctx.beginPath();
      ctx.arc(prompt.x, prompt.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = prompt.label === 1 ? '#00ff00' : '#ff0000';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
      });
    }
    
    // Draw Random Forest annotations (semi-transparent with outline for visibility)
    // Use offscreen canvas so overlapping points don't compound opacity.
    if (state.detectionMethod === 'random_forest') {
      if (state.annotations.length > 0) {
        const offscreen = document.createElement('canvas');
        offscreen.width = canvas.width;
        offscreen.height = canvas.height;
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          state.annotations.forEach(ann => {
            const fillColor = ann.label === 1 
              ? 'rgb(0, 255, 0)'
              : 'rgb(255, 0, 0)';
            const strokeColor = ann.label === 1
              ? 'rgb(0, 200, 0)'
              : 'rgb(200, 0, 0)';

            offCtx.fillStyle = fillColor;
            ann.points.forEach(([x, y]) => {
              offCtx.beginPath();
              offCtx.arc(x, y, 2, 0, Math.PI * 2);
              offCtx.fill();
            });

            offCtx.strokeStyle = strokeColor;
            offCtx.lineWidth = 1;
            ann.points.forEach(([x, y]) => {
              offCtx.beginPath();
              offCtx.arc(x, y, 2, 0, Math.PI * 2);
              offCtx.stroke();
            });
          });

          ctx.save();
          ctx.globalAlpha = 0.45;
          ctx.drawImage(offscreen, 0, 0);
          ctx.restore();
        }
      }
      
      // Draw brush preview (semi-transparent circle following cursor)
      if (state.brushPreview.x !== null && state.brushPreview.y !== null) {
        const previewColor = state.annotationMode === 'exosome' 
          ? 'rgba(0, 255, 0, 0.4)' 
          : 'rgba(255, 0, 0, 0.4)';
        ctx.save();
        ctx.fillStyle = previewColor;
        ctx.strokeStyle = state.annotationMode === 'exosome' ? 'rgba(0, 200, 0, 0.8)' : 'rgba(200, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(state.brushPreview.x, state.brushPreview.y, state.brushSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
      
      // Draw confidence map if enabled (use offscreen canvas to preserve image underneath)
      if (state.showConfidenceMap && state.probabilityMap) {
        const cmCanvas = document.createElement('canvas');
        cmCanvas.width = canvas.width;
        cmCanvas.height = canvas.height;
        const cmCtx = cmCanvas.getContext('2d');
        if (cmCtx) {
          const imageData = cmCtx.createImageData(canvas.width, canvas.height);
          const data = imageData.data;
          
          for (let y = 0; y < Math.min(state.probabilityMap.length, canvas.height); y++) {
            for (let x = 0; x < Math.min(state.probabilityMap[y].length, canvas.width); x++) {
              const prob = state.probabilityMap[y][x];
              const r = Math.min(255, prob * 255);
              const b = Math.min(255, (1 - prob) * 255);
              const g = 0;
              
              const i = (y * canvas.width + x) * 4;
              data[i] = r;
              data[i + 1] = g;
              data[i + 2] = b;
              data[i + 3] = 255;
            }
          }
          
          cmCtx.putImageData(imageData, 0, 0);
          ctx.save();
          ctx.globalAlpha = state.maskOpacity;
          ctx.drawImage(cmCanvas, 0, 0);
          ctx.restore();
        }
      } else if (state.masks && state.masks.length > 0) {
        // Draw binary mask overlay (green semi-transparent)
        ctx.save();
        ctx.globalAlpha = state.maskOpacity;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.4)';
        
        const mask = state.masks[0];
        for (let y = 0; y < Math.min(mask.length, canvas.height); y++) {
          for (let x = 0; x < Math.min(mask[y].length, canvas.width); x++) {
            if (mask[y][x]) {
              ctx.fillRect(x, y, 1, 1);
            }
          }
        }
        
        ctx.restore();
      }
    }

    // Draw detections: use masks if available, otherwise use bboxes
    if (state.masks && state.masks.length > 0) {
      // Draw masks on an offscreen canvas, then composite onto main canvas.
      // This avoids putImageData which destroys the underlying image.
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = canvas.width;
      maskCanvas.height = canvas.height;
      const maskCtx = maskCanvas.getContext('2d');
      
      if (maskCtx) {
        state.masks.forEach((mask, idx) => {
          const maskHeight = mask.length;
          const maskWidth = maskHeight > 0 ? mask[0].length : 0;
          
          if (maskWidth === 0 || maskHeight === 0) return;
          
          const hue = (idx * 137.5) % 360;
          const [r, g, b] = hslToRgb(hue / 360, 0.7, 0.5);
          
          const imageData = maskCtx.createImageData(canvas.width, canvas.height);
          const data = imageData.data;
          
          for (let y = 0; y < Math.min(maskHeight, canvas.height); y++) {
            for (let x = 0; x < Math.min(maskWidth, canvas.width); x++) {
              if (mask[y] && mask[y][x]) {
                const i = (y * canvas.width + x) * 4;
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = 255;
              }
            }
          }
          
          maskCtx.putImageData(imageData, 0, 0);
        });

        ctx.save();
        ctx.globalAlpha = state.maskOpacity;
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.restore();
      }

      // Draw mask outlines
      if (state.showMaskOutlines) {
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 1;
        state.masks.forEach(mask => {
          const maskHeight = mask.length;
          const maskWidth = maskHeight > 0 ? mask[0].length : 0;
          
          if (maskWidth === 0 || maskHeight === 0) return;
          
          // Simple outline: draw border pixels
          for (let y = 1; y < Math.min(maskHeight - 1, canvas.height - 1); y++) {
            for (let x = 1; x < Math.min(maskWidth - 1, canvas.width - 1); x++) {
              if (mask[y] && mask[y][x] && 
                  (y === 0 || !mask[y - 1] || !mask[y - 1][x] || 
                   y >= maskHeight - 1 || !mask[y + 1] || !mask[y + 1][x] ||
                   x === 0 || !mask[y][x - 1] || 
                   x >= maskWidth - 1 || !mask[y][x + 1])) {
                ctx.fillStyle = '#ffff00';
                ctx.fillRect(x, y, 1, 1);
              }
            }
          }
        });
      }

      // Highlight selected detection
      if (selectedDetectionRef.current !== null && state.detections[selectedDetectionRef.current]) {
        const det = state.detections[selectedDetectionRef.current];
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 3;
        ctx.strokeRect(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]);
      }
    } else if (state.detections && state.detections.length > 0) {
      // Draw bboxes when masks are not available (large response size)
      ctx.save();
      ctx.globalAlpha = state.maskOpacity;
      
      state.detections.forEach((detection, idx) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Use different colors for different detections
        const hue = (idx * 137.5) % 360;
        const [r, g, b] = hslToRgb(hue / 360, 0.7, 0.5);
        
        // Draw filled rectangle
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${state.maskOpacity})`;
        ctx.fillRect(x1, y1, width, height);
        
        // Draw outline
        if (state.showMaskOutlines) {
          ctx.strokeStyle = '#ffff00';
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, width, height);
        }
      });
      
      ctx.restore();
      
      // Highlight selected detection
      if (selectedDetectionRef.current !== null && state.detections[selectedDetectionRef.current]) {
        const det = state.detections[selectedDetectionRef.current];
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 3;
        ctx.strokeRect(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]);
      }
    }

    // Draw detection count
    if (state.detections.length > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(10, 10, 200, 30);
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(`Detected: ${state.detections.length} exosomes`, 15, 30);
    }
    
    // Draw calibration crosshairs if calibration mode is enabled
    if (state.calibrationMode && state.debugLogs.length > 0) {
      const lastClick = state.debugLogs[state.debugLogs.length - 1];
      if (lastClick.type === 'click' && lastClick.data) {
        const { vx, vy, imageX, imageY, projectedPx, projectedPy } = lastClick.data;
        
        // Crosshair 1: At computed image coordinates (green) - this is where marker appears
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        const size = 15;
        ctx.beginPath();
        ctx.moveTo(imageX - size, imageY);
        ctx.lineTo(imageX + size, imageY);
        ctx.moveTo(imageX, imageY - size);
        ctx.lineTo(imageX, imageY + size);
        ctx.stroke();
        
        // Crosshair 2: At raw viewport coordinates converted to image space (red) - where mouse actually clicked
        if (viewportRef.current) {
          const viewportRect = viewportRef.current.getBoundingClientRect();
          // Convert viewport coords (vx, vy) to image coords
          const rawImageX = (vx - zoomState.offsetX) / zoomState.scale;
          const rawImageY = (vy - zoomState.offsetY) / zoomState.scale;
          
          ctx.strokeStyle = '#ff0000';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(rawImageX - size, rawImageY);
          ctx.lineTo(rawImageX + size, rawImageY);
          ctx.moveTo(rawImageX, rawImageY - size);
          ctx.lineTo(rawImageX, rawImageY + size);
          ctx.stroke();
        }
      }
    }
  }, [state.boxPrompt, state.pointPrompts, state.masks, state.maskOpacity, state.showMaskOutlines, state.detections, state.annotations, state.probabilityMap, state.showConfidenceMap, state.detectionMethod, state.calibrationMode, state.debugLogs, state.brushPreview, state.brushSize, state.annotationMode, zoomState]);

  // Helper: HSL to RGB
  const hslToRgb = (h: number, s: number, l: number): [number, number, number] => {
    let r, g, b;
    if (s === 0) {
      r = g = b = l;
    } else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      };
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  };

  // Redraw canvas when state changes
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Debug logging for coordinate mapping (legacy - now handled in getCanvasCoords)
  const logClickDebug = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.MouseEvent<HTMLDivElement>, computedCoords: { x: number; y: number }) => {
    // This function is kept for compatibility but coordinate logging is now in getCanvasCoords
    if (!state.debugLogging || !canvasRef.current || !viewportRef.current || !contentRef.current) return;
    
    // Legacy function - coordinate logging is now handled in getCanvasCoords
    // This is kept for backward compatibility but may not be called
  }, [state.debugLogging, zoomState.scale, zoomState.offsetX, zoomState.offsetY]);
  
  // Get canvas coordinates from mouse event (accounting for zoom and pan)
  // Fixed coordinate mapping: viewport -> image coordinates
  // Structure: viewport (captures events) -> content (transformed) -> canvas/image
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement> | React.MouseEvent<HTMLDivElement> | React.WheelEvent<HTMLCanvasElement> | PointerEvent | MouseEvent): { x: number; y: number } => {
    if (!canvasRef.current || !viewportRef.current || !contentRef.current) return { x: 0, y: 0 };
    
    const canvas = canvasRef.current;
    const viewport = viewportRef.current;
    const content = contentRef.current;
    
    // CRITICAL: Always get fresh rects at click-time (never cache)
    const viewportRect = viewport.getBoundingClientRect();
    const contentRect = content.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    
    // A) Event coordinates
    const eventCoords = {
      clientX: e.clientX,
      clientY: e.clientY,
      pageX: (e as any).pageX ?? e.clientX,
      pageY: (e as any).pageY ?? e.clientY,
      screenX: (e as any).screenX ?? 0,
      screenY: (e as any).screenY ?? 0,
    };
    
    // B) Element rects (ALL relevant elements)
    const elementRects = {
      viewport: {
        left: viewportRect.left,
        top: viewportRect.top,
        width: viewportRect.width,
        height: viewportRect.height,
      },
      content: {
        left: contentRect.left,
        top: contentRect.top,
        width: contentRect.width,
        height: contentRect.height,
      },
      canvas: {
        left: canvasRect.left,
        top: canvasRect.top,
        width: canvasRect.width,
        height: canvasRect.height,
      },
    };
    
    // C) Current transform state
    const { offsetX: tx, offsetY: ty, scale: s } = zoomState;
    const transformState = {
      scale: s,
      translateX: tx,
      translateY: ty,
      transformOrigin: '0 0',
    };
    
    // D) Coordinate pipeline (step-by-step)
    // CRITICAL FIX: Use canvas rect directly since it's inside the transformed content
    // The canvas rect is AFTER transform, so we can work backwards from it
    // Canvas internal size = 1024x1024, CSS size = 1024*s x 1024*s (after scale)
    // Canvas top-left in viewport = canvasRect.left/top
    // Canvas top-left in content space (before transform) = (0, 0)
    // After transform: (0, 0) -> (tx, ty) in content space, then positioned by flexbox
    // So canvasRect.left = viewport.left + flexboxOffset + tx
    // And canvasRect.top = viewport.top + flexboxOffset + ty
    
    // Viewport coordinates (relative to viewport top-left)
    const vx = e.clientX - viewportRect.left;
    const vy = e.clientY - viewportRect.top;
    
    // Canvas coordinates (relative to canvas top-left in viewport space)
    const canvasX = e.clientX - canvasRect.left;
    const canvasY = e.clientY - canvasRect.top;
    
    // Canvas CSS size vs internal size ratio
    const cssToInternalRatio = canvas.width / canvasRect.width;
    
    // Convert CSS canvas coordinates to internal canvas coordinates (image pixel space)
    const ix = canvasX * cssToInternalRatio;
    const iy = canvasY * cssToInternalRatio;
    
    const coordinatePipeline = {
      step1_viewportX: vx,
      step1_viewportY: vy,
      step2_canvasX: canvasX,
      step2_canvasY: canvasY,
      step3_cssToInternalRatio: cssToInternalRatio,
      step4_imageX: ix,
      step4_imageY: iy,
    };
    
    // E) Projection sanity check (must match original click)
    // Project back: canvas CSS X = imageX / cssToInternalRatio, then add canvasRect.left
    const projectedCanvasCSSX = ix / cssToInternalRatio;
    const projectedCanvasCSSY = iy / cssToInternalRatio;
    const projectedViewportX = projectedCanvasCSSX + (canvasRect.left - viewportRect.left);
    const projectedViewportY = projectedCanvasCSSY + (canvasRect.top - viewportRect.top);
    const errorX = projectedViewportX - vx;
    const errorY = projectedViewportY - vy;
    
    const projectionCheck = {
      projectedCanvasCSSX,
      projectedCanvasCSSY,
      projectedViewportX,
      projectedViewportY,
      originalVx: vx,
      originalVy: vy,
      errorX,
      errorY,
      isAccurate: Math.abs(errorX) < 0.1 && Math.abs(errorY) < 0.1,
    };
    
    // F) Overlay sizing sanity check
    const overlaySizing = {
      canvasInternalWidth: canvas.width,
      canvasInternalHeight: canvas.height,
      canvasCSSWidth: canvasRect.width,
      canvasCSSHeight: canvasRect.height,
      imageNaturalWidth: imageRef.current?.naturalWidth ?? canvas.width,
      imageNaturalHeight: imageRef.current?.naturalHeight ?? canvas.height,
    };
    
    // Clamp to canvas bounds (canvas is in image pixel space)
    const coords = {
      x: Math.max(0, Math.min(canvas.width, ix)),
      y: Math.max(0, Math.min(canvas.height, iy)),
    };
    
    // Comprehensive console logging (ALWAYS log, not just when debugLogging is enabled)
    const debugObject = {
      timestamp: new Date().toISOString(),
      eventCoords,
      elementRects,
      transformState,
      coordinatePipeline: {
        ...coordinatePipeline,
        finalImageX: coords.x,
        finalImageY: coords.y,
      },
      projectionCheck,
      overlaySizing,
      clampedCoords: coords,
    };
    
    // Log to console with groupCollapsed and table
    console.groupCollapsed('[CLICK DEBUG]');
    console.table(debugObject);
    console.log(debugObject);
    console.groupEnd();
    
    // Store for calibration visualization and UI display
    if (state.debugLogging || state.calibrationMode) {
      setState(prev => ({
        ...prev,
        debugLogs: [...prev.debugLogs.slice(-49), {
          timestamp: Date.now(),
          type: 'click',
          data: debugObject,
        }],
      }));
    }
    
    return coords;
  };
  
  // Handle wheel events (zoom or brush size adjustment)
  const handleWheel = useCallback((e: WheelEvent) => {
    // Always prevent default to avoid page scrolling and browser zoom
    e.preventDefault();
    e.stopPropagation();
    
    if (!viewportRef.current) return;
    
    // Shift + Wheel = adjust brush size (Random Forest only)
    if (e.shiftKey && state.detectionMethod === 'random_forest') {
      const delta = e.deltaY > 0 ? -1 : 1;
      setState(prev => ({
        ...prev,
        brushSize: Math.max(1, Math.min(50, prev.brushSize + delta)),
      }));
      return; // Don't zoom, just adjust brush size
    }
    
    // Wheel alone = zoom the image
    const viewport = viewportRef.current;
    const rect = viewport.getBoundingClientRect();
    
    // Get mouse position relative to viewport
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Calculate zoom factor (1.1 per step, or 0.9 for zoom out)
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    const newScale = Math.max(0.5, Math.min(8.0, zoomState.scale * zoomFactor));
    
    // Calculate new offset to keep zoom centered on mouse cursor
    // Formula: newOffset = mousePos - (mousePos - oldOffset) * (newScale / oldScale)
    const scaleRatio = newScale / zoomState.scale;
    const newOffsetX = mouseX - (mouseX - zoomState.offsetX) * scaleRatio;
    const newOffsetY = mouseY - (mouseY - zoomState.offsetY) * scaleRatio;
    
    setZoomState({
      scale: newScale,
      offsetX: newOffsetX,
      offsetY: newOffsetY,
    });
  }, [state.detectionMethod, zoomState.scale, zoomState.offsetX, zoomState.offsetY]);
  
  // Set up wheel event listener with passive: false
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    
    viewport.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      viewport.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);
  
  // Keyboard shortcuts: [ and ] to adjust brush size (Random Forest)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (state.detectionMethod !== 'random_forest') return;
      // Ignore if focus is on an input/textarea/select
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (e.key === '[') {
        e.preventDefault();
        setState(prev => ({
          ...prev,
          brushSize: Math.max(1, prev.brushSize - 1),
        }));
      } else if (e.key === ']') {
        e.preventDefault();
        setState(prev => ({
          ...prev,
          brushSize: Math.min(50, prev.brushSize + 1),
        }));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.detectionMethod]);

  // ResizeObserver: Detect viewport/layout changes (DevTools, window resize, etc.)
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const rect = entry.contentRect;
        const boundingRect = viewport.getBoundingClientRect();
        console.log('[RESIZE]', {
          timestamp: new Date().toISOString(),
          element: 'viewport',
          contentRect: {
            width: rect.width,
            height: rect.height,
          },
          boundingRect: {
            left: boundingRect.left,
            top: boundingRect.top,
            width: boundingRect.width,
            height: boundingRect.height,
          },
        });
      }
    });
    
    resizeObserver.observe(viewport);
    
    // Also observe window resize
    const handleWindowResize = () => {
      if (viewport) {
        const rect = viewport.getBoundingClientRect();
        console.log('[RESIZE]', {
          timestamp: new Date().toISOString(),
          element: 'window',
          viewportRect: {
            left: rect.left,
            top: rect.top,
            width: rect.width,
            height: rect.height,
          },
        });
      }
    };
    
    window.addEventListener('resize', handleWindowResize);
    
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', handleWindowResize);
    };
  }, []);
  
  // Handle pan start
  const handlePanStart = (e: React.PointerEvent<HTMLDivElement>) => {
    // Only pan if:
    // 1. Zoomed in (scale > 1.0)
    // 2. Not in annotation mode (Random Forest)
    // 3. Not drawing (SAM box mode)
    // 4. Middle mouse button or left click when not in active drawing mode
    const isMiddleButton = e.button === 1;
    const canPan = zoomState.scale > 1.0 && 
                   !isAnnotatingRef.current && 
                   !isDrawingRef.current &&
                   (isMiddleButton || (e.button === 0 && state.detectionMethod !== 'random_forest' && (state.detectionMethod !== 'sam' || state.detectionMode !== 'box')));
    
    if (canPan) {
      e.preventDefault();
      isPanningRef.current = true;
      if (viewportRef.current) {
        const rect = viewportRef.current.getBoundingClientRect();
        panStartRef.current = {
          x: e.clientX - rect.left - zoomState.offsetX,
          y: e.clientY - rect.top - zoomState.offsetY,
        };
        viewportRef.current.style.cursor = 'grabbing';
      }
    }
  };
  
  // Handle pan move
  const handlePanMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isPanningRef.current && panStartRef.current && viewportRef.current) {
      e.preventDefault();
      const rect = viewportRef.current.getBoundingClientRect();
      setZoomState(prev => ({
        ...prev,
        offsetX: e.clientX - rect.left - panStartRef.current!.x,
        offsetY: e.clientY - rect.top - panStartRef.current!.y,
      }));
    }
  };
  
  // Handle pan end
  const handlePanEnd = () => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      panStartRef.current = null;
      if (viewportRef.current) {
        // Update cursor based on current state
        const shouldShowGrab = zoomState.scale > 1.0 && 
                              !isAnnotatingRef.current && 
                              !isDrawingRef.current &&
                              (state.detectionMethod !== 'random_forest' && (state.detectionMethod !== 'sam' || state.detectionMode !== 'box'));
        viewportRef.current.style.cursor = shouldShowGrab ? 'grab' : 'default';
      }
    }
  };
  
  // Reset zoom on double click
  const handleDoubleClick = () => {
    setZoomState({
      scale: 1.0,
      offsetX: 0,
      offsetY: 0,
    });
  };

  // Add brush point with radius (Random Forest)
  const addBrushPoint = useCallback((x: number, y: number, label: number, annotationId?: string) => {
    const radius = state.brushSize;
    const points: Array<[number, number]> = [];
    
    // Generate points in a circle
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy <= radius * radius) {
          points.push([Math.round(x + dx), Math.round(y + dy)]);
        }
      }
    }
    
    currentAnnotationPointsRef.current.push(...points);
    
    // Update annotations state
    setState(prev => {
      if (annotationId) {
        // Add to existing annotation group
        const existingIndex = prev.annotations.findIndex(ann => ann.id === annotationId);
        if (existingIndex >= 0) {
          const updated = [...prev.annotations];
          updated[existingIndex] = {
            ...updated[existingIndex],
            points: [...updated[existingIndex].points, ...points],
          };
          return { ...prev, annotations: updated };
        }
      }
      
      // Create new annotation group, preserving the caller-provided ID if given
      const newId = annotationId || `ann_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      return {
        ...prev,
        annotations: [...prev.annotations, { id: newId, points, label }],
      };
    });
    
    drawCanvas();
  }, [state.brushSize, drawCanvas]);
  
  // Canvas mouse handlers for box/point prompts (SAM only) and brush annotation (Random Forest)
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    // Ignore middle-click — handled at viewport level to prevent browser auto-scroll
    if (e.button === 1) return;

    const { x, y } = getCanvasCoords(e);

    if (state.detectionMethod === 'random_forest') {
      // Brush annotation mode
      isAnnotatingRef.current = true;
      currentAnnotationPointsRef.current = [];
      
      // Generate a tracked annotation ID so click history can reference it
      const annId = `ann_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      currentAnnotationIdRef.current = annId;
      
      // Determine label based on button or mode
      const label = e.button === 0 
        ? (state.annotationMode === 'exosome' ? 1 : 0)
        : (state.annotationMode === 'exosome' ? 0 : 1);
      
      // Add initial point with the tracked ID
      addBrushPoint(x, y, label, annId);
    } else if (state.detectionMethod === 'sam') {
      if (state.detectionMode === 'box') {
        isDrawingRef.current = true;
        startPosRef.current = { x, y };
        setState(prev => ({ ...prev, boxPrompt: [x, y, x, y] }));
      } else if (state.detectionMode === 'point') {
        // Toggle point: left click = positive, right click = negative
        const label = e.button === 0 ? 1 : 0;
        setState(prev => ({
          ...prev,
          pointPrompts: [...prev.pointPrompts, { x, y, label }],
        }));
      }
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    
    const { x, y } = getCanvasCoords(e);
    
    // Update brush preview position (for Random Forest)
    if (state.detectionMethod === 'random_forest') {
      setState(prev => ({
        ...prev,
        brushPreview: { x, y },
      }));
      drawCanvas(); // Redraw to show preview
    }
    
    if (state.detectionMethod === 'random_forest' && isAnnotatingRef.current) {
      // Continue brush stroke with the same annotation ID
      const label = (e.buttons & 1)  // bitwise: left button held (handles left+middle = 5 too)
        ? (state.annotationMode === 'exosome' ? 1 : 0)
        : (state.annotationMode === 'exosome' ? 0 : 1);
      addBrushPoint(x, y, label, currentAnnotationIdRef.current || undefined);
    } else if (state.detectionMethod === 'sam' && isDrawingRef.current && startPosRef.current) {
      if (state.detectionMode === 'box') {
        setState(prev => ({
          ...prev,
          boxPrompt: prev.boxPrompt
            ? [Math.min(startPosRef.current!.x, x), Math.min(startPosRef.current!.y, y), Math.max(startPosRef.current!.x, x), Math.max(startPosRef.current!.y, y)]
            : null,
        }));
      }
    }
  };

  const handleCanvasMouseUp = () => {
    isDrawingRef.current = false;
    startPosRef.current = null;
    isAnnotatingRef.current = false;
    currentAnnotationPointsRef.current = [];
  };
  
  // Handle mouse leave - hide brush preview
  const handleCanvasMouseLeave = () => {
    handleCanvasMouseUp();
    if (state.detectionMethod === 'random_forest') {
      setState(prev => ({
        ...prev,
        brushPreview: { x: null, y: null },
      }));
      drawCanvas();
    }
  };
  
  // Handle canvas click for pixel inspector
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const { x, y } = getCanvasCoords(e);
    
    // Get pixel intensity from image
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const imageData = ctx.getImageData(x, y, 1, 1);
    const intensity = imageData.data[0]; // Grayscale intensity
    
    // Get confidence and predicted class if available
    let confidence: number | null = null;
    let predictedClass: string | null = null;
    
    if (state.probabilityMap && state.probabilityMap.length > 0) {
      const py = Math.floor(y);
      const px = Math.floor(x);
      if (py >= 0 && py < state.probabilityMap.length && px >= 0 && px < state.probabilityMap[py].length) {
        confidence = state.probabilityMap[py][px];
        predictedClass = confidence >= state.confidenceThreshold ? 'Exosome' : 'Background';
      }
    }
    
    // Link to the annotation already created by mousedown (don't create a duplicate)
    const annotationId = currentAnnotationIdRef.current || undefined;
    
    const clickId = `click_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const clickEntry = {
      id: clickId,
      timestamp: Date.now(),
      imageX: Math.round(x),
      imageY: Math.round(y),
      intensity,
      confidence,
      predictedClass,
      scale: zoomState.scale,
      offsetX: zoomState.offsetX,
      offsetY: zoomState.offsetY,
      annotationId,
    };
    
    setState(prev => ({
      ...prev,
      pixelInspector: {
        x: Math.round(x),
        y: Math.round(y),
        intensity,
        confidence,
        predictedClass,
      },
      clickHistory: [clickEntry, ...prev.clickHistory].slice(0, 100), // Keep last 100 clicks
    }));
  };
  
  // Remove a click history entry and its associated annotation
  const handleRemoveClick = (clickId: string, annotationId?: string) => {
    setState(prev => {
      // Remove from click history
      const newClickHistory = prev.clickHistory.filter(click => click.id !== clickId);
      
      // Remove associated annotation if exists
      let newAnnotations = prev.annotations;
      if (annotationId) {
        newAnnotations = prev.annotations.filter(ann => ann.id !== annotationId);
      }
      
      return {
        ...prev,
        clickHistory: newClickHistory,
        annotations: newAnnotations,
      };
    });
    drawCanvas();
  };
  
  // Canvas wheel handler removed - all wheel events handled by container

  // Run segmentation
  const handleRunSegmentation = async () => {
    if (!state.loaded || !state.selectedSample || !state.selectedPosition || !state.selectedChannel) {
      alert('Please load an image first');
      return;
    }

    if (!state.checkpointPath) {
      alert('Please specify SAM checkpoint path');
      return;
    }

    if (state.detectionMethod === 'sam') {
      if (state.detectionMode === 'box' && !state.boxPrompt) {
        alert('Please draw a box prompt');
        return;
      }

      if (state.detectionMode === 'point' && state.pointPrompts.length === 0) {
        alert('Please add at least one point prompt');
        return;
      }

      if (!state.checkpointPath) {
        alert('Please specify SAM checkpoint path');
        return;
      }
    } else if (state.detectionMethod === 'random_forest') {
      if (state.annotations.length === 0) {
        alert('Please annotate some pixels first (draw on the image)');
        return;
      }
    }

    setState(prev => ({ ...prev, isDetecting: true, exportStatus: null }));

    try {
      const prompts: any = {};
      if (state.detectionMode === 'box' && state.boxPrompt) {
        prompts.box = state.boxPrompt;
      } else if (state.detectionMode === 'point') {
        prompts.points = state.pointPrompts.map(p => [p.x, p.y]);
        prompts.labels = state.pointPrompts.map(p => p.label);
      }

      const requestBody: any = {
        sample: state.selectedSample,
        position: state.selectedPosition,
        channel: state.selectedChannel,
        method: state.detectionMethod,
        min_area: state.minArea,
        max_area: state.maxArea,
        remove_small_objects: state.removeSmallObjects,
        fill_holes: state.fillHoles,
      };

      if (state.detectionMethod === 'sam') {
        requestBody.mode = state.detectionMode;
        requestBody.prompts = prompts;
        requestBody.checkpoint_path = state.checkpointPath;
        requestBody.model_type = state.modelType;
        requestBody.device = state.device;
        requestBody.score_thresh = state.confidenceThreshold;
      } else if (state.detectionMethod === 'blob') {
        requestBody.threshold = state.blobThreshold;
        requestBody.min_circularity = state.blobMinCircularity;
        requestBody.max_circularity = state.blobMaxCircularity;
        requestBody.min_inertia_ratio = state.blobMinInertiaRatio;
      } else if (state.detectionMethod === 'random_forest') {
        requestBody.annotations = state.annotations;
        requestBody.confidence_threshold = state.confidenceThreshold;
        requestBody.apply_morphology = state.fillHoles;
        requestBody.n_estimators = 100;
      }

      console.log('[ExosomeDetection] Sending request:', { method: state.detectionMethod, ...requestBody });
      
      const response = await fetch('${getApiBase()}/api/exosome/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      console.log('[ExosomeDetection] Response status:', response.status);
      const data = await response.json();
      console.log('[ExosomeDetection] Response data:', data);
      
      if (!data.success) {
        console.error('[ExosomeDetection] Error from backend:', data.error);
        throw new Error(data.error || 'Segmentation failed');
      }

      // Convert masks from backend format (list of 2D boolean arrays)
      const masksRaw = data.data.masks || [];
      const scores = data.data.scores || [];
      const masksOmitted = data.data._masks_omitted || false;
      const warning = data.data._warning;
      
      console.log('[ExosomeDetection] Received masks:', masksRaw.length, 'scores:', scores.length, 'detections:', data.data.detections?.length || 0);
      if (masksOmitted) {
        console.warn('[ExosomeDetection] Masks omitted due to large response size. Using bboxes for visualization.');
        if (warning) {
          alert(`Warning: ${warning}\n\nVisualization will use bounding boxes instead of full masks.`);
        }
      }
      
      // Ensure masks are in the correct format (list of 2D boolean arrays)
      const masks = masksRaw.map((mask: any) => {
        if (Array.isArray(mask) && Array.isArray(mask[0])) {
          return mask; // Already in correct format
        }
        return mask; // Fallback
      });
      
      const detections: DetectionResult[] = (data.data.detections || []).map((d: any, idx: number) => ({
        id: idx + 1,
        area: d.area || 0,
        centroid: d.centroid || [0, 0],
        bbox: d.bbox || [0, 0, 0, 0],
        score: idx < scores.length ? scores[idx] : undefined,
      }));

      console.log('[ExosomeDetection] Processed detections:', detections.length);

      // Handle Random Forest specific results
      let probabilityMap = null;
      if (state.detectionMethod === 'random_forest' && data.data.probability_map) {
        probabilityMap = data.data.probability_map;
        console.log('[ExosomeDetection] Received probability map');
      }

      setState(prev => ({
        ...prev,
        masks: masks.length > 0 ? masks : null, // null if masks omitted
        scores: scores,
        detections: detections,
        probabilityMap: probabilityMap || prev.probabilityMap,
        isDetecting: false,
      }));

      console.log('[ExosomeDetection] State updated, drawing canvas...');
      drawCanvas();
    } catch (err: any) {
      console.error('Segmentation failed:', err);
      alert('Segmentation failed: ' + err.message);
      setState(prev => ({ ...prev, isDetecting: false }));
    }
  };

  // Clear prompts
  const handleClearPrompts = () => {
    setState(prev => ({
      ...prev,
      boxPrompt: null,
      pointPrompts: [],
    }));
  };

  // Clear masks
  const handleClearMasks = () => {
    setState(prev => ({
      ...prev,
      masks: null,
      scores: null,
      detections: [],
    }));
  };

  // Export results
  const handleExport = async () => {
    if (!state.loaded || state.detections.length === 0) {
      alert('No detections to export');
      return;
    }

    setState(prev => ({ ...prev, exportStatus: 'Exporting...' }));

    try {
      const response = await fetch('${getApiBase()}/api/exosome/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: state.selectedSample,
          position: state.selectedPosition,
          channel: state.selectedChannel,
          detections: state.detections,
          masks: state.masks,
          scores: state.scores,
          settings: {
            model_type: state.modelType,
            checkpoint_path: state.checkpointPath,
            device: state.device,
            confidence_threshold: state.confidenceThreshold,
            min_area: state.minArea,
            max_area: state.maxArea,
          },
        }),
      });

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Export failed');
      }

      setState(prev => ({
        ...prev,
        exportStatus: 'Export completed',
        exportPaths: data.data.paths || [],
      }));
    } catch (err: any) {
      console.error('Export failed:', err);
      alert('Export failed: ' + err.message);
      setState(prev => ({ ...prev, exportStatus: 'Export failed' }));
    }
  };

  // Calculate summary statistics
  const summaryStats = state.detections.length > 0
    ? {
        count: state.detections.length,
        meanArea: state.detections.reduce((sum, d) => sum + d.area, 0) / state.detections.length,
        medianArea: (() => {
          const sorted = [...state.detections].sort((a, b) => a.area - b.area);
          const mid = Math.floor(sorted.length / 2);
          return sorted.length % 2 === 0
            ? (sorted[mid - 1].area + sorted[mid].area) / 2
            : sorted[mid].area;
        })(),
      }
    : null;

  return (
    <div className="exosome-detection">
      <div className="exosome-layout">
        {/* LEFT: Controls Panel */}
        <div className="exosome-controls">
          <h2>🔬 Exosome Detection</h2>

          {/* Image Source */}
          <div className="control-section">
            <h3>Image Source</h3>
            <div className="input-group">
              <label>Sample:</label>
              <select
                value={state.selectedSample}
                onChange={(e) => setState(prev => ({ ...prev, selectedSample: e.target.value }))}
              >
                <option value="">-- Select sample --</option>
                {state.availableSamples.map(sample => (
                  <option key={sample} value={sample}>{sample}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>Position:</label>
              <select
                value={state.selectedPosition}
                onChange={(e) => setState(prev => ({ ...prev, selectedPosition: e.target.value }))}
                disabled={!state.selectedSample}
              >
                <option value="">-- Select position --</option>
                {state.availablePositions.map(pos => (
                  <option key={pos} value={pos}>{pos}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>Channel:</label>
              <select
                value={state.selectedChannel}
                onChange={(e) => setState(prev => ({ ...prev, selectedChannel: e.target.value }))}
                disabled={!state.loaded}
              >
                <option value="">-- Select channel --</option>
                {state.availableItems.map(item => (
                  <option key={item.key} value={item.key}>{item.display_label}</option>
                ))}
              </select>
            </div>
            <button onClick={handleLoadPosition} disabled={!state.selectedSample || !state.selectedPosition}>
              Load Image
            </button>
          </div>

          {/* Detection Method */}
          <div className="control-section">
            <h3>Detection Method</h3>
            <div className="input-group">
              <label>Method:</label>
              <select
                value={state.detectionMethod}
                onChange={(e) => setState(prev => ({ ...prev, detectionMethod: e.target.value as 'sam' | 'blob' | 'random_forest' }))}
              >
                <option value="sam">SAM (Segment Anything Model)</option>
                <option value="blob">Blob-Based Detection</option>
                <option value="random_forest">Random Forest (Interactive, ML)</option>
              </select>
            </div>
          </div>
          
          {/* Random Forest Annotation Controls */}
          {state.detectionMethod === 'random_forest' && (
          <div className="control-section">
            <h3>Annotation Controls</h3>
            <div className="input-group">
              <label>Annotation Mode:</label>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    value="exosome"
                    checked={state.annotationMode === 'exosome'}
                    onChange={(e) => setState(prev => ({ ...prev, annotationMode: e.target.value as 'exosome' | 'background' }))}
                  />
                  Exosome (Left Click)
                </label>
                <label>
                  <input
                    type="radio"
                    value="background"
                    checked={state.annotationMode === 'background'}
                    onChange={(e) => setState(prev => ({ ...prev, annotationMode: e.target.value as 'exosome' | 'background' }))}
                  />
                  Background (Right Click)
                </label>
              </div>
            </div>
            <div className="input-group">
              <label>Brush Size: {state.brushSize}px</label>
              <SliderInput
                label="Brush Size"
                value={state.brushSize}
                min={1}
                max={50}
                step={1}
                isInteger={true}
                precision={0}
                onChange={(value) => setState(prev => ({ ...prev, brushSize: Math.round(value) }))}
              />
              <small style={{color: '#666', display: 'block', marginTop: '0.25rem'}}>
                Shortcuts: <b>[</b> / <b>]</b> keys or Shift + Mouse Wheel
              </small>
            </div>
            <button onClick={() => setState(prev => ({ ...prev, annotations: [] }))}>
              Clear Annotations
            </button>
            <small style={{color: '#666', display: 'block', marginTop: '0.5rem'}}>
              Draw on the image to annotate pixels. Left click = Exosome, Right click = Background.
            </small>
          </div>
          )}

          {/* Model Settings (SAM only) */}
          {state.detectionMethod === 'sam' && (
          <div className="control-section">
            <h3>Model Settings ⚠️ Required</h3>
            <div className="input-group">
              <label>Checkpoint Path: <span style={{color: 'red'}}>*</span></label>
              <input
                type="text"
                value={state.checkpointPath}
                onChange={(e) => setState(prev => ({ ...prev, checkpointPath: e.target.value }))}
                placeholder="/path/to/sam_checkpoint.pth"
                style={{ 
                  borderColor: !state.checkpointPath ? '#e74c3c' : '#ddd',
                  borderWidth: !state.checkpointPath ? '2px' : '1px'
                }}
              />
              {!state.checkpointPath && (
                <small style={{color: '#e74c3c', display: 'block', marginTop: '0.25rem'}}>
                  Checkpoint path is required
                </small>
              )}
              <small style={{color: '#666', display: 'block', marginTop: '0.25rem'}}>
                Default: checkpoints/pretrained/sam_vit_h_4b8939.pth
              </small>
            </div>
            <div className="input-group">
              <label>Model Type:</label>
              <select
                value={state.modelType}
                onChange={(e) => setState(prev => ({ ...prev, modelType: e.target.value }))}
              >
                <option value="sam_vit_h">SAM ViT-H (default)</option>
                <option value="sam_vit_l">SAM ViT-L</option>
                <option value="sam_vit_b">SAM ViT-B</option>
              </select>
            </div>
            <div className="input-group">
              <label>Device:</label>
              <select
                value={state.device}
                onChange={(e) => setState(prev => ({ ...prev, device: e.target.value }))}
              >
                <option value="auto">Auto</option>
                <option value="cuda">CUDA</option>
                <option value="cpu">CPU</option>
              </select>
            </div>
          </div>
          )}

          {/* Blob Detection Parameters */}
          {state.detectionMethod === 'blob' && (
          <div className="control-section">
            <h3>Blob Detection Parameters</h3>
            <SliderInput
              label="Threshold"
              value={state.blobThreshold}
              min={0}
              max={1}
              step={0.001}
              precision={3}
              onChange={(value) => setState(prev => ({ ...prev, blobThreshold: value }))}
            />
            <SliderInput
              label="Min Circularity"
              value={state.blobMinCircularity}
              min={0}
              max={1}
              step={0.001}
              precision={3}
              onChange={(value) => setState(prev => ({ ...prev, blobMinCircularity: value }))}
            />
            <SliderInput
              label="Max Circularity"
              value={state.blobMaxCircularity}
              min={0}
              max={1}
              step={0.001}
              precision={3}
              onChange={(value) => setState(prev => ({ ...prev, blobMaxCircularity: value }))}
            />
            <SliderInput
              label="Min Inertia Ratio"
              value={state.blobMinInertiaRatio}
              min={0}
              max={1}
              step={0.001}
              precision={3}
              onChange={(value) => setState(prev => ({ ...prev, blobMinInertiaRatio: value }))}
            />
          </div>
          )}

          {/* Detection Mode (SAM only) */}
          {state.detectionMethod === 'sam' && (
          <div className="control-section">
            <h3>Detection Mode</h3>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  value="box"
                  checked={state.detectionMode === 'box'}
                  onChange={(e) => setState(prev => ({ ...prev, detectionMode: e.target.value as 'box' | 'point' | 'auto' }))}
                />
                Box Prompt
              </label>
              <label>
                <input
                  type="radio"
                  value="point"
                  checked={state.detectionMode === 'point'}
                  onChange={(e) => setState(prev => ({ ...prev, detectionMode: e.target.value as 'box' | 'point' | 'auto' }))}
                />
                Point Prompt
              </label>
            </div>
          </div>
          )}

          {/* Parameters */}
          <div className="control-section">
            <h3>Parameters</h3>
            <SliderInput
              label="Confidence Threshold"
              value={state.confidenceThreshold}
              min={0}
              max={1}
              step={0.001}
              precision={3}
              onChange={(value) => setState(prev => ({ ...prev, confidenceThreshold: value }))}
            />
            <SliderInput
              label="Min Area (px)"
              value={state.minArea}
              min={0}
              max={100000}
              step={1}
              isInteger={true}
              precision={0}
              onChange={(value) => setState(prev => ({ ...prev, minArea: Math.round(value) }))}
            />
            <SliderInput
              label="Max Area (px)"
              value={state.maxArea}
              min={0}
              max={1000000}
              step={1}
              isInteger={true}
              precision={0}
              onChange={(value) => setState(prev => ({ ...prev, maxArea: Math.round(value) }))}
            />
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={state.removeSmallObjects}
                  onChange={(e) => setState(prev => ({ ...prev, removeSmallObjects: e.target.checked }))}
                />
                Remove Small Objects
              </label>
            </div>
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={state.fillHoles}
                  onChange={(e) => setState(prev => ({ ...prev, fillHoles: e.target.checked }))}
                />
                Fill Holes
              </label>
            </div>
          </div>

          {/* Actions */}
          <div className="control-section">
            <h3>Actions</h3>
            <button
              onClick={handleRunSegmentation}
              disabled={!state.loaded || state.isDetecting}
              className="primary-button"
            >
              {state.isDetecting ? 'Running...' : 'Run Segmentation'}
            </button>
            <button onClick={handleClearPrompts}>Clear Prompts</button>
            <button onClick={handleClearMasks}>Clear Masks</button>
            <button onClick={handleExport} disabled={state.detections.length === 0}>
              Export Results
            </button>
          </div>
        </div>

        {/* CENTER: Image Viewer */}
        <div className="exosome-viewer">
          <div className="viewer-controls">
            <SliderInput
              label="Mask Opacity"
              value={state.maskOpacity}
              min={0}
              max={1}
              step={0.01}
              precision={2}
              onChange={(value) => setState(prev => ({ ...prev, maskOpacity: value }))}
            />
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={state.showMaskOutlines}
                  onChange={(e) => setState(prev => ({ ...prev, showMaskOutlines: e.target.checked }))}
                />
                Show Mask Outlines
              </label>
            </div>
            {state.detectionMethod === 'random_forest' && (
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={state.showConfidenceMap}
                  onChange={(e) => setState(prev => ({ ...prev, showConfidenceMap: e.target.checked }))}
                />
                Show Confidence Map
              </label>
            </div>
            )}
          </div>
          {/* Viewport: Single source of truth for pointer events */}
          <div 
            ref={viewportRef}
            className="canvas-container"
            onAuxClick={(e) => { e.preventDefault(); e.stopPropagation(); }}
            onPointerDown={(e) => {
              // Suppress all default middle-click browser behavior (auto-scroll, new-tab, etc.)
              if (e.button === 1) { e.preventDefault(); e.stopPropagation(); }
              // Only start panning if:
              // 1. Zoomed in
              // 2. Not in annotation/drawing mode
              // 3. Middle mouse button or left click when not in active drawing mode
              if (zoomState.scale > 1.0 &&
                  !isAnnotatingRef.current && 
                  !isDrawingRef.current &&
                  (e.button === 1 || (e.button === 0 && state.detectionMethod !== 'random_forest' && (state.detectionMethod !== 'sam' || state.detectionMode !== 'box')))) {
                handlePanStart(e);
              } else {
                // Let canvas handle the event for annotations/drawing
                handleCanvasMouseDown(e as any);
              }
            }}
            onPointerMove={(e) => {
              if (isPanningRef.current) {
                handlePanMove(e);
              } else {
                handleCanvasMouseMove(e as any);
              }
            }}
            onPointerUp={(e) => {
              handlePanEnd();
              handleCanvasMouseUp();
            }}
            onPointerLeave={(e) => {
              handlePanEnd();
              handleCanvasMouseLeave();
            }}
            onContextMenu={(e) => e.preventDefault()}
            onDoubleClick={handleDoubleClick}
            style={{
              position: 'relative',
              overflow: 'hidden',
              width: '100%',
              height: '100%',
              cursor: zoomState.scale > 1.0 && !isAnnotatingRef.current && !isDrawingRef.current ? 'grab' : 'default',
            }}
          >
            {state.currentImageUrl ? (
              <>
                {/* Zoom indicator */}
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  background: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '0.5rem 1rem',
                  borderRadius: '4px',
                  fontSize: '0.9rem',
                  zIndex: 10,
                  pointerEvents: 'none',
                }}>
                  Zoom: {Math.round(zoomState.scale * 100)}%
                </div>
                
                {/* Content wrapper: ONLY element with transform */}
                <div
                  ref={contentRef}
                  style={{
                    transform: `translate(${zoomState.offsetX}px, ${zoomState.offsetY}px) scale(${zoomState.scale})`,
                    transformOrigin: '0 0',
                    transition: isPanningRef.current ? 'none' : 'transform 0.1s ease-out',
                    willChange: 'transform',
                    width: 'fit-content',
                    height: 'fit-content',
                  }}
                >
                  {/* Canvas: matches image natural dimensions, scaled by parent transform */}
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    style={{ 
                      display: 'block',
                      maxWidth: 'none',
                      touchAction: 'none',
                    }}
                  />
                </div>
              </>
            ) : (
              <div className="placeholder">Load an image to start detection</div>
            )}
          </div>

          {/* Results Section: Summary + Histogram + Table (below image) */}
          {state.detections.length > 0 && summaryStats && (
            <div className="viewer-results">
              {/* Summary Cards Row */}
              <div className="summary-row">
                <div className="summary-stat">
                  <span className="summary-stat-value">{summaryStats.count}</span>
                  <span className="summary-stat-label">Objects</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-stat-value">{summaryStats.meanArea.toFixed(1)}</span>
                  <span className="summary-stat-label">Mean Area (px²)</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-stat-value">{summaryStats.medianArea.toFixed(1)}</span>
                  <span className="summary-stat-label">Median Area (px²)</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-stat-value">
                    {Math.min(...state.detections.map(d => d.area)).toFixed(1)}
                  </span>
                  <span className="summary-stat-label">Min Area (px²)</span>
                </div>
                <div className="summary-stat">
                  <span className="summary-stat-value">
                    {Math.max(...state.detections.map(d => d.area)).toFixed(1)}
                  </span>
                  <span className="summary-stat-label">Max Area (px²)</span>
                </div>
              </div>

              {/* Area Distribution Histogram */}
              <AreaHistogram detections={state.detections} />

              {/* Detected Objects Table */}
              <div className="results-table-container">
                <h3>Detected Objects ({state.detections.length})</h3>
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  <table className="detections-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Area (px²)</th>
                        <th>Centroid (x, y)</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {state.detections.map((det, idx) => (
                        <tr
                          key={det.id}
                          onClick={() => {
                            selectedDetectionRef.current = idx;
                            drawCanvas();
                          }}
                          className={selectedDetectionRef.current === idx ? 'selected' : ''}
                        >
                          <td>{det.id}</td>
                          <td>{det.area.toFixed(1)}</td>
                          <td>({det.centroid[0].toFixed(1)}, {det.centroid[1].toFixed(1)})</td>
                          <td>{det.score !== undefined ? det.score.toFixed(3) : 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Export Status */}
              {state.exportStatus && (
                <div className="export-status">
                  <h3>Export Status</h3>
                  <p>{state.exportStatus}</p>
                  {state.exportPaths.length > 0 && (
                    <ul>
                      {state.exportPaths.map((path, idx) => (
                        <li key={idx}>{path}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* RIGHT: Annotation Tools Panel */}
        <div className="exosome-results">
          <h2>Annotation Tools</h2>
          
          {/* Debug Logging Panel */}
          <div className="control-section" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h3>Debug & Calibration</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={state.debugLogging}
                  onChange={(e) => setState(prev => ({ ...prev, debugLogging: e.target.checked }))}
                  style={{ marginRight: '0.5rem' }}
                />
                Debug coordinate logging
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={state.calibrationMode}
                  onChange={(e) => setState(prev => ({ ...prev, calibrationMode: e.target.checked }))}
                  style={{ marginRight: '0.5rem' }}
                />
                Calibration Mode (show crosshairs)
              </label>
            </div>
            {state.debugLogging && (
              <div>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <button 
                    onClick={() => setState(prev => ({ ...prev, debugLogs: [] }))}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
                  >
                    Clear Logs
                  </button>
                  <button 
                    onClick={() => {
                      const json = JSON.stringify(state.debugLogs, null, 2);
                      navigator.clipboard.writeText(json);
                      alert('Debug logs copied to clipboard');
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
                  >
                    Copy JSON
                  </button>
                </div>
                <div style={{ 
                  maxHeight: '200px', 
                  overflowY: 'auto', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px',
                  padding: '0.5rem',
                  fontSize: '0.75rem',
                  fontFamily: 'monospace',
                  backgroundColor: '#f5f5f5'
                }}>
                  {state.debugLogs.length === 0 ? (
                    <p style={{color: '#999', margin: 0}}>No debug logs yet. Click on the image to generate logs.</p>
                  ) : (
                    state.debugLogs.map((log, idx) => (
                      <div key={idx} style={{ 
                        marginBottom: '0.5rem',
                        padding: '0.5rem',
                        backgroundColor: 'white',
                        borderRadius: '4px',
                        border: '1px solid #ddd'
                      }}>
                        <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
                          [{log.type}] {new Date(log.timestamp).toLocaleTimeString()}
                        </div>
                        {log.type === 'click' && log.data && (
                          <div style={{ fontSize: '0.7rem' }}>
                            {log.data.vx !== undefined ? (
                              <>
                                <div><strong>Viewport:</strong> vx={log.data.vx.toFixed(1)}, vy={log.data.vy.toFixed(1)}</div>
                                <div><strong>Transform:</strong> scale={log.data.s.toFixed(3)}, offset=({log.data.tx.toFixed(1)}, {log.data.ty.toFixed(1)})</div>
                                <div><strong>Image:</strong> ({log.data.imageX.toFixed(2)}, {log.data.imageY.toFixed(2)})</div>
                                <div><strong>Projected:</strong> px={log.data.projectedPx.toFixed(1)}, py={log.data.projectedPy.toFixed(1)}</div>
                                <div><strong>Match:</strong> {Math.abs(log.data.vx - log.data.projectedPx) < 1 && Math.abs(log.data.vy - log.data.projectedPy) < 1 ? '✓' : '✗'}</div>
                              </>
                            ) : (
                              <>
                                <div><strong>Raw:</strong> clientX={log.data.rawEvent?.clientX}, clientY={log.data.rawEvent?.clientY}</div>
                                {log.data.transformState && (
                                  <>
                                    <div><strong>Transform:</strong> scale={log.data.transformState.scale.toFixed(3)}, offset=({log.data.transformState.offsetX.toFixed(1)}, {log.data.transformState.offsetY.toFixed(1)})</div>
                                    {log.data.conversions && (
                                      <div><strong>Image:</strong> ({log.data.conversions.step3_divideByScaleX.toFixed(2)}, {log.data.conversions.step3_divideByScaleY.toFixed(2)})</div>
                                    )}
                                  </>
                                )}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* Pixel Inspector (Random Forest) */}
          {state.detectionMethod === 'random_forest' && (
          <div className="control-section" style={{ marginBottom: '1.5rem' }}>
            <h3>Pixel Inspector</h3>
            {state.pixelInspector.x !== null ? (
              <div>
                <p><strong>Coordinates:</strong> ({state.pixelInspector.x}, {state.pixelInspector.y})</p>
                <p><strong>Intensity:</strong> {state.pixelInspector.intensity !== null ? state.pixelInspector.intensity : 'N/A'}</p>
                {state.pixelInspector.confidence !== null && (
                  <>
                    <p><strong>Confidence:</strong> {state.pixelInspector.confidence.toFixed(3)}</p>
                    <p><strong>Predicted Class:</strong> {state.pixelInspector.predictedClass || 'N/A'}</p>
                  </>
                )}
                {state.pixelInspector.confidence === null && (
                  <p style={{color: '#999', fontSize: '0.85rem'}}>Click "Run Segmentation" to see confidence values</p>
                )}
              </div>
            ) : (
              <p style={{color: '#999', fontSize: '0.9rem'}}>Click on the image to inspect pixels</p>
            )}
            
            {/* Click History */}
            <div style={{ marginTop: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem' }}>Click History ({state.clickHistory.length})</h4>
                <div>
                  <button 
                    onClick={() => {
                      setState(prev => ({ ...prev, clickHistory: [], annotations: [] }));
                      drawCanvas();
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', marginRight: '0.25rem' }}
                  >
                    Clear All
                  </button>
                  <button 
                    onClick={() => {
                      const json = JSON.stringify(state.clickHistory, null, 2);
                      navigator.clipboard.writeText(json);
                      alert('Click history copied to clipboard as JSON');
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', marginRight: '0.25rem' }}
                  >
                    JSON
                  </button>
                  <button 
                    onClick={() => {
                      const csv = [
                        'Timestamp,ImageX,ImageY,Intensity,Confidence,PredictedClass,Scale,OffsetX,OffsetY',
                        ...state.clickHistory.map(h => 
                          `${h.timestamp},${h.imageX},${h.imageY},${h.intensity || ''},${h.confidence || ''},${h.predictedClass || ''},${h.scale},${h.offsetX},${h.offsetY}`
                        )
                      ].join('\n');
                      navigator.clipboard.writeText(csv);
                      alert('Click history copied to clipboard as CSV');
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                  >
                    CSV
                  </button>
                </div>
              </div>
              <div style={{ 
                maxHeight: '200px', 
                overflowY: 'auto', 
                border: '1px solid #ddd', 
                borderRadius: '4px',
                padding: '0.5rem',
                fontSize: '0.85rem',
                backgroundColor: '#f9f9f9'
              }}>
                {state.clickHistory.length === 0 ? (
                  <p style={{color: '#999', fontSize: '0.85rem', margin: 0}}>No clicks yet</p>
                ) : (
                  state.clickHistory.map((click, idx) => (
                    <div key={click.id} style={{ 
                      padding: '0.5rem',
                      borderBottom: idx < state.clickHistory.length - 1 ? '1px solid #eee' : 'none',
                      backgroundColor: 'white',
                      borderRadius: '4px',
                      marginBottom: '0.25rem'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                            <span style={{ fontWeight: 'bold' }}>#{idx + 1} ({click.imageX}, {click.imageY})</span>
                            <span style={{ color: '#666', fontSize: '0.75rem' }}>
                              {new Date(click.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.75rem', color: '#666' }}>
                            <div><strong>Intensity:</strong> {click.intensity !== null ? click.intensity : 'N/A'}</div>
                            {click.confidence !== null && <div><strong>Confidence:</strong> {click.confidence.toFixed(3)}</div>}
                            {click.predictedClass && <div><strong>Class:</strong> {click.predictedClass}</div>}
                            <div style={{ fontSize: '0.7rem', color: '#999', marginTop: '0.25rem' }}>
                              Zoom: {Math.round(click.scale * 100)}% | Pan: ({Math.round(click.offsetX)}, {Math.round(click.offsetY)})
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => handleRemoveClick(click.id, click.annotationId)}
                          style={{
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.7rem',
                            marginLeft: '0.5rem',
                            backgroundColor: '#ff4444',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                          }}
                          title="Remove this point"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExosomeDetection;

