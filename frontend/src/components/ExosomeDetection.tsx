/**
 * Exosome Detection Component
 * Uses SAM (Segment Anything Model) for exosome segmentation
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import '../styles/ExosomeDetection.css';
import { getApiBase } from '../lib/apiBase';
import { copyText } from '../lib/clipboard';
import * as storage from '../lib/storage';

interface NormStats {
  dtype: string;
  original_min: number;
  original_max: number;
  p0_5: number;
  p99_5: number;
  display_min: number;
  display_max: number;
  normalization: string;
  cache_hit?: boolean;
  preview_path?: string;
}

interface ChannelItem {
  key: string;
  display_label: string;
  preview_url?: string;
  // Normalization / display-pipeline debug info
  norm_stats?: NormStats;
  display_source?: string;  // e.g. 'previews/abc123.png'
  detection_source?: string; // e.g. 'crop_channels/def456.tif'
  crop_mode?: boolean;
  source_tiff?: string;
  channel_index?: number;
  label?: string;
}

interface DetectionResult {
  id: number;
  area: number;
  centroid: [number, number];
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score?: number;
  perimeter?: number;
  circularity?: number;
}

interface SavedRfModelItem {
  position: string;
  saved_at?: string | null;
  trained_on?: {
    sample?: string;
    position?: string;
    channel?: string;
  };
  sklearn_version?: string | null;
  path: string;
}

interface GuidePoint {
  id: number;
  x: number;
  y: number;
}

/** Mask foreground at (y,x); out-of-range is background — used for contour tests. */
function maskPixelOn(mask: boolean[][], y: number, x: number): boolean {
  if (y < 0 || x < 0) return false;
  const row = mask[y];
  if (!row || x >= row.length) return false;
  return !!row[x];
}

function hexToRgb(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

/**
 * Visit every outline pixel (4-connected foreground with at least one off-mask neighbor).
 * Scans full mask bounds inside the canvas (fixes the old loops that skipped y=0,x=0 and broke edge logic).
 */
function forEachContourPixel(
  mask: boolean[][],
  canvasWidth: number,
  canvasHeight: number,
  visit: (x: number, y: number) => void,
): void {
  const h = mask.length;
  if (h === 0) return;
  const w = mask[0]?.length ?? 0;
  if (w === 0) return;
  const maxY = Math.min(h, canvasHeight) - 1;
  const maxX = Math.min(w, canvasWidth) - 1;
  for (let y = 0; y <= maxY; y++) {
    const row = mask[y];
    if (!row) continue;
    for (let x = 0; x <= maxX; x++) {
      if (!row[x]) continue;
      const border =
        !maskPixelOn(mask, y - 1, x) ||
        !maskPixelOn(mask, y + 1, x) ||
        !maskPixelOn(mask, y, x - 1) ||
        !maskPixelOn(mask, y, x + 1);
      if (border) visit(x, y);
    }
  }
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
  annotationMode: 'exosome' | 'background' | 'eraser'; // Current annotation mode (E toggles eraser)
  
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
    clickType?: 'exosome' | 'background';
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
  rfModelStatus: string | null;
  rfModelWarning: string | null;
  rfModelSavedPath: string | null;
  availableRfModels: SavedRfModelItem[];
  selectedRfModelSourcePosition: string;

  // Ground truth overlay
  showGroundTruth: boolean;
  groundTruthPoints: Array<{ x: number; y: number }>;
  groundTruthCount: number;
  groundTruthFile: string | null;
  groundTruthPixelSizeUm: number; // µm/pixel from TIFF metadata (returned by backend)
  gtImageWidth: number | null;    // raw TIFF width in pixels (from backend)
  gtImageHeight: number | null;   // raw TIFF height in pixels (from backend)
  // raw µm sample for debug display (first 20 CSV rows before conversion)
  rawGtSampleUm: Array<{ x_um: number; y_um: number; x_px: number; y_px: number }>;

  // GT transform controls (all diagnostic / togglable)
  gtDebugMode: boolean;
  gtSwapXY: boolean;    // swap CSV X↔Y before any other transform
  gtYFlip: boolean;     // flip Y axis: y_final = imageHeight - y  (bottom-left origin correction)
  gtOffsetX: number;    // add this pixel offset to every X after other transforms
  gtOffsetY: number;    // add this pixel offset to every Y after other transforms

  // Crop mode info (returned by backend when position contains 'crop')
  cropMode: boolean;
  cropTiffFile: string | null;
  cropTiffShape: number[] | null;  // e.g. [14, 835, 900]
  cropTiffAxes: string | null;     // e.g. 'CYX'
  cropNumChannels: number | null;
  cropPixelSizeUm: number | null;
  cropPixelSizeSource: string | null; // 'metadata' | 'fallback'

  // Exosome-detection-only display mode (does NOT affect detection pipeline)
  displayMode: 'raw_16bit' | 'enhanced' | 'minmax';
  displayLut: 'gray' | 'red';
  // Normalization stats for the CURRENTLY displayed image (updated on every channel/mode change)
  currentNormStats: NormStats | null;
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
        onContextMenu={(e) => e.preventDefault()}
        style={{ width: '100%', height: '220px', borderRadius: '4px' }}
      />
    </div>
  );
};

// ---------------------------------------------------------------------------
// DetectionTable: sortable + filterable result table
// ---------------------------------------------------------------------------
type SortKey = 'id' | 'area' | 'perimeter' | 'circularity';
type SortDir = 'asc' | 'desc';

interface DetectionTableProps {
  detections: DetectionResult[];
  selectedIdx: number | null;
  onRowClick: (idx: number) => void;
  filterArea: { min: string; max: string };
  filterCirc: { min: string; max: string };
  onFilterAreaChange: React.Dispatch<React.SetStateAction<{ min: string; max: string }>>;
  onFilterCircChange: React.Dispatch<React.SetStateAction<{ min: string; max: string }>>;
  onFilteredIndicesChange: (indices: number[]) => void;
  onApplyFilters: () => void;
  onSaveFilter: () => void;
  saveMessage?: string;
}

const DetectionTable: React.FC<DetectionTableProps> = React.memo(({
  detections,
  selectedIdx,
  onRowClick,
  filterArea,
  filterCirc,
  onFilterAreaChange,
  onFilterCircChange,
  onFilteredIndicesChange,
  onApplyFilters,
  onSaveFilter,
  saveMessage,
}) => {
  const [sortKey, setSortKey] = React.useState<SortKey>('id');
  const [sortDir, setSortDir] = React.useState<SortDir>('asc');

  const handleHeaderClick = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortArrow = (key: SortKey) => sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  const filtered = React.useMemo(() => {
    const aMin = filterArea.min !== '' ? parseFloat(filterArea.min) : -Infinity;
    const aMax = filterArea.max !== '' ? parseFloat(filterArea.max) :  Infinity;
    const cMin = filterCirc.min !== '' ? parseFloat(filterCirc.min) : -Infinity;
    const cMax = filterCirc.max !== '' ? parseFloat(filterCirc.max) :  Infinity;

    return detections
      .map((d, origIdx) => ({ d, origIdx }))
      .filter(({ d }) => {
        if (d.area < aMin || d.area > aMax) return false;
        const c = d.circularity ?? 0;
        if (c < cMin || c > cMax) return false;
        return true;
      })
      .sort((a, b) => {
        let av = 0, bv = 0;
        if (sortKey === 'id')          { av = a.d.id;                     bv = b.d.id; }
        else if (sortKey === 'area')   { av = a.d.area;                   bv = b.d.area; }
        else if (sortKey === 'perimeter') { av = a.d.perimeter ?? 0;      bv = b.d.perimeter ?? 0; }
        else if (sortKey === 'circularity') { av = a.d.circularity ?? 0;  bv = b.d.circularity ?? 0; }
        return sortDir === 'asc' ? av - bv : bv - av;
      });
  }, [detections, sortKey, sortDir, filterArea, filterCirc]);

  const applyFilters = () => {
    onFilteredIndicesChange(filtered.map(({ origIdx }) => origIdx));
    onApplyFilters();
  };

  const thStyle: React.CSSProperties = {
    cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap',
    padding: '6px 8px', background: '#f0f0f0', borderBottom: '2px solid #ccc',
  };
  const filterInputStyle: React.CSSProperties = {
    width: '60px', padding: '2px 4px', border: '1px solid #ccc',
    borderRadius: '3px', fontSize: '0.75rem',
  };

  return (
    <div className="results-table-container">
      <h3>Detected Objects ({filtered.length} / {detections.length})</h3>

      {/* Filter row */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '0.5rem', fontSize: '0.8rem', alignItems: 'center' }}>
        <span style={{ fontWeight: 600 }}>Filter:</span>
        <label>Area min <input style={filterInputStyle} value={filterArea.min} onChange={e => onFilterAreaChange(p => ({ ...p, min: e.target.value }))} placeholder="—" /></label>
        <label>Area max <input style={filterInputStyle} value={filterArea.max} onChange={e => onFilterAreaChange(p => ({ ...p, max: e.target.value }))} placeholder="—" /></label>
        <label>Circ min <input style={filterInputStyle} value={filterCirc.min} onChange={e => onFilterCircChange(p => ({ ...p, min: e.target.value }))} placeholder="—" /></label>
        <label>Circ max <input style={filterInputStyle} value={filterCirc.max} onChange={e => onFilterCircChange(p => ({ ...p, max: e.target.value }))} placeholder="—" /></label>
        <button style={{ padding: '2px 8px', fontSize: '0.75rem' }} onClick={applyFilters}>Apply</button>
        <button
          style={{
            padding: '2px 10px',
            fontSize: '0.75rem',
            background: '#2563eb',
            color: 'white',
            border: '1px solid #1d4ed8',
            borderRadius: 4,
            cursor: 'pointer',
          }}
          onClick={onSaveFilter}
          title="Persist this filtered set for Results Viewer"
        >
          💾 Save Filter
        </button>
        <button style={{ padding: '2px 8px', fontSize: '0.75rem' }} onClick={() => { onFilterAreaChange({ min: '', max: '' }); onFilterCircChange({ min: '', max: '' }); }}>Reset</button>
        {!!saveMessage && (
          <span style={{ color: '#2563eb', fontWeight: 600 }}>
            {saveMessage}
          </span>
        )}
      </div>

      <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
        <table className="detections-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle} onClick={() => handleHeaderClick('id')}>ID{sortArrow('id')}</th>
              <th style={thStyle} onClick={() => handleHeaderClick('area')}>Area (px²){sortArrow('area')}</th>
              <th style={{ ...thStyle, cursor: 'default' }}>Centroid (x, y)</th>
              <th style={thStyle} onClick={() => handleHeaderClick('perimeter')}>Perimeter{sortArrow('perimeter')}</th>
              <th style={thStyle} onClick={() => handleHeaderClick('circularity')}>Circularity{sortArrow('circularity')}</th>
              <th style={{ ...thStyle, cursor: 'default' }}>Score</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(({ d, origIdx }) => (
              <tr
                key={d.id}
                onClick={() => onRowClick(origIdx)}
                className={selectedIdx === origIdx ? 'selected' : ''}
                style={{ cursor: 'pointer' }}
              >
                <td style={{ padding: '4px 8px', textAlign: 'center' }}>{d.id}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{d.area.toFixed(1)}</td>
                <td style={{ padding: '4px 8px', textAlign: 'center' }}>({d.centroid[0].toFixed(1)}, {d.centroid[1].toFixed(1)})</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{d.perimeter != null ? d.perimeter.toFixed(1) : '—'}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{d.circularity != null ? d.circularity.toFixed(3) : '—'}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{d.score !== undefined ? d.score.toFixed(3) : 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Exosome-detection-only display helper (does NOT touch any detection pipeline)
// ---------------------------------------------------------------------------

/**
 * Fetch the display-mode-correct preview URL for a channel.
 * - enhanced + gray  → reuse pre-cached preview_url from items (no extra round-trip)
 * - all other modes  → call /api/exosome/channel_display
 */
async function fetchChannelDisplay(
  apiBase: string,
  sample: string,
  position: string,
  channel: string,
  mode: string,
  lut: string,
  availableItems: ChannelItem[],
): Promise<{ url: string; normStats: NormStats | null } | null> {
  if (!sample || !position || !channel) return null;

  // enhanced + gray: the pre-generated preview from load_position is already correct
  if (mode === 'enhanced' && lut === 'gray') {
    const item = availableItems.find(i => i.key === channel);
    if (item?.preview_url) {
      return { url: item.preview_url, normStats: item.norm_stats || null };
    }
  }

  try {
    const resp = await fetch(`${apiBase}/api/exosome/channel_display`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample, position, channel, mode, lut }),
    });
    const data = await resp.json();
    if (data.success) {
      return { url: data.preview_url, normStats: data.norm_stats || null };
    }
    console.error('[fetchChannelDisplay] backend error:', data.error);
  } catch (err) {
    console.error('[fetchChannelDisplay] fetch error:', err);
  }
  return null;
}

// ---------------------------------------------------------------------------

const ExosomeDetection: React.FC<{ isActive?: boolean }> = ({ isActive = true }) => {
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
    
    detectionMethod: 'random_forest',
    
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
    rfModelStatus: null,
    rfModelWarning: null,
    rfModelSavedPath: null,
    availableRfModels: [],
    selectedRfModelSourcePosition: '',

    showGroundTruth: false,
    groundTruthPoints: [],
    groundTruthCount: 0,
    groundTruthFile: null,
    groundTruthPixelSizeUm: 0.21,
    gtImageWidth: null,
    gtImageHeight: null,
    rawGtSampleUm: [],

    gtDebugMode: false,
    gtSwapXY: false,
    gtYFlip: false,
    gtOffsetX: 0,
    gtOffsetY: 0,

    cropMode: false,
    cropTiffFile: null,
    cropTiffShape: null,
    cropTiffAxes: null,
    cropNumChannels: null,
    cropPixelSizeUm: null,
    cropPixelSizeSource: null,

    displayMode: 'raw_16bit',
    displayLut: 'gray',
    currentNormStats: null,
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
  const [filterArea, setFilterArea] = useState<{ min: string; max: string }>({ min: '', max: '' });
  const [filterCirc, setFilterCirc] = useState<{ min: string; max: string }>({ min: '', max: '' });
  const [showFilteredOnly, setShowFilteredOnly] = useState(false);
  const [showGuideFromRef, setShowGuideFromRef] = useState(false);
  const [guideRefPoints, setGuideRefPoints] = useState<GuidePoint[]>([]);
  const [guideRefAvailable, setGuideRefAvailable] = useState(false);
  const [filteredDetectionIndices, setFilteredDetectionIndices] = useState<number[]>([]);
  const [filterApplyNonce, setFilterApplyNonce] = useState(0);
  /** Bumps only on a fresh detection batch (new segmentation, clear, or load position) so filter indices are not reset on incidental state churn. */
  const [detectionBatchId, setDetectionBatchId] = useState(0);
  const [saveFilterMessage, setSaveFilterMessage] = useState<string>('');
  const isPanningRef = useRef<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number } | null>(null);

  const exosomeDiskTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const exosomeDiskPendingRef = useRef<Record<string, string>>({});
  const selectedSampleRef = useRef(state.selectedSample);
  selectedSampleRef.current = state.selectedSample;

  const scheduleExosomeKeysToDisk = (kv: Record<string, string>) => {
    Object.assign(exosomeDiskPendingRef.current, kv);
    if (exosomeDiskTimerRef.current) window.clearTimeout(exosomeDiskTimerRef.current);
    exosomeDiskTimerRef.current = window.setTimeout(() => {
      exosomeDiskTimerRef.current = null;
      const sample = selectedSampleRef.current;
      const merged = { ...exosomeDiskPendingRef.current };
      exosomeDiskPendingRef.current = {};
      if (sample && Object.keys(merged).length) {
        void storage.mergeExosomeStorageKeysOnDisk(sample, merged);
      }
    }, 500);
  };

  useEffect(() => () => {
    if (exosomeDiskTimerRef.current) window.clearTimeout(exosomeDiskTimerRef.current);
  }, []);

  useEffect(() => {
    if (!state.selectedSample || !state.selectedPosition) return;
    void storage.saveUiTabSlice(state.selectedSample, 'exosome', {
      position: state.selectedPosition,
      channel: state.selectedChannel || undefined,
      lastActiveTab: isActive ? 'exosome' : undefined,
    });
  }, [isActive, state.selectedSample, state.selectedPosition, state.selectedChannel]);

  const filteredDetectionIndexSet = useMemo(() => new Set(filteredDetectionIndices), [filteredDetectionIndices]);
  const annotationLabelById = useMemo(() => {
    const map = new Map<string, number>();
    state.annotations.forEach((ann) => map.set(ann.id, ann.label));
    return map;
  }, [state.annotations]);
  const getClickCategory = useCallback((click: ExosomeDetectionState['clickHistory'][number]): 'exosome' | 'background' => {
    if (click.clickType) return click.clickType;
    if (click.annotationId) {
      const label = annotationLabelById.get(click.annotationId);
      if (label === 0) return 'background';
      if (label === 1) return 'exosome';
    }
    if ((click.predictedClass || '').toLowerCase() === 'background') return 'background';
    return 'exosome';
  }, [annotationLabelById]);
  const filteredOutIndices = useMemo(() => {
    if (state.detections.length === 0) return [];
    const inSet = filteredDetectionIndexSet;
    return state.detections.map((_, i) => i).filter(i => !inSet.has(i));
  }, [filteredDetectionIndexSet, state.detections]);
  const handleFilteredIndicesChange = useCallback((indices: number[]) => {
    setFilteredDetectionIndices(prev => {
      if (prev.length === indices.length && prev.every((v, i) => v === indices[i])) return prev;
      return indices;
    });
  }, []);
  const handleApplyFilters = useCallback(() => {
    setFilterApplyNonce(v => v + 1);
  }, []);
  const handleSaveFilter = useCallback(async () => {
    try {
      const sample = state.selectedSample;
      const position = state.selectedPosition;
      const channel = state.selectedChannel;
      if (!sample || !position || !channel) {
        setSaveFilterMessage('Select sample/position/channel first');
        return;
      }
      const storageKey = `sea_filtered_detections_${sample}_${position}_${channel}`;
      const objectIds = filteredDetectionIndices
        .map((i) => state.detections[i]?.id)
        .filter((id): id is number => typeof id === 'number' && Number.isFinite(id));

      const filterPayload = JSON.stringify({
        enabled: true,
        objectIds,
        totalDetections: state.detections.length,
        savedAt: Date.now(),
      });
      await storage.set(storageKey, filterPayload);
      void storage.mergeExosomeStorageKeysOnDisk(sample, { [storageKey]: filterPayload });

      // Investigation logs (remove after debugging)
      try {
        const raw = await storage.get(storageKey);
        const parsed = raw ? JSON.parse(raw) : null;
        const ids = Array.isArray(parsed?.objectIds) ? parsed.objectIds : [];
        const uniq = new Set(ids.map((v: any) => String(v)));
        console.log('[SaveFilter] key=', storageKey);
        console.log('[SaveFilter] totalDetections=', parsed?.totalDetections, 'objectIds.length=', ids.length);
        console.log('[SaveFilter] idTypeSample=', ids.slice(0, 5).map((v: any) => typeof v), 'idSample=', ids.slice(0, 5));
        console.log('[SaveFilter] uniqueCount=', uniq.size, 'hasDuplicates=', uniq.size !== ids.length);
      } catch (e) {
        console.log('[SaveFilter] failed to log saved payload', e);
      }

      setSaveFilterMessage(`Saved ${objectIds.length} filtered detections`);
      window.setTimeout(() => setSaveFilterMessage(''), 2500);
    } catch {
      setSaveFilterMessage('Save failed');
      window.setTimeout(() => setSaveFilterMessage(''), 2500);
    }
  }, [filteredDetectionIndices, state.detections, state.selectedChannel, state.selectedPosition, state.selectedSample]);

  useEffect(() => {
    if (state.detections.length === 0) {
      setFilteredDetectionIndices([]);
      return;
    }
    setFilteredDetectionIndices(state.detections.map((_, idx) => idx));
  }, [detectionBatchId]); // eslint-disable-line react-hooks/exhaustive-deps -- reset filters only when detectionBatchId bumps (fresh detection batch).

  useEffect(() => {
    // Guide overlay is only for non-reference channels.
    if (state.selectedChannel === 'C0') {
      setShowGuideFromRef(false);
      setGuideRefPoints([]);
      setGuideRefAvailable(false);
      return;
    }
    const sample = state.selectedSample;
    const position = state.selectedPosition;
    if (!sample || !position) {
      setGuideRefAvailable(false);
      setGuideRefPoints([]);
      setShowGuideFromRef(false);
      return;
    }
    let cancelled = false;
    const loadGuideAvailability = async () => {
      const raw = await storage.get(`sea_filtered_detections_${sample}_${position}_C0`);
      if (cancelled || !raw) {
        setGuideRefAvailable(false);
        setGuideRefPoints([]);
        setShowGuideFromRef(false);
        return;
      }
      try {
        const parsed = JSON.parse(raw);
        const ids = Array.isArray(parsed?.objectIds) ? parsed.objectIds : [];
        const enabled = parsed?.enabled ?? true;
        const available = enabled && ids.length > 0;
        if (cancelled) return;
        setGuideRefAvailable(available);
        if (!available) {
          setGuideRefPoints([]);
          setShowGuideFromRef(false);
        }
      } catch {
        if (!cancelled) {
          setGuideRefAvailable(false);
          setGuideRefPoints([]);
          setShowGuideFromRef(false);
        }
      }
    };
    void loadGuideAvailability();
    return () => { cancelled = true; };
  }, [state.selectedSample, state.selectedPosition, state.selectedChannel]);

  useEffect(() => {
    const sample = state.selectedSample;
    const position = state.selectedPosition;
    if (!showGuideFromRef || !guideRefAvailable || !sample || !position || state.selectedChannel === 'C0') {
      setGuideRefPoints([]);
      return;
    }

    let cancelled = false;
    const loadGuide = async () => {
      const raw = await storage.get(`sea_filtered_detections_${sample}_${position}_C0`);
      if (!raw) {
        if (!cancelled) {
          setGuideRefPoints([]);
          setShowGuideFromRef(false);
        }
        return;
      }
      let objectIds: number[] = [];
      try {
        const parsed = JSON.parse(raw);
        objectIds = Array.isArray(parsed?.objectIds)
          ? parsed.objectIds.map((v: any) => Number(v)).filter((n: number) => Number.isFinite(n))
          : [];
      } catch {
        if (!cancelled) {
          setGuideRefPoints([]);
          setShowGuideFromRef(false);
        }
        return;
      }
      if (objectIds.length === 0) {
        if (!cancelled) {
          setGuideRefPoints([]);
          setShowGuideFromRef(false);
        }
        return;
      }
      try {
        const resp = await fetch(
          `${getApiBase()}/api/exosome/channel_detections?sample=${encodeURIComponent(sample)}&position=${encodeURIComponent(position)}&channel=${encodeURIComponent('C0')}`
        );
        const data = await resp.json();
        if (!resp.ok || !data.success) {
          throw new Error(data?.error || 'Failed to load reference detections');
        }
        if (cancelled) return;
        const byId = new Map<number, { x: number; y: number }>();
        (data.data?.detections || []).forEach((d: any) => {
          const id = Number(d.id);
          if (!Number.isFinite(id)) return;
          byId.set(id, { x: Number(d.centroid_x) || 0, y: Number(d.centroid_y) || 0 });
        });
        const points: GuidePoint[] = objectIds
          .map((id) => {
            const p = byId.get(id);
            if (!p) return null;
            return { id, x: p.x, y: p.y };
          })
          .filter((p): p is GuidePoint => !!p);
        setGuideRefPoints(points);
      } catch {
        if (!cancelled) {
          setGuideRefPoints([]);
        }
      }
    };
    loadGuide();
    return () => { cancelled = true; };
  }, [showGuideFromRef, guideRefAvailable, state.selectedSample, state.selectedPosition, state.selectedChannel]);
  
  // Viewport and content refs (single source of truth)
  const viewportRef = useRef<HTMLDivElement>(null); // Captures pointer events
  const contentRef = useRef<HTMLDivElement>(null); // Gets transform (translate + scale)

  // Default auto-load
  const autoLoadFiredRef = useRef(false);
  const [pendingAutoLoad, setPendingAutoLoad] = useState(false);

  // Fetch available samples on mount; auto-select default
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await fetch(`${getApiBase()}/api/input/samples`);
        const data = await response.json();
        if (data.success) {
          setState(prev => {
            const next = { ...prev, availableSamples: data.data };
            if (!prev.selectedSample && data.data.includes('A2780Cis10'))
              next.selectedSample = 'A2780Cis10';
            return next;
          });
        }
      } catch (err) {
        console.error('Failed to fetch samples:', err);
      }
    };
    fetchSamples();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Restore sea-storage keys from sample-local exosome_state.json when sample changes
  useEffect(() => {
    if (!state.selectedSample) return;
    let cancelled = false;
    void (async () => {
      try {
        const disk = await storage.loadStateFromDisk(state.selectedSample, true);
        if (cancelled) return;
        const ex = disk.exosome as { storage?: Record<string, string> } | undefined;
        if (!ex?.storage) return;
        for (const [k, v] of Object.entries(ex.storage)) {
          if (typeof v !== 'string' || !v.length) continue;
          if (!k.startsWith(`sea_`)) continue;
          await storage.set(k, v);
        }
      } catch {
        /* ignore */
      }
    })();
    return () => { cancelled = true; };
  }, [state.selectedSample]);

  // Fetch positions when sample changes; auto-select default (prefer .sea_state/ui)
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

    let cancelled = false;
    const fetchPositions = async () => {
      try {
        const response = await fetch(
          `${getApiBase()}/api/input/samples/${state.selectedSample}/positions`
        );
        const data = await response.json();
        if (!data.success || cancelled) return;
        const positions: string[] = data.data;
        const disk = await storage.loadStateFromDisk(state.selectedSample);
        if (cancelled) return;
        const ui = disk.ui as { exosome?: { position?: string; channel?: string } } | undefined;
        const wantPos = ui?.exosome?.position;
        setState(prev => {
          const next = { ...prev, availablePositions: positions };
          let sp = prev.selectedPosition;
          if (!sp || !positions.includes(sp)) {
            if (wantPos && positions.includes(wantPos)) sp = wantPos;
            else if (positions.includes('P1')) sp = 'P1';
            else sp = positions[0] || '';
          }
          next.selectedPosition = sp;
          if (
            !prev.selectedPosition &&
            positions.includes('P1') &&
            sp === 'P1' &&
            prev.selectedSample === 'A2780Cis10' &&
            !autoLoadFiredRef.current
          ) {
            setPendingAutoLoad(true);
          }
          return next;
        });
      } catch (err) {
        console.error('Failed to fetch positions:', err);
      }
    };
    void fetchPositions();
    return () => { cancelled = true; };
  }, [state.selectedSample]);

  /** Fit the loaded image in the viewport with a small margin (0.95). */
  const fitToViewport = useCallback(() => {
    const vp = viewportRef.current;
    const img = imageRef.current;
    if (!vp || !img) return;
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    if (iw <= 0 || ih <= 0) return;
    const rect = vp.getBoundingClientRect();
    const viewportWidth = rect.width;
    const viewportHeight = rect.height;
    if (viewportWidth <= 0 || viewportHeight <= 0) return;
    const scale = Math.min(viewportWidth / iw, viewportHeight / ih) * 0.95;
    const offsetX = (viewportWidth - iw * scale) / 2;
    const offsetY = (viewportHeight - ih * scale) / 2;
    setZoomState({ scale, offsetX, offsetY });
  }, []);

  // Load position and get channel info
  const handleLoadPosition = async () => {
    if (!state.selectedSample || !state.selectedPosition) {
      alert('Please select both sample and position');
      return;
    }

    try {
      const response = await fetch(`${getApiBase()}/api/input/load_position`, {
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
      const d = data.data;

      let initialChannel = firstItem?.key || '';
      try {
        const diskUi = await storage.loadStateFromDisk(state.selectedSample, true);
        const wantCh = (diskUi.ui as { exosome?: { channel?: string } } | undefined)?.exosome?.channel;
        if (wantCh && items.some(i => i.key === wantCh)) initialChannel = wantCh;
      } catch { /* keep default */ }
      
      setDetectionBatchId(b => b + 1);
      setState(prev => ({
        ...prev,
        availableItems: items,
        selectedChannel: initialChannel,
        loaded: items.length > 0,
        currentImageUrl: firstItem?.preview_url || null,
        boxPrompt: null,
        pointPrompts: [],
        masks: null,
        scores: null,
        detections: [],
        clickHistory: [], // Reset click history when loading new image
        debugLogs: [], // Reset debug logs when loading new image
        // Crop mode metadata
        cropMode: d.crop_mode ?? false,
        cropTiffFile: d.tiff_file ?? null,
        cropTiffShape: d.tiff_shape ?? null,
        cropTiffAxes: d.tiff_axes ?? null,
        cropNumChannels: d.n_channels ?? null,
        cropPixelSizeUm: d.pixel_size_um ?? null,
        cropPixelSizeSource: d.pixel_size_source ?? null,
      }));

      // Auto-load ground truth for this sample/position
      fetch(`${getApiBase()}/api/exosome/ground_truth?sample=${encodeURIComponent(state.selectedSample)}&position=${encodeURIComponent(state.selectedPosition)}`)
        .then(r => r.json())
        .then(gt => {
          if (gt.success) {
            setState(prev => ({
              ...prev,
              groundTruthPoints: gt.data.points,
              groundTruthCount: gt.data.count,
              groundTruthFile: gt.data.file,
              groundTruthPixelSizeUm: gt.data.pixel_size_um ?? 0.21,
              gtImageWidth:  gt.data.image_width  ?? null,
              gtImageHeight: gt.data.image_height ?? null,
              rawGtSampleUm: gt.data.raw_sample_um ?? [],
            }));
          }
        })
        .catch(() => { /* no ground truth available */ });

      // Load display image for the initial channel using the current display mode
      if (firstItem) {
        const result = await fetchChannelDisplay(
          getApiBase(),
          state.selectedSample,
          state.selectedPosition,
          initialChannel || firstItem.key,
          state.displayMode,
          state.displayLut,
          items,
        );
        if (result) {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            setState(prev => ({
              ...prev,
              imageWidth: img.width,
              imageHeight: img.height,
              currentNormStats: result.normStats,
            }));
            imageRef.current = img;
            drawCanvas();
            requestAnimationFrame(() => {
              fitToViewport();
            });
          };
          const url = result.url.startsWith('http') ? result.url : `${getApiBase()}${result.url}`;
          img.src = url;
        }
      }
    } catch (err) {
      console.error('Failed to load position:', err);
      alert('Failed to load position: ' + err);
    }
  };

  // Auto-load effect — placed after handleLoadPosition to avoid hoisting error
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!pendingAutoLoad || autoLoadFiredRef.current) return;
    autoLoadFiredRef.current = true;
    setPendingAutoLoad(false);
    handleLoadPosition();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAutoLoad]);

  // ── Click history localStorage helpers ──────────────────────────────────────

  const clickStorageKey = (sample: string, position: string, channel: string, method: string) =>
    `sea_clicks_${sample}_${position}_${channel}_${method}`;

  const saveClickHistory = () => {
    const { selectedSample, selectedPosition, selectedChannel, detectionMethod, clickHistory, annotations } = state;
    if (!selectedSample || !selectedPosition || !selectedChannel) return;
    const payload = { clickHistory, annotations, savedAt: new Date().toISOString() };
    const key = clickStorageKey(selectedSample, selectedPosition, selectedChannel, detectionMethod);
    const json = JSON.stringify(payload);
    void storage.set(key, json);
    scheduleExosomeKeysToDisk({ [key]: json });
  };

  const loadClickHistory = () => {
    const { selectedSample, selectedPosition, selectedChannel, detectionMethod } = state;
    void (async () => {
      const raw = await storage.get(clickStorageKey(selectedSample, selectedPosition, selectedChannel, detectionMethod));
      if (!raw) { alert('No saved click history for this sample/position/channel/method.'); return; }
      try {
        const { clickHistory, annotations } = JSON.parse(raw);
        setState(prev => ({ ...prev, clickHistory: clickHistory || [], annotations: annotations || [] }));
      } catch { alert('Failed to parse saved click history.'); }
    })();
  };

  const resetClickHistory = () => {
    const { selectedSample, selectedPosition, selectedChannel, detectionMethod } = state;
    void storage.remove(clickStorageKey(selectedSample, selectedPosition, selectedChannel, detectionMethod));
    setState(prev => ({ ...prev, clickHistory: [], annotations: [] }));
  };

  // Auto-save click history whenever it changes
  useEffect(() => {
    const { selectedSample, selectedPosition, selectedChannel, detectionMethod, clickHistory } = state;
    if (!selectedSample || !selectedPosition || !selectedChannel || clickHistory.length === 0) return;
    const payload = { clickHistory, annotations: state.annotations, savedAt: new Date().toISOString() };
    const key = clickStorageKey(selectedSample, selectedPosition, selectedChannel, detectionMethod);
    const json = JSON.stringify(payload);
    void storage.set(key, json);
    scheduleExosomeKeysToDisk({ [key]: json });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.clickHistory]);

  // Auto-restore click history when channel or method changes
  useEffect(() => {
    const { selectedSample, selectedPosition, selectedChannel, detectionMethod } = state;
    if (!selectedSample || !selectedPosition || !selectedChannel) return;
    let cancelled = false;
    const restoreClickHistory = async () => {
      const raw = await storage.get(clickStorageKey(selectedSample, selectedPosition, selectedChannel, detectionMethod));
      if (cancelled) return;
      if (!raw) {
        setState(prev => ({ ...prev, clickHistory: [], annotations: [] }));
        return;
      }
      try {
        const { clickHistory, annotations } = JSON.parse(raw);
        if (!cancelled) {
          setState(prev => ({ ...prev, clickHistory: clickHistory || [], annotations: annotations || [] }));
        }
      } catch {
        // ignore corrupt storage
      }
    };
    void restoreClickHistory();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.selectedChannel, state.detectionMethod]);

  // Update image when channel, display mode, or LUT changes
  useEffect(() => {
    if (!state.loaded || !state.selectedChannel) return;

    let cancelled = false;
    fetchChannelDisplay(
      getApiBase(),
      state.selectedSample,
      state.selectedPosition,
      state.selectedChannel,
      state.displayMode,
      state.displayLut,
      state.availableItems,
    ).then(result => {
      if (cancelled || !result) return;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        if (cancelled) return;
        const firstImageInSession = imageRef.current === null;
        setState(prev => ({
          ...prev,
          currentImageUrl: result.url,
          imageWidth: img.width,
          imageHeight: img.height,
          currentNormStats: result.normStats,
        }));
        imageRef.current = img;
        drawCanvas();
        // Avoid resetting zoom on channel / display / LUT changes; only fit when no image was shown yet.
        if (firstImageInSession) {
          requestAnimationFrame(() => {
            fitToViewport();
          });
        }
      };
      const url = result.url.startsWith('http') ? result.url : `${getApiBase()}${result.url}`;
      img.src = url;
    });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.selectedChannel, state.loaded, state.displayMode, state.displayLut]);

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

    // Guide from reference (C0) overlay for non-reference channels.
    if (showGuideFromRef && state.selectedChannel !== 'C0' && guideRefPoints.length > 0) {
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.95)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 3]);
      guideRefPoints.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
        ctx.stroke();
      });
      ctx.restore();
    }

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
    
    // Draw Random Forest annotations (fill only, no outline).
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

            offCtx.fillStyle = fillColor;
            ann.points.forEach(([x, y]) => {
              offCtx.beginPath();
              offCtx.arc(x, y, 2, 0, Math.PI * 2);
              offCtx.fill();
            });
          });

          ctx.save();
          ctx.globalAlpha = 0.45;
          ctx.drawImage(offscreen, 0, 0);
          ctx.restore();
        }
      }
      
      // Draw brush preview (semi-transparent circle following cursor; eraser = red outline only)
      if (state.brushPreview.x !== null && state.brushPreview.y !== null) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(state.brushPreview.x, state.brushPreview.y, state.brushSize, 0, Math.PI * 2);
        if (state.annotationMode === 'eraser') {
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.95)';
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          const previewColor = state.annotationMode === 'exosome'
            ? 'rgba(0, 255, 0, 0.4)'
            : 'rgba(255, 0, 0, 0.4)';
          ctx.fillStyle = previewColor;
          ctx.fill();
        }
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
      }
    }

    // Draw detections: use masks if available, otherwise use bboxes
    console.log('[FILTER DEBUG] showFilteredOnly=', showFilteredOnly, 'applyNonce=', filterApplyNonce, 'appliedFilterIds.size=', filteredDetectionIndexSet.size, 'totalDetections=', state.detections.length);
    // Render one group of detections (by original index) in a single color.
    // Handles both mask path and bbox fallback; works regardless of showMaskOutlines.
    const drawDetectionGroup = (indices: number[], color: string) => {
      if (indices.length === 0) return;
      const [cr, cg, cb] = hexToRgb(color);

      if (state.masks && state.masks.length > 0) {
        // state.masks[0] is a single combined binary mask; isolate each detection via its bbox.
        const combinedMask = state.masks[0];
        if (state.showMaskOutlines) {
          // Can't extract per-object contours from a combined mask — use bbox strokes.
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          indices.forEach(idx => {
            const det = state.detections[idx];
            if (!det) return;
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          });
          ctx.restore();
        } else {
          // Paint only the combined-mask pixels that fall within each detection's bbox.
          const imgData = new ImageData(canvas.width, canvas.height);
          const d = imgData.data;
          indices.forEach(idx => {
            const det = state.detections[idx];
            if (!det) return;
            const [x1, y1, x2, y2] = det.bbox;
            for (let y = Math.floor(y1); y < Math.ceil(y2); y++) {
              for (let x = Math.floor(x1); x < Math.ceil(x2); x++) {
                if (combinedMask[y]?.[x]) {
                  const i = (y * canvas.width + x) * 4;
                  d[i] = cr; d[i + 1] = cg; d[i + 2] = cb; d[i + 3] = 255;
                }
              }
            }
          });
          const off = document.createElement('canvas');
          off.width = canvas.width; off.height = canvas.height;
          off.getContext('2d')!.putImageData(imgData, 0, 0);
          ctx.save();
          ctx.globalAlpha = showFilteredOnly ? 1.0 : state.maskOpacity;
          ctx.drawImage(off, 0, 0);
          ctx.restore();
        }
      } else if (state.detections.length > 0) {
        ctx.save();
        ctx.globalAlpha = state.maskOpacity;
        if (state.showMaskOutlines) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          indices.forEach(idx => {
            const det = state.detections[idx];
            if (!det) return;
            const [x1, y1, x2, y2] = det.bbox;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          });
        } else {
          ctx.fillStyle = color;
          indices.forEach(idx => {
            const det = state.detections[idx];
            if (!det) return;
            const [x1, y1, x2, y2] = det.bbox;
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          });
        }
        ctx.restore();
      }
    };

    const allIndices = state.detections.map((_, i) => i);
    if (showFilteredOnly) {
      drawDetectionGroup(filteredOutIndices, '#ff4444');  // red: filtered-out
      drawDetectionGroup(filteredDetectionIndices, '#00ff00');  // green: filtered-in
    } else {
      drawDetectionGroup(allIndices, '#ff4444');  // red: all objects (no filter active)
    }

    // Highlight selected detection
    if (selectedDetectionRef.current !== null && state.detections[selectedDetectionRef.current]) {
      if (!showFilteredOnly || filteredDetectionIndexSet.has(selectedDetectionRef.current)) {
        const det = state.detections[selectedDetectionRef.current];
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 3;
        ctx.strokeRect(det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]);
      }
    }

    // Draw ground-truth overlay
    // Transform pipeline (applied in order):
    //   Step 1  swapXY   — swap X↔Y from CSV (diagnose row/col transposition)
    //   Step 2  yFlip    — y = imageH - y  (bottom-left → top-left origin)
    //   Step 3  offset   — add (gtOffsetX, gtOffsetY)  (crop / ROI origin shift)
    if (state.showGroundTruth && state.groundTruthPoints.length > 0) {
      ctx.save();

      const PIXEL_SIZE_UM = state.groundTruthPixelSizeUm;
      // imageH used for Y-flip; fall back to canvas height if backend didn't return it
      const imageH = state.gtImageHeight ?? canvas.height;
      const imageW = state.gtImageWidth  ?? canvas.width;

      // ── Build display points through the full transform chain ────────────
      const applyGtTransform = (p: { x: number; y: number }) => {
        let { x, y } = p;
        if (state.gtSwapXY)  { const tmp = x; x = y; y = tmp; }
        if (state.gtYFlip)   { y = imageH - y; }
        x += state.gtOffsetX;
        y += state.gtOffsetY;
        return { x, y };
      };

      const gtDisplayPoints = state.groundTruthPoints.map(applyGtTransform);

      // ── Build mapping-chain rows for the debug box ────────────────────────
      // We use rawGtSampleUm (raw CSV µm) so users can see the full chain.
      // If not available, reconstruct from the converted points.
      const makeMappingChain = (idx: number) => {
        const raw = state.rawGtSampleUm[idx];
        const orig = state.groundTruthPoints[idx];
        const display = gtDisplayPoints[idx];
        const x_um = raw ? raw.x_um : (orig.x * PIXEL_SIZE_UM);
        const y_um = raw ? raw.y_um : (orig.y * PIXEL_SIZE_UM);
        const x_px = raw ? raw.x_px : orig.x;
        const y_px = raw ? raw.y_px : orig.y;
        // after swapXY
        const [sx_px, sy_px] = state.gtSwapXY ? [y_px, x_px] : [x_px, y_px];
        // after yFlip
        const fy_px = state.gtYFlip ? imageH - sy_px : sy_px;
        return { x_um, y_um, x_px, y_px, sx_px, sy_px, fy_px, final_x: display.x, final_y: display.y };
      };

      if (state.gtDebugMode) {
        // ── Debug mode: large labeled magenta markers + full mapping chain ──

        // Console: full mapping chain for first 10 points
        console.group('[GT Debug] Full coordinate mapping chain');
        console.log(`Canvas: ${canvas.width}×${canvas.height} px`);
        console.log(`Image (from TIFF): ${imageW}×${imageH} px`);
        console.log(`pixel_size_um: ${PIXEL_SIZE_UM} µm/px`);
        console.log(`Transforms active: swapXY=${state.gtSwapXY} yFlip=${state.gtYFlip} offset=(${state.gtOffsetX},${state.gtOffsetY})`);
        console.table(
          state.groundTruthPoints.slice(0, 10).map((_, i) => {
            const c = makeMappingChain(i);
            return {
              'CSV X (µm)':   c.x_um.toFixed(4),
              'CSV Y (µm)':   c.y_um.toFixed(4),
              '÷ px_size → X_px': c.x_px.toFixed(2),
              '÷ px_size → Y_px': c.y_px.toFixed(2),
              'after swapXY X': c.sx_px.toFixed(2),
              'after swapXY Y': c.sy_px.toFixed(2),
              'after yFlip  Y': c.fy_px.toFixed(2),
              'final canvas X': c.final_x.toFixed(2),
              'final canvas Y': c.final_y.toFixed(2),
            };
          })
        );
        console.groupEnd();

        // Draw all GT points with large magenta cross+circle + index label
        const rBig = 10;
        gtDisplayPoints.forEach(({ x, y }, i) => {
          ctx.strokeStyle = 'rgba(255, 0, 255, 0.9)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, rBig, 0, Math.PI * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x - rBig, y); ctx.lineTo(x + rBig, y);
          ctx.moveTo(x, y - rBig); ctx.lineTo(x, y + rBig);
          ctx.stroke();
          if (i < 30) {
            ctx.fillStyle = 'yellow';
            ctx.font = 'bold 10px monospace';
            ctx.fillText(`${i}`, x + rBig + 2, y - 2);
          }
        });

        // ── Info box in top-left: full mapping chain for 10 sample points ──
        const lineH = 13;
        const cols = ['idx', 'CSV_X_µm', 'CSV_Y_µm', 'px', 'py', 'swX', 'swY', 'flipY', 'finX', 'finY'];
        const numRows = Math.min(10, state.groundTruthPoints.length);
        const boxW = 580;
        const boxH = lineH * (numRows + 3) + 6;
        const boxX = 5, boxY = 5;

        ctx.fillStyle = 'rgba(0,0,0,0.82)';
        ctx.fillRect(boxX, boxY, boxW, boxH);

        ctx.fillStyle = 'white';
        ctx.font = 'bold 10px monospace';
        const transforms = [
          state.gtSwapXY ? 'SWAP-XY' : '',
          state.gtYFlip  ? 'Y-FLIP' : '',
          (state.gtOffsetX || state.gtOffsetY) ? `OFS(${state.gtOffsetX},${state.gtOffsetY})` : '',
        ].filter(Boolean).join(' ');
        ctx.fillText(
          `GT Debug  imgSize=${imageW}×${imageH}  px_size=${PIXEL_SIZE_UM.toFixed(5)}µm  ${transforms || 'no transforms'}`,
          boxX + 4, boxY + lineH
        );

        ctx.font = '9px monospace';
        ctx.fillStyle = '#aaa';
        ctx.fillText(
          'idx  CSV_Xµm    CSV_Yµm    x_px     y_px   →swX   →swY   →flipY  finalX  finalY',
          boxX + 4, boxY + lineH * 2 + 2
        );

        for (let i = 0; i < numRows; i++) {
          const c = makeMappingChain(i);
          const row = [
            String(i).padStart(3),
            c.x_um.toFixed(2).padStart(9),
            c.y_um.toFixed(2).padStart(9),
            c.x_px.toFixed(1).padStart(7),
            c.y_px.toFixed(1).padStart(7),
            c.sx_px.toFixed(1).padStart(7),
            c.sy_px.toFixed(1).padStart(7),
            c.fy_px.toFixed(1).padStart(8),
            c.final_x.toFixed(1).padStart(7),
            c.final_y.toFixed(1).padStart(7),
          ].join('  ');
          ctx.fillStyle = i % 2 === 0 ? 'cyan' : '#88eeff';
          ctx.fillText(row, boxX + 4, boxY + lineH * (i + 3) + 4);
        }

      } else {
        // ── Normal mode: circle-only (no crosshair) ──
        ctx.strokeStyle = 'rgba(0, 230, 255, 0.85)';
        ctx.lineWidth = 1.5;
        const r = 5;
        gtDisplayPoints.forEach(({ x, y }) => {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.stroke();
        });
      }
      ctx.restore();
    }

    // Draw detection count
    if (state.detections.length > 0 || (state.showGroundTruth && state.groundTruthPoints.length > 0)) {
      const lines: string[] = [];
      if (state.detections.length > 0) lines.push(`Detected: ${state.detections.length}`);
      if (state.showGroundTruth && state.groundTruthPoints.length > 0) lines.push(`GT: ${state.groundTruthPoints.length}`);
      const boxH = lines.length * 20 + 10;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(10, 10, 200, boxH);
      ctx.fillStyle = '#ffffff';
      ctx.font = '14px Arial';
      lines.forEach((line, i) => ctx.fillText(line, 15, 28 + i * 20));
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
  }, [state.boxPrompt, state.pointPrompts, state.masks, state.maskOpacity, state.showMaskOutlines, state.detections, state.annotations, state.probabilityMap, state.showConfidenceMap, state.detectionMethod, state.calibrationMode, state.debugLogs, state.brushPreview, state.brushSize, state.annotationMode, state.showGroundTruth, state.groundTruthPoints, state.groundTruthPixelSizeUm, state.gtImageWidth, state.gtImageHeight, state.rawGtSampleUm, state.gtDebugMode, state.gtSwapXY, state.gtYFlip, state.gtOffsetX, state.gtOffsetY, zoomState, showFilteredOnly, filteredDetectionIndices, filteredOutIndices, filterApplyNonce, showGuideFromRef, guideRefPoints, state.selectedChannel]);

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
    const newScale = Math.max(0.05, Math.min(8.0, zoomState.scale * zoomFactor));
    
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
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        setState(prev => ({
          ...prev,
          annotationMode: prev.annotationMode === 'eraser' ? 'exosome' : 'eraser',
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
    // 4. Shift + right-click drag
    const isShiftRightDrag = e.button === 2 && e.shiftKey;
    const canPan = zoomState.scale > 1.0 && 
                   !isAnnotatingRef.current && 
                   !isDrawingRef.current &&
                   isShiftRightDrag;
    
    if (canPan) {
      e.preventDefault();
      e.stopPropagation();
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
    const panStart = panStartRef.current;
    const viewport = viewportRef.current;
    if (isPanningRef.current && panStart && viewport) {
      e.preventDefault();
      e.stopPropagation();
      const rect = viewport.getBoundingClientRect();
      setZoomState(prev => ({
        ...prev,
        offsetX: e.clientX - rect.left - panStart.x,
        offsetY: e.clientY - rect.top - panStart.y,
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
  
  // Double-click: fit image to viewport
  const handleDoubleClick = () => {
    fitToViewport();
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

  /** Remove annotation points within brush radius of (cx, cy) from all groups (Random Forest eraser). */
  const eraseBrushAt = useCallback((cx: number, cy: number) => {
    const r = state.brushSize;
    const r2 = r * r;
    setState(prev => {
      const nextAnnotations = prev.annotations
        .map(ann => ({
          ...ann,
          points: ann.points.filter(([px, py]) => {
            const dx = px - cx;
            const dy = py - cy;
            return dx * dx + dy * dy > r2;
          }),
        }))
        .filter(ann => ann.points.length > 0);
      return { ...prev, annotations: nextAnnotations };
    });
    drawCanvas();
  }, [state.brushSize, drawCanvas]);
  
  // Canvas mouse handlers for box/point prompts (SAM only) and brush annotation (Random Forest)
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    // Ignore middle-click entirely.
    if (e.button === 1) return;

    const { x, y } = getCanvasCoords(e);

    if (state.detectionMethod === 'random_forest') {
      if (state.annotationMode === 'eraser') {
        isAnnotatingRef.current = true;
        currentAnnotationPointsRef.current = [];
        currentAnnotationIdRef.current = null;
        eraseBrushAt(x, y);
        return;
      }

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

      // Right-click does not emit the browser click event, so record history here.
      if (e.button === 2 && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        let intensity: number | null = null;
        if (ctx) {
          try {
            const pixel = ctx.getImageData(Math.round(x), Math.round(y), 1, 1);
            intensity = pixel.data[0] ?? null;
          } catch {
            intensity = null;
          }
        }
        const clickId = `click_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const clickEntry = {
          id: clickId,
          timestamp: Date.now(),
          imageX: Math.round(x),
          imageY: Math.round(y),
          intensity,
          confidence: null,
          predictedClass: null,
          scale: zoomState.scale,
          offsetX: zoomState.offsetX,
          offsetY: zoomState.offsetY,
          annotationId: annId,
          clickType: 'background' as const,
        };
        setState(prev => ({
          ...prev,
          clickHistory: [clickEntry, ...prev.clickHistory].slice(0, 100),
        }));
      }
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
      if (state.annotationMode === 'eraser') {
        if (e.buttons & 1 || e.buttons & 2) {
          eraseBrushAt(x, y);
        }
      } else {
        // Continue brush stroke with the same annotation ID
        const label = (e.buttons & 1)  // bitwise: left button held (handles left+middle = 5 too)
          ? (state.annotationMode === 'exosome' ? 1 : 0)
          : (state.annotationMode === 'exosome' ? 0 : 1);
        addBrushPoint(x, y, label, currentAnnotationIdRef.current || undefined);
      }
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
      clickType: (() => {
        if (annotationId) {
          const ann = state.annotations.find((a) => a.id === annotationId);
          if (ann?.label === 0) return 'background' as const;
          if (ann?.label === 1) return 'exosome' as const;
        }
        return (predictedClass === 'Background' ? 'background' : 'exosome') as const;
      })(),
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

    // ── GT nearest-point debug (always active when GT is loaded) ──
    if (state.groundTruthPoints.length > 0) {
      const PIXEL_SIZE_UM = state.groundTruthPixelSizeUm;
      const canvasEl = canvasRef.current!;
      const canvasRect = canvasEl.getBoundingClientRect();
      const cssToInternal = canvasEl.width / canvasRect.width;
      const imageH = state.gtImageHeight ?? canvasEl.height;
      const imageW = state.gtImageWidth  ?? canvasEl.width;

      // Apply the same full transform chain as drawCanvas
      const applyGtTransform = (p: { x: number; y: number }) => {
        let { x: px, y: py } = p;
        if (state.gtSwapXY) { const tmp = px; px = py; py = tmp; }
        if (state.gtYFlip)  { py = imageH - py; }
        px += state.gtOffsetX;
        py += state.gtOffsetY;
        return { x: px, y: py };
      };

      const gtPtsForNearest = state.groundTruthPoints.map(applyGtTransform);

      let nearestDist = Infinity;
      let nearestIdx = -1;
      gtPtsForNearest.forEach((pt, i) => {
        const d = Math.hypot(pt.x - x, pt.y - y);
        if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
      });

      const nearest       = gtPtsForNearest[nearestIdx];
      const nearestRaw    = state.groundTruthPoints[nearestIdx];
      const nearestSample = state.rawGtSampleUm[nearestIdx];

      console.group('[GT Click Debug] — click on image');
      console.log('── Click coordinates ──────────────────────────────');
      console.log('  Clicked CSS canvas:', {
        cssX: (e.clientX - canvasRect.left).toFixed(1),
        cssY: (e.clientY - canvasRect.top).toFixed(1),
      });
      console.log('  CSS→Internal ratio:', cssToInternal.toFixed(4));
      console.log('  Mapped image pixel coord:', { x: x.toFixed(2), y: y.toFixed(2) });
      console.log('── Canvas / image metadata ────────────────────────');
      console.log('  Canvas internal:', canvasEl.width, '×', canvasEl.height, 'px');
      console.log('  Canvas CSS:    ', canvasRect.width.toFixed(1), '×', canvasRect.height.toFixed(1), 'px');
      console.log('  Image (TIFF):  ', imageW, '×', imageH, 'px');
      console.log('  pixel_size_um: ', PIXEL_SIZE_UM, 'µm/px');
      console.log('── Active GT transforms ───────────────────────────');
      console.log('  gtSwapXY:', state.gtSwapXY,
                  ' gtYFlip:', state.gtYFlip,
                  ' gtOffset:', `(${state.gtOffsetX}, ${state.gtOffsetY})`);
      console.log('── Nearest GT point ───────────────────────────────');
      console.log('  Index:', nearestIdx);
      if (nearestSample) {
        console.log('  CSV raw:       ', `X=${nearestSample.x_um}µm  Y=${nearestSample.y_um}µm`);
        console.log('  ÷ pixel_size → ', `X=${nearestSample.x_px}px  Y=${nearestSample.y_px}px`);
      } else {
        console.log('  ÷ pixel_size → ', `X=${nearestRaw.x.toFixed(2)}px  Y=${nearestRaw.y.toFixed(2)}px`);
      }
      if (state.gtSwapXY) console.log('  after swapXY: ', `X=${nearestRaw.y.toFixed(2)}px  Y=${nearestRaw.x.toFixed(2)}px`);
      if (state.gtYFlip)  console.log('  after yFlip:  ', `Y=${(imageH - (state.gtSwapXY ? nearestRaw.x : nearestRaw.y)).toFixed(2)}px`);
      console.log('  Final display: ', `X=${nearest.x.toFixed(2)}px  Y=${nearest.y.toFixed(2)}px`);
      console.log('  Distance:      ', nearestDist.toFixed(2), 'px =', (nearestDist * PIXEL_SIZE_UM).toFixed(3), 'µm');
      console.groupEnd();
    }
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

  const refreshRfModelList = useCallback(async () => {
    if (!state.selectedSample || !state.selectedChannel) {
      setState(prev => ({ ...prev, availableRfModels: [], selectedRfModelSourcePosition: '' }));
      return;
    }
    try {
      const response = await fetch(
        `${getApiBase()}/api/exosome/rf_model/list?sample=${encodeURIComponent(state.selectedSample)}&channel=${encodeURIComponent(state.selectedChannel)}`
      );
      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to load saved models');
      }
      const models: SavedRfModelItem[] = Array.isArray(data.data) ? data.data : [];
      setState(prev => ({
        ...prev,
        availableRfModels: models,
        selectedRfModelSourcePosition: models.some(m => m.position === prev.selectedRfModelSourcePosition)
          ? prev.selectedRfModelSourcePosition
          : (models[0]?.position || ''),
      }));
    } catch (err: any) {
      setState(prev => ({ ...prev, rfModelStatus: `Failed to list RF models: ${err.message}` }));
    }
  }, [state.selectedSample, state.selectedChannel]);

  const handleSaveRfModel = async () => {
    if (!state.loaded || !state.selectedSample || !state.selectedPosition || !state.selectedChannel) {
      alert('Please load an image first');
      return;
    }
    if (state.annotations.length === 0) {
      alert('Please annotate pixels before saving model');
      return;
    }
    try {
      setState(prev => ({ ...prev, rfModelStatus: 'Saving RF model...', rfModelWarning: null }));
      const response = await fetch(`${getApiBase()}/api/exosome/rf_model/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: state.selectedSample,
          position: state.selectedPosition,
          channel: state.selectedChannel,
          annotations: state.annotations,
        }),
      });
      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to save RF model');
      }
      setState(prev => ({
        ...prev,
        rfModelStatus: `RF model saved: ${data.saved_path}`,
        rfModelSavedPath: data.saved_path || null,
      }));
      await refreshRfModelList();
    } catch (err: any) {
      setState(prev => ({ ...prev, rfModelStatus: `Failed to save RF model: ${err.message}` }));
      alert(`Failed to save RF model: ${err.message}`);
    }
  };

  const handleLoadRfModelAndSegment = async () => {
    if (!state.loaded || !state.selectedSample || !state.selectedPosition || !state.selectedChannel) {
      alert('Please load an image first');
      return;
    }
    if (!state.selectedRfModelSourcePosition) {
      alert('Please select a saved model position');
      return;
    }
    setState(prev => ({
      ...prev,
      isDetecting: true,
      exportStatus: null,
      rfModelStatus: `Running segmentation with model from ${state.selectedRfModelSourcePosition}...`,
      rfModelWarning: null,
    }));

    try {
      const response = await fetch(`${getApiBase()}/api/exosome/rf_model/load_and_segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: state.selectedSample,
          position: state.selectedPosition,
          channel: state.selectedChannel,
          model_source_position: state.selectedRfModelSourcePosition,
          confidence_threshold: state.confidenceThreshold,
          min_area: state.minArea,
          apply_morphology: state.fillHoles,
        }),
      });
      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'RF model segmentation failed');
      }

      const masksRaw = data.data.masks || [];
      const scores = data.data.scores || [];
      const detections: DetectionResult[] = (data.data.detections || []).map((d: any, idx: number) => ({
        id: idx + 1,
        area: d.area || 0,
        centroid: d.centroid || [0, 0],
        bbox: d.bbox || [0, 0, 0, 0],
        score: idx < scores.length ? scores[idx] : undefined,
        perimeter: d.perimeter != null ? d.perimeter : undefined,
        circularity: d.circularity != null ? d.circularity : undefined,
      }));

      setDetectionBatchId(b => b + 1);
      setState(prev => ({
        ...prev,
        masks: masksRaw.length > 0 ? masksRaw : null,
        scores,
        detections,
        probabilityMap: data.data.probability_map || null,
        isDetecting: false,
        rfModelStatus: `Loaded model from ${state.selectedRfModelSourcePosition} and segmented ${state.selectedPosition}`,
        rfModelWarning: data.model_warning || data.data._model_warning || null,
      }));
      drawCanvas();
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        isDetecting: false,
        rfModelStatus: `Failed to run loaded model: ${err.message}`,
      }));
      alert(`RF model load-and-segment failed: ${err.message}`);
    }
  };

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
      
      const response = await fetch(`${getApiBase()}/api/exosome/segment`, {
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
        perimeter: d.perimeter != null ? d.perimeter : undefined,
        circularity: d.circularity != null ? d.circularity : undefined,
      }));

      console.log('[ExosomeDetection] Processed detections:', detections.length);

      // Handle Random Forest specific results
      let probabilityMap = null;
      if (state.detectionMethod === 'random_forest' && data.data.probability_map) {
        probabilityMap = data.data.probability_map;
        console.log('[ExosomeDetection] Received probability map');
      }

      setDetectionBatchId(b => b + 1);
      setState(prev => ({
        ...prev,
        masks: masks.length > 0 ? masks : null, // null if masks omitted
        scores: scores,
        detections: detections,
        probabilityMap: probabilityMap ?? null,
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

  useEffect(() => {
    if (state.detectionMethod !== 'random_forest') return;
    refreshRfModelList();
  }, [state.detectionMethod, state.selectedSample, state.selectedChannel, refreshRfModelList]);

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
    setDetectionBatchId(b => b + 1);
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
      const response = await fetch(`${getApiBase()}/api/exosome/export`, {
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
                onChange={(e) => setState(prev => ({
                  ...prev,
                  selectedSample: e.target.value,
                  availableRfModels: [],
                  selectedRfModelSourcePosition: '',
                  rfModelStatus: null,
                  rfModelWarning: null,
                }))}
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
                onChange={(e) => setState(prev => ({
                  ...prev,
                  selectedPosition: e.target.value,
                  availableRfModels: [],
                  selectedRfModelSourcePosition: '',
                  rfModelStatus: null,
                  rfModelWarning: null,
                }))}
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
                onChange={(e) => {
                  // Persist current channel annotations/history before switching context.
                  saveClickHistory();
                  // Clear previous-channel segmentation artifacts on channel switch.
                  setFilteredDetectionIndices([]);
                  setState(prev => ({
                    ...prev,
                    selectedChannel: e.target.value,
                    masks: null,
                    scores: null,
                    detections: [],
                    probabilityMap: null,
                    boxPrompt: null,
                    pointPrompts: [],
                    availableRfModels: [],
                    selectedRfModelSourcePosition: '',
                    rfModelStatus: null,
                    rfModelWarning: null,
                  }));
                }}
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

          {/* Image Source & Display Pipeline Debug Info */}
          {state.loaded && (() => {
            const selItem = state.availableItems.find(i => i.key === state.selectedChannel);
            // Prefer currentNormStats (from most-recent display refresh); fall back to item stats
            const ns = state.currentNormStats || selItem?.norm_stats;
            return (
              <div className="control-section" style={{ fontSize: '0.73rem', lineHeight: 1.6 }}>
                <h3 style={{ fontSize: '0.8rem', marginBottom: 4 }}>Image Info</h3>

                {/* Load mode */}
                <div style={{ color: state.cropMode ? '#27ae60' : '#555', fontWeight: 600, marginBottom: 2 }}>
                  {state.cropMode ? '🟢 Crop mode (multi-ch TIFF)' : '⬜ Normal (per-channel TIFFs)'}
                </div>

                <div><strong>Sample:</strong> {state.selectedSample}</div>
                <div><strong>Position:</strong> {state.selectedPosition}</div>

                {state.cropMode && (
                  <>
                    <div><strong>TIFF:</strong> {state.cropTiffFile}</div>
                    <div>
                      <strong>Stack shape:</strong>{' '}
                      {state.cropTiffShape ? `[${state.cropTiffShape.join('×')}]` : '–'}
                      {state.cropTiffAxes ? ` (${state.cropTiffAxes})` : ''}
                    </div>
                    <div><strong>Channels:</strong> {state.cropNumChannels ?? '–'}</div>
                    <div>
                      <strong>Pixel size:</strong>{' '}
                      {state.cropPixelSizeUm != null ? `${state.cropPixelSizeUm.toFixed(4)} µm/px` : '–'}
                      {state.cropPixelSizeSource
                        ? <span style={{ color: state.cropPixelSizeSource === 'fallback' ? '#e67e22' : '#27ae60', marginLeft: 4 }}>
                            ({state.cropPixelSizeSource})
                          </span>
                        : null}
                    </div>
                  </>
                )}

                <div style={{ marginTop: 6, borderTop: '1px solid #ddd', paddingTop: 4 }}>
                  <strong>Selected channel:</strong> {state.selectedChannel || '–'}
                  {selItem?.label ? (
                    <div style={{ color: '#888', fontSize: '0.68rem' }}>{selItem.label.split(' aligned')[0]}</div>
                  ) : null}
                </div>

                {/* Display pipeline box */}
                {ns && (
                  <div style={{
                    marginTop: 6, background: '#f0f4ff', border: '1px solid #b0c4de',
                    borderRadius: 4, padding: '4px 6px',
                  }}>
                    <div style={{ fontWeight: 700, color: '#2c3e50', marginBottom: 2 }}>📊 Display pipeline</div>
                    <div>
                      <span style={{ color: '#555' }}>mode:</span>{' '}
                      <strong style={{ color: state.displayMode === 'raw_16bit' ? '#1565c0' : state.displayMode === 'enhanced' ? '#e65100' : '#4a148c' }}>
                        {state.displayMode === 'raw_16bit' ? 'ImageJ-like raw 16-bit' :
                         state.displayMode === 'enhanced'  ? 'Enhanced (p0.5–p99.5)' :
                         'Raw min→max'}
                      </strong>
                      {state.displayMode === 'raw_16bit' && (
                        <span style={{ fontSize: '0.68rem', color: '#27ae60', marginLeft: 4 }}>(default)</span>
                      )}
                    </div>
                    <div>
                      <span style={{ color: '#555' }}>LUT:</span>{' '}
                      <strong style={{ color: state.displayLut === 'red' ? '#c0392b' : '#333' }}>
                        {state.displayLut === 'red' ? 'Red (ImageJ-like)' : 'Grayscale'}
                      </strong>
                    </div>
                    <div><span style={{ color: '#555' }}>dtype:</span> <strong>{ns.dtype}</strong></div>
                    <div><span style={{ color: '#555' }}>raw range:</span> [{ns.original_min?.toFixed(0)}, {ns.original_max?.toFixed(0)}]</div>
                    <div><span style={{ color: '#555' }}>p0.5 / p99.5:</span> {ns.p0_5?.toFixed(0)} / {ns.p99_5?.toFixed(0)}</div>
                    <div>
                      <span style={{ color: '#555' }}>display stretch:</span>{' '}
                      <strong>[{ns.display_min?.toFixed(0)}, {ns.display_max?.toFixed(0)}]</strong>{' '}
                      → 8-bit
                    </div>
                    <div><span style={{ color: '#555' }}>method:</span> {ns.normalization}</div>
                    {ns.cache_hit != null && (
                      <div>
                        <span style={{ color: '#555' }}>preview cache:</span>{' '}
                        <span style={{ color: ns.cache_hit ? '#888' : '#27ae60' }}>
                          {ns.cache_hit ? 'reused' : 'generated'}
                        </span>
                      </div>
                    )}
                    {state.displayMode !== 'raw_16bit' && (
                      <div style={{ marginTop: 4, color: '#e65100', fontSize: '0.68rem' }}>
                        ⚠ Not default — switch to "ImageJ-like" for faithful comparison
                      </div>
                    )}
                    {state.displayMode === 'raw_16bit' && (
                      <div style={{ marginTop: 4, color: '#1565c0', fontSize: '0.68rem' }}>
                        ✓ Raw 16-bit linear — matches ImageJ default display
                      </div>
                    )}
                  </div>
                )}

                {/* Display vs Detection source */}
                {selItem && (
                  <div style={{
                    marginTop: 6, background: '#fff8e1', border: '1px solid #ffe082',
                    borderRadius: 4, padding: '4px 6px',
                  }}>
                    <div style={{ fontWeight: 700, color: '#795548', marginBottom: 2 }}>🔍 Display vs Detection</div>
                    <div>
                      <span style={{ color: '#555' }}>Display:</span>{' '}
                      <span title={selItem.display_source} style={{ color: '#1565c0', wordBreak: 'break-all' }}>
                        preview PNG (8-bit, stretched)
                      </span>
                    </div>
                    <div>
                      <span style={{ color: '#555' }}>Detection:</span>{' '}
                      <span style={{ color: '#2e7d32' }}>
                        {state.cropMode ? 'raw TIFF slice (16-bit)' : 'raw TIFF (16-bit)'}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#888', marginTop: 2 }}>
                      RF/SAM/Blob use the raw 16-bit data, not the display image.
                    </div>
                  </div>
                )}

                {state.groundTruthPixelSizeUm ? (
                  <div style={{ marginTop: 4 }}>
                    <strong>GT pixel size:</strong> {state.groundTruthPixelSizeUm.toFixed(4)} µm/px
                  </div>
                ) : null}
              </div>
            );
          })()}

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
                    onChange={(e) => setState(prev => ({ ...prev, annotationMode: e.target.value as 'exosome' | 'background' | 'eraser' }))}
                  />
                  Exosome (Left Click)
                </label>
                <label>
                  <input
                    type="radio"
                    value="background"
                    checked={state.annotationMode === 'background'}
                    onChange={(e) => setState(prev => ({ ...prev, annotationMode: e.target.value as 'exosome' | 'background' | 'eraser' }))}
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
                Shortcuts: <b>[</b> / <b>]</b> brush size, <b>E</b> eraser toggle, or Shift + Mouse Wheel
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
            {state.detectionMethod === 'random_forest' && (
              <>
                <button
                  onClick={handleSaveRfModel}
                  disabled={!state.loaded || state.isDetecting || state.annotations.length === 0}
                  style={{ marginTop: '0.4rem' }}
                  title="Train and save RF model for this sample/position/channel"
                >
                  Save Model
                </button>
                <div style={{ marginTop: '0.6rem', border: '1px solid #ddd', borderRadius: 6, padding: '0.55rem' }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>Load Model from Another Position</div>
                  <select
                    value={state.selectedRfModelSourcePosition}
                    onChange={(e) => setState(prev => ({ ...prev, selectedRfModelSourcePosition: e.target.value }))}
                    style={{ width: '100%', marginBottom: 6 }}
                    disabled={!state.selectedSample || !state.selectedChannel}
                  >
                    <option value="">-- Select saved RF model --</option>
                    {state.availableRfModels.map((model) => (
                      <option key={model.path} value={model.position}>
                        {`${model.position} | ${model.saved_at ? new Date(model.saved_at).toLocaleString() : 'unknown time'} | ${model.trained_on?.position || model.position}`}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={handleLoadRfModelAndSegment}
                    disabled={!state.selectedRfModelSourcePosition || state.isDetecting}
                    style={{ width: '100%' }}
                  >
                    Segment with Loaded Model
                  </button>
                </div>
                {state.rfModelStatus && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#1a237e', wordBreak: 'break-all' }}>
                    {state.rfModelStatus}
                  </div>
                )}
                {state.rfModelSavedPath && (
                  <div style={{ marginTop: '0.25rem', fontSize: '0.75rem', color: '#555', wordBreak: 'break-all' }}>
                    Saved path: {state.rfModelSavedPath}
                  </div>
                )}
                {state.rfModelWarning && (
                  <div style={{
                    marginTop: '0.5rem',
                    background: '#fff8e1',
                    border: '1px solid #fbc02d',
                    borderRadius: 4,
                    color: '#8d6e00',
                    padding: '0.5rem',
                    fontSize: '0.8rem',
                  }}>
                    {state.rfModelWarning}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* CENTER: Image Viewer */}
        <div className="exosome-viewer">
          <div className="viewer-controls">
            {/* ── Exosome-Detection Display Mode (does NOT affect detection) ── */}
            <div style={{
              background: '#f0f4ff', border: '1px solid #b0c4de',
              borderRadius: 6, padding: '6px 10px', marginBottom: 8,
            }}>
              <div style={{ fontWeight: 700, fontSize: '0.78rem', color: '#1a237e', marginBottom: 4 }}>
                🎨 Display Mode
                <span style={{ fontWeight: 400, color: '#888', marginLeft: 6, fontSize: '0.68rem' }}>
                  (visualisation only — detection unaffected)
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {(
                  [
                    ['raw_16bit', 'ImageJ-like raw 16-bit', '(default — arr/65535×255, dark)'],
                    ['enhanced',  'Enhanced stretch',       '(p0.5–p99.5, bright)'],
                    ['minmax',    'Raw min→max',            '(arr.min→arr.max)'],
                  ] as [string, string, string][]
                ).map(([val, label, hint]) => (
                  <label key={val} style={{ display: 'flex', alignItems: 'baseline', gap: 5, cursor: 'pointer', fontSize: '0.78rem' }}>
                    <input
                      type="radio"
                      name="exo-display-mode"
                      value={val}
                      checked={state.displayMode === val}
                      onChange={() => setState(prev => ({ ...prev, displayMode: val as 'raw_16bit' | 'enhanced' | 'minmax' }))}
                    />
                    <span style={{ fontWeight: state.displayMode === val ? 700 : 400 }}>{label}</span>
                    {state.displayMode === val && (
                      <span style={{ color: '#555', fontSize: '0.68rem' }}>{hint}</span>
                    )}
                    {val === 'raw_16bit' && state.displayMode !== val && (
                      <span style={{ color: '#888', fontSize: '0.68rem' }}>(default)</span>
                    )}
                  </label>
                ))}
              </div>
              {/* LUT toggle */}
              <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#333' }}>LUT:</span>
                {(['gray', 'red'] as const).map(l => (
                  <label key={l} style={{ fontSize: '0.75rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3 }}>
                    <input
                      type="radio"
                      name="exo-lut"
                      value={l}
                      checked={state.displayLut === l}
                      onChange={() => setState(prev => ({ ...prev, displayLut: l }))}
                    />
                    <span style={{ color: l === 'red' ? '#c0392b' : '#333' }}>
                      {l === 'gray' ? 'Grayscale' : 'Red (ImageJ-like)'}
                    </span>
                  </label>
                ))}
              </div>
            </div>

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
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={showFilteredOnly}
                  onChange={(e) => setShowFilteredOnly(e.target.checked)}
                  disabled={state.detections.length === 0}
                />
                Show Filtered Only
                {state.detections.length > 0 && (
                  <span style={{ fontSize: '0.75rem', color: '#666', marginLeft: 6 }}>
                    ({filteredDetectionIndices.length} / {state.detections.length})
                  </span>
                )}
              </label>
            </div>
            <div className="input-group">
              <label title={
                state.selectedChannel === 'C0'
                  ? 'Switch to another channel to use this guide'
                  : (!guideRefAvailable ? 'Save filtered detections from C0 first' : '')
              }>
                <input
                  type="checkbox"
                  checked={showGuideFromRef}
                  onChange={(e) => setShowGuideFromRef(e.target.checked)}
                  disabled={state.selectedChannel === 'C0' || !guideRefAvailable}
                />
                Guide from Ref
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
            <div className="input-group">
              <label>
                <input
                  type="checkbox"
                  checked={state.showGroundTruth}
                  onChange={(e) => setState(prev => ({ ...prev, showGroundTruth: e.target.checked }))}
                  disabled={state.groundTruthPoints.length === 0}
                />
                Show Ground Truth
                {state.groundTruthFile && (
                  <span style={{ fontSize: '0.75rem', color: '#666', marginLeft: 6 }}>
                    ({state.groundTruthCount} pts)
                  </span>
                )}
                {!state.groundTruthFile && state.loaded && (
                  <span style={{ fontSize: '0.75rem', color: '#aaa', marginLeft: 6 }}>
                    (no CSV found)
                  </span>
                )}
              </label>
            </div>
            {state.showGroundTruth && state.groundTruthPoints.length > 0 && (
              <div className="input-group" style={{ marginLeft: 16 }}>
                <label>
                  <input
                    type="checkbox"
                    checked={state.gtDebugMode}
                    onChange={(e) => setState(prev => ({ ...prev, gtDebugMode: e.target.checked }))}
                  />
                  <span style={{ fontSize: '0.8rem', color: '#c0392b', fontWeight: 600 }}> GT Debug Mode</span>
                  <span style={{ fontSize: '0.72rem', color: '#888', marginLeft: 4 }}>
                    (magenta markers + labels + console log)
                  </span>
                </label>
                <div style={{ fontSize: '0.72rem', color: '#888', marginTop: 2, marginLeft: 20 }}>
                  Click on any spot → nearest GT printed to console
                </div>
                {(state.gtImageHeight != null) && (
                  <div style={{ fontSize: '0.72rem', color: '#888', marginTop: 4, marginLeft: 2 }}>
                    TIFF: {state.gtImageWidth}×{state.gtImageHeight} px
                    &nbsp;|&nbsp; px_size: {state.groundTruthPixelSizeUm.toFixed(5)} µm/px
                  </div>
                )}
              </div>
            )}
            {state.showGroundTruth && state.groundTruthPoints.length > 0 && state.detections.length > 0 && (() => {
              // Match stats using the same transform applied in drawCanvas
              const imgH = state.gtImageHeight ?? 0;
              const applyT = (p: { x: number; y: number }) => {
                let { x, y } = p;
                if (state.gtSwapXY) { const t = x; x = y; y = t; }
                if (state.gtYFlip)  y = imgH - y;
                x += state.gtOffsetX;
                y += state.gtOffsetY;
                return { x, y };
              };
              const transformedGt = state.groundTruthPoints.map(applyT);
              const TOL = 10;
              let matched = 0;
              transformedGt.forEach(gt => {
                const hit = state.detections.some(d => {
                  const cx = (d.bbox[0] + d.bbox[2]) / 2;
                  const cy = (d.bbox[1] + d.bbox[3]) / 2;
                  return Math.hypot(cx - gt.x, cy - gt.y) < TOL;
                });
                if (hit) matched++;
              });
              const fp = state.detections.length - matched;
              const missed = state.groundTruthPoints.length - matched;
              const pct = ((matched / state.groundTruthPoints.length) * 100).toFixed(1);
              return (
                <div style={{ background: '#e8f5e9', padding: '6px 8px', borderRadius: 4,
                  fontSize: '0.78rem', lineHeight: 1.7, marginTop: 4 }}>
                  <strong>GT Stats</strong><br />
                  GT: {state.groundTruthPoints.length} &nbsp;|&nbsp; Detected: {state.detections.length}<br />
                  Matched: {matched} ({pct}%)<br />
                  False positives: {fp} &nbsp;|&nbsp; Missed: {missed}
                </div>
              );
            })()}
          </div>
          {/* Image source label — clarifies what is displayed vs what is used for detection */}
          {state.loaded && (
            <div style={{
              fontSize: '0.72rem', color: '#888', padding: '2px 4px',
              borderTop: '1px solid #eee', background: '#fafafa',
            }}>
              Display: raw image (percentile-normalized for browser) &nbsp;|&nbsp;
              Detection: raw TIFF
              {state.groundTruthFile && (
                <>&nbsp;|&nbsp; GT pixel size: {state.groundTruthPixelSizeUm.toFixed(5)} µm/px</>
              )}
            </div>
          )}
          {/* Viewport: Single source of truth for pointer events */}
          <div
            ref={viewportRef}
            className="canvas-container"
            onPointerDown={(e) => {
              // Ignore middle-click entirely to avoid browser auto-scroll conflicts.
              if (e.button === 1) return;
              // Shift + right-click drag starts pan mode.
              if (e.button === 2 && e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                handlePanStart(e);
                return;
              }
              // All other pointer downs are handled by the canvas (annotation/drawing).
              // Regular right-click remains unaffected.
              handleCanvasMouseDown(e as any);
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
            onPointerCancel={() => {
              handlePanEnd();
              handleCanvasMouseUp();
            }}
            onContextMenu={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onDoubleClick={handleDoubleClick}
            style={{
              position: 'relative',
              overflow: 'hidden',
              width: '100%',
              height: '100%',
              cursor: isPanningRef.current
                ? 'grabbing'
                : (zoomState.scale > 1.0 && !isAnnotatingRef.current && !isDrawingRef.current ? 'grab' : 'default'),
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
                    onContextMenu={(e) => e.preventDefault()}
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
              <DetectionTable
                detections={state.detections}
                selectedIdx={selectedDetectionRef.current}
                onRowClick={(idx) => { selectedDetectionRef.current = idx; drawCanvas(); }}
                filterArea={filterArea}
                filterCirc={filterCirc}
                onFilterAreaChange={setFilterArea}
                onFilterCircChange={setFilterCirc}
                onFilteredIndicesChange={handleFilteredIndicesChange}
                onApplyFilters={handleApplyFilters}
                onSaveFilter={handleSaveFilter}
                saveMessage={saveFilterMessage}
              />

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
                    onClick={async () => {
                      const json = JSON.stringify(state.debugLogs, null, 2);
                      const copied = await copyText(json);
                      alert(copied ? 'Debug logs copied to clipboard' : 'Failed to copy debug logs to clipboard');
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
                <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                  <button
                    onClick={saveClickHistory}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', background: '#1976d2', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer' }}
                    title="Save click history to localStorage"
                  >Save</button>
                  <button
                    onClick={loadClickHistory}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', cursor: 'pointer' }}
                    title="Load click history from localStorage"
                  >Load</button>
                  <button
                    onClick={resetClickHistory}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', background: '#fff3e0', color: '#b71c1c', cursor: 'pointer' }}
                    title="Clear history and remove from localStorage"
                  >Reset</button>
                  <button
                    onClick={async () => {
                      const json = JSON.stringify(state.clickHistory, null, 2);
                      const copied = await copyText(json);
                      alert(copied ? 'Click history copied to clipboard as JSON' : 'Failed to copy click history JSON');
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                  >
                    JSON
                  </button>
                  <button
                    onClick={async () => {
                      const csv = [
                        'Timestamp,ImageX,ImageY,Intensity,Confidence,PredictedClass,Scale,OffsetX,OffsetY',
                        ...state.clickHistory.map(h =>
                          `${h.timestamp},${h.imageX},${h.imageY},${h.intensity || ''},${h.confidence || ''},${h.predictedClass || ''},${h.scale},${h.offsetX},${h.offsetY}`
                        )
                      ].join('\n');
                      const copied = await copyText(csv);
                      alert(copied ? 'Click history copied to clipboard as CSV' : 'Failed to copy click history CSV');
                    }}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                  >
                    CSV
                  </button>
                </div>
              </div>
              {(() => {
                const exosomeClicks = state.clickHistory.filter((click) => getClickCategory(click) === 'exosome');
                const backgroundClicks = state.clickHistory.filter((click) => getClickCategory(click) === 'background');
                const renderPanel = (
                  title: string,
                  clicks: ExosomeDetectionState['clickHistory'],
                  accent: string,
                ) => (
                  <div style={{
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    backgroundColor: '#f9f9f9',
                    marginBottom: '0.6rem',
                  }}>
                    <div style={{
                      fontWeight: 700,
                      color: accent,
                      padding: '0.45rem 0.55rem',
                      borderBottom: '1px solid #e5e5e5',
                    }}>
                      {title} ({clicks.length})
                    </div>
                    <div style={{
                      maxHeight: '180px',
                      overflowY: 'auto',
                      padding: '0.45rem',
                      fontSize: '0.85rem',
                    }}>
                      {clicks.length === 0 ? (
                        <div style={{ color: '#999', fontSize: '0.8rem', paddingLeft: '0.25rem' }}>(empty)</div>
                      ) : (
                        clicks.map((click, idx) => (
                          <div key={click.id} style={{
                            padding: '0.5rem',
                            borderBottom: idx < clicks.length - 1 ? '1px solid #eee' : 'none',
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
                                  <div><strong>Class:</strong> {getClickCategory(click) === 'background' ? 'Background' : 'Exosome'}</div>
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
                );
                return (
                  <>
                    {renderPanel('🟢 Exosome', exosomeClicks, '#0b8f3a')}
                    {renderPanel('⬜ Background', backgroundClicks, '#c0392b')}
                  </>
                );
              })()}
            </div>
          </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExosomeDetection;

