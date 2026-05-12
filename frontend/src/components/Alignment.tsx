/**
 * Alignment Component
 * Two-stage workflow:
 *   Stage A — Grid Feature Detection & Preview
 *   Stage B — Alignment Execution
 */

import React, { useState, useEffect, useRef } from 'react';
import AlignmentColorOverlay from './AlignmentColorOverlay';
import ChannelInfoPanel from './ChannelInfoPanel';
import AlignmentShiftPanel from './AlignmentShiftPanel';
import ManualDiagonalEditor, { DiagonalBox } from './ManualDiagonalEditor';
import '../styles/PipelineControl.css';
import { getApiBase } from '../lib/apiBase';
import * as storage from '../lib/storage';

// ─── Types ────────────────────────────────────────────────────────────────────

interface PreviewStage {
  [channel: string]: string;
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

// Re-use the ShiftVector interface from AlignmentShiftPanel via a local alias
// (avoid duplicate declarations that confuse TS)
type ShiftVector = {
  dx: number;
  dy: number;
  magnitude: number;
  angle_deg?: number;
  type: 'affine' | 'tps' | 'identity' | 'reference';
  residual_error?: number;
  num_matches?: number;
  note?: string;
};

// Shared preprocessing parameters (grid methods)
interface GridPreprocParams {
  blur_ksize: number;         // Gaussian kernel — larger suppresses sub-grid features
  threshold_method: 'otsu' | 'adaptive_mean' | 'adaptive_gaussian';
  invert: boolean;            // invert so dark grid lines → bright targets
}

// Method-specific parameter types
interface GridCornerParams {
  max_corners: number;
  quality_level: number;
  min_distance: number;
  block_size: number;
}

interface GridHoughParams {
  hough_threshold: number;
  min_line_length: number;
  max_line_gap: number;
  angle_tolerance: number;
  merge_distance: number;
}

type AlignMethod = 'grid_intersection_homography' | 'grid_line_hough' | 'frequency_domain_fft' | 'manual_diagonal';

// Preview layers — a superset covering all methods
type PreviewLayer =
  | 'preprocessed' | 'binary_mask' | 'detected' | 'intersections'  // Hough / corner
  | 'fft_spectrum' | 'synthetic_grid' | 'all';                       // FFT

interface FftParams {
  fft_mode: 'grid_detection' | 'simple_correlation';
  window_fn: 'hann' | 'hamming' | 'none';
  min_spacing_px: number;
  max_spacing_px: number;
  peak_threshold: number;
  enable_rotation: boolean;
  grid_line_width: number;
}

// ─── Persistence ──────────────────────────────────────────────────────────────

const DEFAULT_SAMPLE   = 'A2780Cis10';
const DEFAULT_POSITION = 'P1';

export interface CropRect { x: number; y: number; w: number; h: number }

interface TiffExportPreference {
  autoDownload: boolean;
  format: 'zip' | 'composite';
}

interface AlignmentSnapshot {
  diagonalBoxes:        Record<string, DiagonalBox | null>;
  shiftVectors:         Record<string, ShiftVector>;
  refChannel:           string;
  selectedChannels:     string[];
  alignMethod:          AlignMethod;
  savedAt:              string;
  cropRect:             CropRect | null;
  tiffExportPreference: TiffExportPreference | null;
}

const snapshotKey = (sample: string, position: string) =>
  `sea_alignment_${sample}_${position}`;

// ─── Default parameters ────────────────────────────────────────────────────────

const DEFAULT_PREPROC_PARAMS: GridPreprocParams = {
  blur_ksize: 21,
  threshold_method: 'otsu',
  invert: true,
};

const DEFAULT_CORNER_PARAMS: GridCornerParams = {
  max_corners: 200,
  quality_level: 0.01,
  min_distance: 10,
  block_size: 3,
};

const DEFAULT_HOUGH_PARAMS: GridHoughParams = {
  hough_threshold: 50,
  min_line_length: 80,
  max_line_gap: 20,
  angle_tolerance: 15,
  merge_distance: 20,
};

const DEFAULT_FFT_PARAMS: FftParams = {
  fft_mode: 'grid_detection',
  window_fn: 'hann',
  min_spacing_px: 20,
  max_spacing_px: 200,
  peak_threshold: 0.3,
  enable_rotation: true,
  grid_line_width: 2,
};

// ─── Component ────────────────────────────────────────────────────────────────

interface AlignmentProps {
  isActive?: boolean;
}

const Alignment: React.FC<AlignmentProps> = ({ isActive }) => {
  // ── Selection & load state ──
  const [selectedSample, setSelectedSample] = useState('');
  const [selectedPosition, setSelectedPosition] = useState('');
  const [availableSamples, setAvailableSamples] = useState<string[]>([]);
  const [availablePositions, setAvailablePositions] = useState<string[]>([]);
  const [availableItems, setAvailableItems] = useState<ChannelItem[]>([]);
  const [selectedChannels, setSelectedChannels] = useState<Set<string>>(new Set());
  const [isLoadingPosition, setIsLoadingPosition] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Processed stage selection (input from Image Processing)
  const [availableProcessedStages, setAvailableProcessedStages] = useState<string[]>([]);
  const [selectedInputStage, setSelectedInputStage] = useState('');

  // Stage previews
  const [stages, setStages] = useState<Record<string, PreviewStage>>({});
  const [currentStage, setCurrentStage] = useState<string>('aligned');
  const [backendStats, setBackendStats] = useState<Record<string, any>>({});

  // ── Method & parameters ──
  const [alignMethod, setAlignMethod] = useState<AlignMethod>('manual_diagonal');
  const [transformType, setTransformType] = useState('EuclideanTransform');
  const [refChannel, setRefChannel] = useState('');
  const [preprocParams, setPreprocParams] = useState<GridPreprocParams>(DEFAULT_PREPROC_PARAMS);
  const [cornerParams, setCornerParams] = useState<GridCornerParams>(DEFAULT_CORNER_PARAMS);
  const [houghParams, setHoughParams] = useState<GridHoughParams>(DEFAULT_HOUGH_PARAMS);
  const [fftParams, setFftParams] = useState<FftParams>(DEFAULT_FFT_PARAMS);

  // ── Stage A: Feature detection ──
  const [isDetecting, setIsDetecting] = useState(false);
  // preview_layers[channel][layer] = base64 PNG
  const [previewLayers, setPreviewLayers] = useState<Record<string, Record<string, string>>>({});
  const [selectedPreviewLayer, setSelectedPreviewLayer] = useState<PreviewLayer>('detected');
  const [featureCounts, setFeatureCounts] = useState<Record<string, any>>({});
  const [featureSummary, setFeatureSummary] = useState('');
  const [featureDetected, setFeatureDetected] = useState(false);
  const [showFeatureOverlay, setShowFeatureOverlay] = useState(true);
  const [featureDetectionError, setFeatureDetectionError] = useState('');

  // ── Stage B: Alignment ──
  const [isAligning, setIsAligning] = useState(false);
  const [shiftVectors, setShiftVectors] = useState<Record<string, ShiftVector>>({});
  const [lastAlignmentRefChannel, setLastAlignmentRefChannel] = useState('');
  const [lastAlignmentInputStage, setLastAlignmentInputStage] = useState('');
  const [alignmentRunId, setAlignmentRunId] = useState<number | null>(null);

  // ── Channel interaction (for shift panel / overlay) ──
  const [highlightedChannel, setHighlightedChannel] = useState<string | null>(null);
  const [channelVisibility, setChannelVisibility] = useState<Record<string, boolean>>({});

  // ── Manual Diagonal method ──
  const [diagonalBoxes, setDiagonalBoxes] = useState<Record<string, DiagonalBox | null>>({});
  const diagonalBoxesRef = useRef<Record<string, DiagonalBox | null>>({});

  // ── Persistence ──
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [pendingAutoLoad, setPendingAutoLoad] = useState(false);
  const autoLoadFiredRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Crop tool ──
  const [cropRect, setCropRect] = useState<CropRect | null>(null);
  const [cropExpanded, setCropExpanded] = useState(false);
  const [previewCropMode, setPreviewCropMode] = useState(false);
  const previewImageRef = useRef<HTMLImageElement | null>(null);
  const previewCropDragRef = useRef<{
    dragging: boolean;
    startX: number;
    startY: number;
    currentX: number;
    currentY: number;
  } | null>(null);
  const [previewCropDragDisplay, setPreviewCropDragDisplay] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);
  const [previewImageMeta, setPreviewImageMeta] = useState<{
    naturalW: number;
    naturalH: number;
    displayW: number;
    displayH: number;
  }>({ naturalW: 0, naturalH: 0, displayW: 0, displayH: 0 });

  // ── TIFF export ──
  const [tiffExportPref, setTiffExportPref] = useState<TiffExportPreference>({
    autoDownload: false,
    format: 'zip',
  });
  const [exportStatus, setExportStatus] = useState<string>('');

  // ── Preview channel selection ──
  const [selectedPreviewChannel, setSelectedPreviewChannel] = useState('');

  // ─── Effects ────────────────────────────────────────────────────────────────

  // Fetch samples on mount; auto-select default
  useEffect(() => {
    fetch(`${getApiBase()}/api/input/samples`)
      .then(r => r.json())
      .then(d => {
        if (!d.success) return;
        setAvailableSamples(d.data);
        if (!selectedSample && d.data.includes(DEFAULT_SAMPLE))
          setSelectedSample(DEFAULT_SAMPLE);
      })
      .catch(console.error);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch positions when sample changes; auto-select default (prefer .sea_state/ui)
  useEffect(() => {
    if (!selectedSample) {
      setAvailablePositions([]);
      setSelectedPosition('');
      setLoaded(false);
      setStages({});
      return;
    }
    let cancelled = false;
    fetch(`${getApiBase()}/api/input/samples/${selectedSample}/positions`)
      .then(r => r.json())
      .then(async d => {
        if (cancelled || !d.success) return;
        const positions: string[] = d.data;
        setAvailablePositions(positions);
        const disk = await storage.loadStateFromDisk(selectedSample);
        if (cancelled) return;
        const ui = disk.ui as { alignment?: { position?: string; channel?: string } } | undefined;
        const wantPos = ui?.alignment?.position;
        setSelectedPosition(prev => {
          let next: string;
          if (prev && positions.includes(prev)) next = prev;
          else if (wantPos && positions.includes(wantPos)) next = wantPos;
          else if (positions.includes(DEFAULT_POSITION)) next = DEFAULT_POSITION;
          else next = positions[0] || '';
          if (
            next === DEFAULT_POSITION &&
            selectedSample === DEFAULT_SAMPLE &&
            !autoLoadFiredRef.current
          ) {
            setPendingAutoLoad(true);
          }
          return next;
        });
      })
      .catch(console.error);
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSample]);

  // Re-fetch processed previews + stats when the Alignment tab becomes active,
  // so changes made in the Image Processing tab are reflected here without
  // requiring the user to re-load the position.
  useEffect(() => {
    if (!isActive || !loaded || !selectedSample || !selectedPosition) return;
    fetch(
      `${getApiBase()}/api/input/preprocess/final?sample=${encodeURIComponent(selectedSample)}&position=${encodeURIComponent(selectedPosition)}`
    )
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        if (!data?.success) return;
        const processedPreviews: PreviewStage = data.data.previews || {};
        const stats: Record<string, any> = data.data.stats || {};
        setStages(prev => ({ ...prev, processed: processedPreviews }));
        setBackendStats(stats);
        // Only advance stage from 'raw' → 'processed'; never overwrite 'features' or 'aligned'
        if (Object.keys(processedPreviews).length > 0) {
          setCurrentStage(prev => (prev === 'raw' ? 'processed' : prev));
        }
      })
      .catch(console.error);
  }, [isActive]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist last alignment selection + tab to sample-local ui_state.json
  useEffect(() => {
    if (!selectedSample || !selectedPosition) return;
    void storage.saveUiTabSlice(selectedSample, 'alignment', {
      position: selectedPosition,
      channel: selectedPreviewChannel || undefined,
      lastActiveTab: isActive ? 'alignment' : undefined,
    });
  }, [isActive, selectedSample, selectedPosition, selectedPreviewChannel]);

  // Reset Stage A when relevant inputs change (keep diagonalBoxes — user controls via Reset button)
  useEffect(() => {
    setFeatureDetected(false);
    setPreviewLayers({});
    setFeatureSummary('');
    setFeatureDetectionError('');
  }, [selectedInputStage, alignMethod, refChannel]);

  // Keep a synchronous ref to avoid stale-state payloads when user clicks Align
  // immediately after manipulating the box.
  useEffect(() => {
    diagonalBoxesRef.current = diagonalBoxes;
  }, [diagonalBoxes]);

  // Crop mode is only meaningful for aligned previews.
  useEffect(() => {
    if (currentStage !== 'aligned' && previewCropMode) setPreviewCropMode(false);
  }, [currentStage, previewCropMode]);

  useEffect(() => {
    if (!previewCropMode) {
      setPreviewCropDragDisplay(null);
      previewCropDragRef.current = null;
    }
  }, [previewCropMode]);

  // ─── Handlers ────────────────────────────────────────────────────────────────

  // ─── Persistence helpers ─────────────────────────────────────────────────────

  const restoreAlignmentFromStorage = async (sample: string, position: string): Promise<boolean> => {
    try {
      const disk = await storage.loadStateFromDisk(sample, true);
      const al = disk.alignment as { byPosition?: Record<string, AlignmentSnapshot> } | undefined;
      const fromDisk = al?.byPosition?.[position];
      if (fromDisk && typeof fromDisk === 'object') {
        const snap = fromDisk;
        if (snap.diagonalBoxes) setDiagonalBoxes(snap.diagonalBoxes);
        if (snap.shiftVectors) setShiftVectors(snap.shiftVectors);
        if (snap.refChannel) setRefChannel(snap.refChannel);
        if (snap.selectedChannels) setSelectedChannels(new Set(snap.selectedChannels));
        if (snap.alignMethod) setAlignMethod(snap.alignMethod as AlignMethod);
        setSavedAt(snap.savedAt || null);
        setCropRect(snap.cropRect ?? null);
        if (snap.tiffExportPreference) setTiffExportPref(snap.tiffExportPreference);
        return true;
      }
    } catch {
      /* fall through to legacy store */
    }
    const raw = await storage.get(snapshotKey(sample, position));
    if (!raw) return false;
    try {
      const snap: AlignmentSnapshot = JSON.parse(raw);
      if (snap.diagonalBoxes)    setDiagonalBoxes(snap.diagonalBoxes);
      if (snap.shiftVectors)     setShiftVectors(snap.shiftVectors);
      if (snap.refChannel)       setRefChannel(snap.refChannel);
      if (snap.selectedChannels) setSelectedChannels(new Set(snap.selectedChannels));
      if (snap.alignMethod)      setAlignMethod(snap.alignMethod as AlignMethod);
      setSavedAt(snap.savedAt || null);
      setCropRect(snap.cropRect ?? null);
      if (snap.tiffExportPreference) setTiffExportPref(snap.tiffExportPreference);
      return true;
    } catch { return false; }
  };

  const saveAlignment = async () => {
    if (!selectedSample || !selectedPosition) return;
    const ts = new Date().toISOString();
    const snap: AlignmentSnapshot = {
      diagonalBoxes,
      shiftVectors,
      refChannel,
      selectedChannels: Array.from(selectedChannels),
      alignMethod,
      savedAt: ts,
      cropRect,
      tiffExportPreference: tiffExportPref,
    };
    await storage.set(snapshotKey(selectedSample, selectedPosition), JSON.stringify(snap));
    setSavedAt(ts);

    try {
      const disk = await storage.loadStateFromDisk(selectedSample, true);
      const prevAlign = (disk.alignment || {}) as { byPosition?: Record<string, AlignmentSnapshot> };
      const byPosition = { ...(prevAlign.byPosition || {}), [selectedPosition]: snap };
      void storage.saveStateToDisk(selectedSample, 'alignment', { byPosition });
    } catch {
      void storage.saveStateToDisk(selectedSample, 'alignment', { byPosition: { [selectedPosition]: snap } });
    }

    const inputStageForMirror = (lastAlignmentInputStage || selectedInputStage || 'raw').trim() || 'raw';
    if (alignmentRunId !== null) {
      try {
        const mirrorRes = await fetch(`${getApiBase()}/api/input/align_save_mirror`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample: selectedSample,
            position: selectedPosition,
            input_stage: inputStageForMirror,
            crop_rect: cropRect
              ? { x: cropRect.x, y: cropRect.y, w: cropRect.w, h: cropRect.h }
              : null,
          }),
        });
        const mirrorData = await mirrorRes.json().catch(() => ({}));
        if (!mirrorRes.ok || !mirrorData.success) {
          const msg = (mirrorData as { error?: string }).error || `HTTP ${mirrorRes.status}`;
          setExportStatus(`Align mirror: ${msg}`);
          setTimeout(() => setExportStatus(''), 8000);
        }
      } catch (e) {
        setExportStatus(`Align mirror: ${String(e)}`);
        setTimeout(() => setExportStatus(''), 8000);
      }
    }

    if (!tiffExportPref.autoDownload || alignmentRunId === null) return;

    const channelCount = Object.keys(shiftVectors).length;
    const label = tiffExportPref.format === 'composite'
      ? '1 composite TIFF'
      : `${channelCount} TIFF${channelCount !== 1 ? 's' : ''}`;
    setExportStatus(`Saved · Downloading ${label}…`);
    try {
      const res = await fetch(`${getApiBase()}/api/input/export_tiff`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
          input_stage: lastAlignmentInputStage,
          format: tiffExportPref.format,
          ...(cropRect ? { crop_rect: cropRect } : {}),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        setExportStatus(`Export failed: ${err.error || 'Unknown error'}`);
        setTimeout(() => setExportStatus(''), 5000);
        return;
      }
      const blob = await res.blob();
      const disposition = res.headers.get('Content-Disposition') || '';
      const fnMatch = disposition.match(/filename="([^"]+)"/);
      const filename = fnMatch
        ? fnMatch[1]
        : tiffExportPref.format === 'composite'
          ? `${selectedSample}_${selectedPosition}_aligned_composite.tif`
          : `${selectedSample}_${selectedPosition}_aligned.zip`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setExportStatus(`Saved · ${label} downloaded ✓`);
      setTimeout(() => setExportStatus(''), 3000);
    } catch (err) {
      setExportStatus(`Export error: ${err}`);
      setTimeout(() => setExportStatus(''), 5000);
    }
  };

  const loadSavedAlignment = () => {
    void restoreAlignmentFromStorage(selectedSample, selectedPosition);
  };

  const resetAlignment = () => {
    if (!selectedSample || !selectedPosition) return;
    void storage.remove(snapshotKey(selectedSample, selectedPosition));
    void (async () => {
      try {
        const disk = await storage.loadStateFromDisk(selectedSample, true);
        const prevAlign = (disk.alignment || {}) as { byPosition?: Record<string, AlignmentSnapshot> };
        const byPosition = { ...(prevAlign.byPosition || {}) };
        delete byPosition[selectedPosition];
        void storage.saveStateToDisk(selectedSample, 'alignment', { byPosition });
      } catch { /* ignore */ }
    })();
    const cleared: Record<string, DiagonalBox | null> = {};
    availableItems.forEach(i => { cleared[i.key] = null; });
    setDiagonalBoxes(cleared);
    setShiftVectors({});
    setAlignmentRunId(null);
    setSavedAt(null);
    setCropRect(null);
    setExportStatus('');
  };

  const handleLoadPosition = async () => {
    if (!selectedSample || !selectedPosition) return;
    setIsLoadingPosition(true);

    try {
      const loadRes = await fetch(`${getApiBase()}/api/input/load_position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample: selectedSample, position: selectedPosition }),
      });
      const loadData = await loadRes.json();
      if (!loadData.success) {
        alert(loadData.error || 'Failed to load position');
        setIsLoadingPosition(false);
        return;
      }

      const items: ChannelItem[] = loadData.data.items || [];
      const rawPreviews: PreviewStage = {};
      items.forEach(item => { if (item.preview_url) rawPreviews[item.key] = item.preview_url; });

      // Query available processed stages (raw is always offered separately in the UI)
      let availableStages: string[] = [];
      let defaultInputStage = 'raw';
      try {
        const stateRes = await fetch(
          `${getApiBase()}/api/input/preprocess/state?sample=${encodeURIComponent(selectedSample)}&position=${encodeURIComponent(selectedPosition)}`
        );
        if (stateRes.ok) {
          const stateData = await stateRes.json();
          if (stateData.success && stateData.data.channel_states) {
            const stageSet = new Set<string>();
            Object.values(stateData.data.channel_states).forEach((cs: any) => {
              if (cs && typeof cs === 'object') Object.keys(cs).forEach(s => { if (s !== 'raw') stageSet.add(s); });
            });
            const order = ['contrast_enhance', 'step1', 'step2', 'step3', 'step4'];
            availableStages = Array.from(stageSet).sort((a, b) => {
              const ai = order.indexOf(a), bi = order.indexOf(b);
              if (ai === -1 && bi === -1) return a.localeCompare(b);
              if (ai === -1) return 1; if (bi === -1) return -1;
              return ai - bi;
            });
            if (availableStages.length > 0) defaultInputStage = availableStages[availableStages.length - 1];
          }
        }
      } catch { /* fallback */ }

      // Fetch processed final previews
      let processedPreviews: PreviewStage = {};
      let stats: Record<string, any> = {};
      try {
        const finalRes = await fetch(
          `${getApiBase()}/api/input/preprocess/final?sample=${encodeURIComponent(selectedSample)}&position=${encodeURIComponent(selectedPosition)}`
        );
        if (finalRes.ok) {
          const finalData = await finalRes.json();
          if (finalData.success) {
            processedPreviews = finalData.data.previews || {};
            stats = finalData.data.stats || {};
          }
        }
      } catch { /* no processed previews available */ }

      setAvailableItems(items);
      setSelectedChannels(new Set(items.map(i => i.key)));
      setStages({ raw: rawPreviews, processed: processedPreviews });
      setBackendStats(stats);
      setCurrentStage(Object.keys(processedPreviews).length > 0 ? 'processed' : 'raw');
      setAvailableProcessedStages(availableStages);
      setSelectedInputStage(defaultInputStage);
      setLoaded(items.length > 0);
      let previewCh = items[0]?.key || '';
      try {
        const diskUi = await storage.loadStateFromDisk(selectedSample, true);
        const want = (diskUi.ui as { alignment?: { channel?: string } } | undefined)?.alignment?.channel;
        if (want && items.some(i => i.key === want)) previewCh = want;
      } catch { /* keep default */ }
      setSelectedPreviewChannel(previewCh);
      if (items.length > 0) setRefChannel(previewCh || items[0].key);

      // Reset Stage A/B state
      setFeatureDetected(false);
      setPreviewLayers({});
      setFeatureSummary('');
      setFeatureDetectionError('');
      setShiftVectors({});
      setAlignmentRunId(null);
      setDiagonalBoxes({});
      setCropRect(null);
      setExportStatus('');

      // Auto-restore saved alignment for this sample/position (overwrites the blanked state)
      await restoreAlignmentFromStorage(selectedSample, selectedPosition);
    } catch (err) {
      alert('Failed to load position: ' + err);
    } finally {
      setIsLoadingPosition(false);
    }
  };

  const handleToggleChannel = (key: string) => {
    setSelectedChannels(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const handleSelectImageFile = () => {
    fileInputRef.current?.click();
  };

  const handleImageFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    alert(`Selected image file:\n${file.name}\n\nUse Sample/Position selection for alignment workflow.`);
  };

  // ── Stage A: Detect Features ────────────────────────────────────────────────

  const handleDetectFeatures = async () => {
    if (!loaded || !selectedInputStage) return;
    setIsDetecting(true);
    setFeatureDetected(false);
    setFeatureDetectionError('');

    // Build method_params: shared preproc params (grid methods) + method-specific params
    const methodParams: Record<string, any> =
      alignMethod === 'frequency_domain_fft'
        ? { ...fftParams }
        : {
            ...preprocParams,
            ...(alignMethod === 'grid_intersection_homography' ? cornerParams : {}),
            ...(alignMethod === 'grid_line_hough' ? houghParams : {}),
          };

    try {
      const res = await fetch(`${getApiBase()}/api/input/detect_features`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
          input_stage: selectedInputStage,
          ref_channel: refChannel,
          method: alignMethod,
          method_params: methodParams,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setPreviewLayers(data.data.preview_layers || {});
        setFeatureCounts(data.data.feature_counts || {});
        setFeatureSummary(data.data.summary || '');
        setFeatureDetected(true);
        setSelectedPreviewLayer('detected');
        setCurrentStage('features');
      } else {
        setFeatureDetectionError(data.error || 'Feature detection failed');
      }
    } catch (err) {
      setFeatureDetectionError('Feature detection failed: ' + err);
    } finally {
      setIsDetecting(false);
    }
  };

  // ── Stage B: Run Alignment ───────────────────────────────────────────────────

  const handleRunAlignment = async () => {
    if (!loaded || !selectedInputStage) return;
    if (selectedChannels.size < 2) { alert('Please select at least 2 channels'); return; }
    if (!selectedChannels.has(refChannel)) { alert('Reference channel must be selected'); return; }
    const diagonalBoxesPayload = diagonalBoxesRef.current;
    if (alignMethod === 'manual_diagonal') {
      const refBox = diagonalBoxesPayload[refChannel];
      if (!refBox) {
        alert('Reference channel is missing a manual diagonal box.');
        return;
      }
      const selected = Array.from(selectedChannels);
      const allTargetsMatchRef = selected
        .filter(ch => ch !== refChannel)
        .every(ch => {
          const b = diagonalBoxesPayload[ch];
          if (!b) return false;
          return (
            Math.abs(b.angle - refBox.angle) < 1e-6 &&
            Math.abs(b.cx - refBox.cx) < 1e-6 &&
            Math.abs(b.cy - refBox.cy) < 1e-6 &&
            Math.abs(b.width - refBox.width) < 1e-6 &&
            Math.abs(b.height - refBox.height) < 1e-6
          );
        });
      if (allTargetsMatchRef) {
        console.warn(
          '[Alignment] All target boxes match reference. Proceeding with identity transform.'
        );
      }
      const selectedMovingChannel =
        selectedPreviewChannel !== refChannel ? selectedPreviewChannel : selected.find(ch => ch !== refChannel) || selectedPreviewChannel;
      const targetBox = diagonalBoxesPayload[selectedMovingChannel] || null;
      console.log('[Alignment debug] manual box state before align', {
        selectedMovingChannel,
        referenceChannel: refChannel,
        ref_box: refBox,
        target_box: targetBox,
        ref_box_angle: refBox?.angle ?? null,
        target_box_angle: targetBox?.angle ?? null,
        ref_center: refBox ? { cx: refBox.cx, cy: refBox.cy } : null,
        target_center: targetBox ? { cx: targetBox.cx, cy: targetBox.cy } : null,
      });
    }

    setIsAligning(true);
    try {
      const requestPayload = {
        sample: selectedSample,
        position: selectedPosition,
        input_stage: selectedInputStage,
        ref_channel: refChannel,
        method: alignMethod,
        transform: transformType,
        selected_channels: Array.from(selectedChannels),
        ...(alignMethod === 'manual_diagonal' && { diagonal_boxes: diagonalBoxesPayload }),
      };
      console.log('[Alignment debug] /api/input/align request payload', requestPayload);
      const res = await fetch(`${getApiBase()}/api/input/align`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload),
      });
      const data = await res.json();
      if (data.success) {
        console.log('[Alignment debug] backend align response', data);
        const alignedPreviews = data.data.previews || {};
        const newVectors = data.data.shift_vectors || {};
        const newStats = data.data.stats || {};
        const responseRunId = data.data.alignment_run_id ?? Date.now();
        console.log('[Alignment debug] align response details', {
          stagesAligned: alignedPreviews,
          selectedPreviewStage: 'aligned',
          selectedPreviewChannelBeforeUpdate: selectedPreviewChannel,
          shiftVectorsFromBackend: newVectors,
          alignmentRunId: responseRunId,
        });
        setStages(prev => ({ ...prev, aligned: alignedPreviews }));
        setBackendStats(prev => ({ ...prev, ...newStats }));
        setCurrentStage('aligned');
        setLastAlignmentRefChannel(refChannel);
        setLastAlignmentInputStage(selectedInputStage);
        setShiftVectors(newVectors);
        setAlignmentRunId(responseRunId);
        // Auto-switch the preview to the first non-reference channel that has an
        // aligned preview, so the user immediately sees the rotated result rather
        // than the reference channel (which is always unrotated by definition).
        const firstTargetWithPreview = Object.keys(alignedPreviews).find(
          k => k !== refChannel && alignedPreviews[k]
        );
        if (firstTargetWithPreview) {
          setSelectedPreviewChannel(firstTargetWithPreview);
        } else if (!alignedPreviews[selectedPreviewChannel]) {
          // Fallback: current channel has no preview — switch to whatever is available.
          const anyWithPreview = Object.keys(alignedPreviews).find(k => alignedPreviews[k]);
          if (anyWithPreview) setSelectedPreviewChannel(anyWithPreview);
        }
      } else {
        alert(data.error || 'Alignment failed');
      }
    } catch (err) {
      alert('Alignment failed: ' + err);
    } finally {
      setIsAligning(false);
    }
  };

  // Auto-load default sample/position on first mount (placed after handler to avoid hoisting error)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!pendingAutoLoad || autoLoadFiredRef.current) return;
    autoLoadFiredRef.current = true;
    setPendingAutoLoad(false);
    handleLoadPosition();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAutoLoad]);

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  const stageLabel = (s: string) => {
    const map: Record<string, string> = {
      raw: 'Raw',
      contrast_enhance: 'Contrast Enhanced',
      step1: 'Bkg Subtracted',
      step2: 'Clipped',
      step3: 'Gaussian Blurred',
      step4: 'Connected',
    };
    return map[s] ?? s;
  };

  const featureCountLabel = (ch: string) => {
    const fc = featureCounts[ch];
    if (!fc) return '';
    if (fc.corners !== undefined) return `${fc.corners} corners`;
    if (fc.h_lines !== undefined) return `${fc.h_lines}H + ${fc.v_lines}V lines, ${fc.intersections} pts`;
    if (fc.h_spacing_px !== undefined) {
      if (fc.warning) return `⚠️ ${fc.warning}`;
      return `${fc.h_spacing_px ?? '?'}×${fc.v_spacing_px ?? '?'} px, ${fc.intersections} pts`;
    }
    if (fc.mode === 'simple_correlation') return 'simple PCC';
    return '';
  };

  // True when alignment can run.
  // manual_diagonal: every selected channel must have a box drawn.
  // All other methods: no mandatory prerequisite — feature detection is optional QC.
  const readyToAlign =
    alignMethod === 'manual_diagonal'
      ? selectedChannels.size >= 2 &&
        Array.from(selectedChannels).every(ch => diagonalBoxes[ch] != null)
      : selectedChannels.size >= 2 && !!selectedInputStage;

  // Image URL for the ManualDiagonalEditor.
  // Always uses processed (or raw) so the user can still see and edit boxes on the
  // unwarped image.  After alignment runs, currentStage switches to 'aligned', and
  // the ternary below will fall through to getPreviewSrc() instead of the editor.
  const diagImageSrc = (() => {
    if (alignMethod !== 'manual_diagonal') return null;
    const url = stages.processed?.[selectedPreviewChannel] ?? stages.raw?.[selectedPreviewChannel];
    if (!url) return null;
    return url.startsWith('http') ? url : `${getApiBase()}${url}`;
  })();

  // True when the center preview should show the aligned result rather than the editor.
  // Condition: alignment has run (stages.aligned exists) AND the stage selector is 'aligned'.
  const showAlignedPreview =
    alignMethod === 'manual_diagonal' &&
    !!stages.aligned &&
    currentStage === 'aligned';

  // Get the currently displayed image URL
  const getPreviewSrc = (): string | null => {
    if (currentStage === 'features' && showFeatureOverlay) {
      const channelLayers = previewLayers[selectedPreviewChannel];
      const b64 = channelLayers?.[selectedPreviewLayer] ?? channelLayers?.['detected'];
      if (b64) return `data:image/png;base64,${b64}`;
    }
    let url: string | undefined;
    if (currentStage === 'processed') {
      // Per-channel: use raw preview when Image Processing did not produce one for this channel.
      url = stages.processed?.[selectedPreviewChannel] ?? stages.raw?.[selectedPreviewChannel];
    } else {
      const stageData = stages[currentStage];
      if (!stageData) return null;
      url = stageData[selectedPreviewChannel];
    }
    if (!url) return null;
    const abs = url.startsWith('http') ? url : `${getApiBase()}${url}`;
    const cacheBuster =
      currentStage === 'aligned'
        ? (alignmentRunId ?? Date.now())
        : Date.now();
    return `${abs}${abs.includes('?') ? '&' : '?'}t=${cacheBuster}`;
  };

  const updatePreviewImageMeta = () => {
    const img = previewImageRef.current;
    if (!img) return;
    setPreviewImageMeta({
      naturalW: img.naturalWidth || 0,
      naturalH: img.naturalHeight || 0,
      displayW: img.clientWidth || 0,
      displayH: img.clientHeight || 0,
    });
  };

  const setCropRectFromDisplay = (x0: number, y0: number, x1: number, y1: number) => {
    const { naturalW, naturalH, displayW, displayH } = previewImageMeta;
    if (!naturalW || !naturalH || !displayW || !displayH) return;
    const clamp = (v: number, max: number) => Math.max(0, Math.min(v, max));
    const sx0 = clamp(x0, displayW);
    const sy0 = clamp(y0, displayH);
    const sx1 = clamp(x1, displayW);
    const sy1 = clamp(y1, displayH);
    const left = Math.min(sx0, sx1);
    const top = Math.min(sy0, sy1);
    const w = Math.abs(sx1 - sx0);
    const h = Math.abs(sy1 - sy0);
    if (w < 2 || h < 2) return;
    setCropRect({
      x: Math.round((left / displayW) * naturalW),
      y: Math.round((top / displayH) * naturalH),
      w: Math.round((w / displayW) * naturalW),
      h: Math.round((h / displayH) * naturalH),
    });
  };

  const handlePreviewCropMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!previewCropMode || currentStage !== 'aligned') return;
    const box = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    previewCropDragRef.current = { dragging: true, startX: x, startY: y, currentX: x, currentY: y };
    setPreviewCropDragDisplay({ x, y, w: 0, h: 0 });
    e.preventDefault();
  };

  const handlePreviewCropMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!previewCropDragRef.current?.dragging) return;
    const box = e.currentTarget.getBoundingClientRect();
    const curX = e.clientX - box.left;
    const curY = e.clientY - box.top;
    const drag = previewCropDragRef.current;
    drag.currentX = curX;
    drag.currentY = curY;
    setPreviewCropDragDisplay({
      x: Math.min(drag.startX, curX),
      y: Math.min(drag.startY, curY),
      w: Math.abs(curX - drag.startX),
      h: Math.abs(curY - drag.startY),
    });
    setCropRectFromDisplay(drag.startX, drag.startY, curX, curY);
  };

  const handlePreviewCropMouseUp = () => {
    const drag = previewCropDragRef.current;
    if (!drag?.dragging) return;
    drag.dragging = false;
    setCropRectFromDisplay(drag.startX, drag.startY, drag.currentX, drag.currentY);
    setPreviewCropDragDisplay(null);
    previewCropDragRef.current = null;
  };

  const previewCropDisplayRect = (() => {
    if (!cropRect) return null;
    const { naturalW, naturalH, displayW, displayH } = previewImageMeta;
    if (!naturalW || !naturalH || !displayW || !displayH) return null;
    return {
      x: (cropRect.x / naturalW) * displayW,
      y: (cropRect.y / naturalH) * displayH,
      w: (cropRect.w / naturalW) * displayW,
      h: (cropRect.h / naturalH) * displayH,
    };
  })();

  const renderPreviewImage = (src: string, alt: string) => (
    <div
      style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}
      onContextMenu={(e) => e.preventDefault()}
    >
      <img
        ref={previewImageRef}
        src={src}
        alt={alt}
        onLoad={updatePreviewImageMeta}
        onContextMenu={(e) => e.preventDefault()}
        style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: previewCropMode && currentStage === 'aligned' ? 'auto' : 'none',
          cursor: previewCropMode && currentStage === 'aligned' ? 'crosshair' : 'default',
        }}
        onMouseDown={handlePreviewCropMouseDown}
        onMouseMove={handlePreviewCropMouseMove}
        onMouseUp={handlePreviewCropMouseUp}
        onMouseLeave={handlePreviewCropMouseUp}
      >
        <svg width="100%" height="100%" style={{ display: 'block' }}>
          {(previewCropDragDisplay || previewCropDisplayRect) && (() => {
            const r = previewCropDragDisplay || previewCropDisplayRect!;
            const x0 = r.x;
            const y0 = r.y;
            const x1 = r.x + r.w;
            const y1 = r.y + r.h;
            const cornerLen = 14;
            return (
              <>
                <defs>
                  <mask id="preview-crop-mask">
                    <rect width="100%" height="100%" fill="white" />
                    <rect x={r.x} y={r.y} width={r.w} height={r.h} fill="black" />
                  </mask>
                </defs>
                <rect width="100%" height="100%" fill="rgba(0,0,0,0.35)" mask="url(#preview-crop-mask)" />
                <path d={`M ${x0 + cornerLen} ${y0} L ${x0} ${y0} L ${x0} ${y0 + cornerLen}`} stroke="#ffffff" strokeWidth={2.5} fill="none" />
                <path d={`M ${x1 - cornerLen} ${y0} L ${x1} ${y0} L ${x1} ${y0 + cornerLen}`} stroke="#ffffff" strokeWidth={2.5} fill="none" />
                <path d={`M ${x0 + cornerLen} ${y1} L ${x0} ${y1} L ${x0} ${y1 - cornerLen}`} stroke="#ffffff" strokeWidth={2.5} fill="none" />
                <path d={`M ${x1 - cornerLen} ${y1} L ${x1} ${y1} L ${x1} ${y1 - cornerLen}`} stroke="#ffffff" strokeWidth={2.5} fill="none" />
              </>
            );
          })()}
        </svg>
      </div>
    </div>
  );

  useEffect(() => {
    if (!loaded) return;
    const stageUrl =
      currentStage === 'processed'
        ? stages.processed?.[selectedPreviewChannel] ?? stages.raw?.[selectedPreviewChannel] ?? null
        : stages[currentStage]?.[selectedPreviewChannel] ?? null;
    const previewSrc = getPreviewSrc();
    const shiftForPreviewChannel = shiftVectors[selectedPreviewChannel] ?? null;
    console.log('[Alignment debug] preview/state snapshot', {
      selectedPreviewStage: currentStage,
      selectedPreviewChannel,
      stagesAlignedValue: stages.aligned ?? null,
      stageUrlForSelectedChannel: stageUrl,
      actualPreviewImageUrl: previewSrc,
      transformMetadataForSelectedChannel: shiftForPreviewChannel,
      shiftVectorsAll: shiftVectors,
      alignmentRunId,
    });
  }, [
    loaded,
    currentStage,
    selectedPreviewChannel,
    stages,
    shiftVectors,
    alignmentRunId,
  ]);

  const channelsConfig = (() => {
    const cfg: Record<string, { enabled: boolean; color: [number, number, number]; name: string }> = {};
    const colors: [number, number, number][] = [[1,0,0],[1,1,0],[0,1,0],[0,0,1]];
    const names = ['Red', 'Yellow', 'Green', 'Blue'];
    availableItems.forEach(item => {
      const m = item.channel?.match(/ch(\d+)/i);
      if (m) {
        const idx = (parseInt(m[1], 10) - 1) % 4;
        cfg[item.key] = { enabled: channelVisibility[item.key] !== false, color: colors[idx], name: names[idx] };
      }
    });
    return cfg;
  })();

  // ─── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="pipeline-control">
      <div className="pipeline-layout">

        {/* ── LEFT SIDEBAR ── */}
        <div className="sidebar-panel">

          {/* 1. File Operations */}
          <div className="sidebar-section">
            <h3>📁 File Operations</h3>
            <div className="sidebar-content">
              <div className="input-group">
                <label>Sample:</label>
                <select value={selectedSample} onChange={e => setSelectedSample(e.target.value)}
                  disabled={availableSamples.length === 0}>
                  <option value="">-- Select a sample --</option>
                  {availableSamples.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="input-group">
                <label>Position:</label>
                <select value={selectedPosition} onChange={e => setSelectedPosition(e.target.value)}
                  disabled={availablePositions.length === 0}>
                  <option value="">-- Select position --</option>
                  {availablePositions.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <button className="sidebar-btn" onClick={handleLoadPosition}
                disabled={!selectedSample || !selectedPosition || isLoadingPosition}>
                {isLoadingPosition ? 'Loading...' : 'Load Position'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                style={{ display: 'none' }}
                onChange={handleImageFileInputChange}
              />
              <button className="sidebar-btn btn-image" onClick={handleSelectImageFile}>
                Load Image File
              </button>
            </div>
          </div>

          {/* 2. Input Stage (renamed from "Contrast Enhanced") */}
          {loaded && (
            <div className="sidebar-section">
              <h3>📥 Input Stage</h3>
              <div className="sidebar-content">
                <div style={{ background: '#e8f4f8', padding: '10px', borderRadius: 4, marginBottom: 10, border: '1px solid #3498db', fontSize: '0.85rem' }}>
                  <strong style={{ color: '#2980b9' }}>Input Source:</strong> Use <strong>Raw</strong> (TIFFs after Load Position) or a <strong>preprocessed stage</strong> from Image Processing.
                </div>
                <div className="input-group">
                  <label>Input stage:</label>
                  <select value={selectedInputStage}
                    onChange={e => setSelectedInputStage(e.target.value)}
                    disabled={isAligning || isDetecting}>
                    <option value="raw">
                      {availableProcessedStages.length === 0 ? 'Raw (no preprocessing)' : 'Raw'}
                    </option>
                    {availableProcessedStages.map(s => (
                      <option key={s} value={s}>{stageLabel(s)}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* 3. Channel Selection */}
          {loaded && availableItems.length > 0 && (
            <div className="sidebar-section">
              <h3>📋 Channel Selection</h3>
              <div className="sidebar-content">
                <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                  <button className="sidebar-btn" style={{ fontSize: '0.82rem', padding: '5px 10px' }}
                    onClick={() => setSelectedChannels(new Set(availableItems.map(i => i.key)))}
                    disabled={isAligning || isDetecting}>Select All</button>
                  <button className="sidebar-btn" style={{ fontSize: '0.82rem', padding: '5px 10px' }}
                    onClick={() => setSelectedChannels(new Set())}
                    disabled={isAligning || isDetecting}>Deselect All</button>
                </div>
                <div style={{ maxHeight: 240, overflowY: 'auto', border: '1px solid #ddd', borderRadius: 4, padding: 8, background: '#f9f9f9' }}>
                  {availableItems.map(item => {
                    const isSel = selectedChannels.has(item.key);
                    const isRef = item.key === refChannel;
                    return (
                      <div key={item.key} onClick={() => !isAligning && !isDetecting && handleToggleChannel(item.key)}
                        style={{ display: 'flex', alignItems: 'center', padding: '6px 8px', marginBottom: 4, cursor: 'pointer',
                          background: isSel ? (isRef ? '#e8f5e9' : '#e3f2fd') : '#fff',
                          border: isRef ? '2px solid #4caf50' : isSel ? '1px solid #2196f3' : '1px solid #ddd',
                          borderRadius: 4, opacity: (isAligning || isDetecting) ? 0.6 : 1 }}>
                        <input type="checkbox" checked={isSel} onChange={() => handleToggleChannel(item.key)}
                          disabled={isAligning || isDetecting} style={{ marginRight: 8 }} />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: isRef ? 'bold' : 'normal', color: isRef ? '#2e7d32' : '#333', fontSize: '0.88rem' }}>
                            {item.display_label}{isRef && <span style={{ marginLeft: 6, color: '#4caf50' }}>⭐ Ref</span>}
                          </div>
                          {item.marker && <div style={{ fontSize: '0.74rem', color: '#666' }}>Marker: {item.marker}</div>}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div style={{ marginTop: 8, padding: 8, background: '#fff3cd', borderRadius: 4, fontSize: '0.82rem', color: '#856404' }}>
                  <strong>Selected:</strong> {selectedChannels.size} / {availableItems.length} channels
                  {selectedChannels.size < 2 && <div style={{ color: '#d32f2f', marginTop: 4 }}>⚠️ Select at least 2 channels</div>}
                </div>
              </div>
            </div>
          )}

          {/* 4. Method Selection */}
          {loaded && (
            <div className="sidebar-section">
              <h3>🔬 Method</h3>
              <div className="sidebar-content">
                <div className="input-group">
                  <label>Reference Channel:</label>
                  <select value={refChannel} onChange={e => setRefChannel(e.target.value)}
                    disabled={!loaded || isAligning || isDetecting}>
                    {availableItems.map(item => <option key={item.key} value={item.key}>{item.display_label}</option>)}
                  </select>
                </div>

                <div className="input-group">
                  <label>Alignment Method:</label>
                  <select value={alignMethod} onChange={e => setAlignMethod(e.target.value as AlignMethod)}
                    disabled={isAligning || isDetecting}>
                    <option value="grid_intersection_homography">Grid Intersection Homography</option>
                    <option value="grid_line_hough">Grid Line Hough Transform</option>
                    <option value="frequency_domain_fft">Frequency Domain (FFT)</option>
                    <option value="manual_diagonal">Manual Diagonal (Interactive)</option>
                  </select>
                </div>

                <div className="input-group">
                  <label>Transform:</label>
                  <select value={transformType} onChange={e => setTransformType(e.target.value)}
                    disabled={isAligning || isDetecting}>
                    <option value="EuclideanTransform">Euclidean (rotation + translation)</option>
                    <option value="AffineTransform">Affine (+ shear + scale)</option>
                  </select>
                </div>

                {/* Shared grid preprocessing parameters (both grid methods) */}
                {(alignMethod === 'grid_intersection_homography' || alignMethod === 'grid_line_hough') && (
                  <div style={{ background: '#fafafa', border: '1px solid #ddd', padding: 10, borderRadius: 4, marginTop: 10 }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 'bold', marginBottom: 8, color: '#444' }}>
                      Grid Preprocessing
                      <span style={{ fontWeight: 'normal', color: '#777', marginLeft: 6 }}>blur → invert → threshold</span>
                    </div>

                    <div className="input-group" style={{ marginBottom: 6 }}>
                      <label style={{ fontSize: '0.8rem' }}>
                        Blur Kernel: <strong>{preprocParams.blur_ksize}px</strong>
                        <span style={{ color: '#888', fontWeight: 'normal', marginLeft: 4 }}>(larger = suppress cell features)</span>
                      </label>
                      <input type="range" min={5} max={61} step={2}
                        value={preprocParams.blur_ksize}
                        onChange={e => setPreprocParams(p => ({ ...p, blur_ksize: parseInt(e.target.value) }))}
                        disabled={isAligning || isDetecting} style={{ width: '100%' }} />
                    </div>

                    <div className="input-group" style={{ marginBottom: 6 }}>
                      <label style={{ fontSize: '0.8rem' }}>Threshold Method:</label>
                      <select value={preprocParams.threshold_method}
                        onChange={e => setPreprocParams(p => ({ ...p, threshold_method: e.target.value as any }))}
                        disabled={isAligning || isDetecting}>
                        <option value="otsu">Otsu (global)</option>
                        <option value="adaptive_mean">Adaptive Mean</option>
                        <option value="adaptive_gaussian">Adaptive Gaussian</option>
                      </select>
                    </div>

                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8rem', cursor: 'pointer' }}>
                      <input type="checkbox" checked={preprocParams.invert}
                        onChange={e => setPreprocParams(p => ({ ...p, invert: e.target.checked }))}
                        disabled={isAligning || isDetecting} />
                      Invert image (ON for dark grid lines — critical)
                    </label>
                  </div>
                )}

                {/* Grid Intersection Homography: corner-specific params */}
                {alignMethod === 'grid_intersection_homography' && (
                  <div style={{ background: '#f5f5f5', padding: 10, borderRadius: 4, marginTop: 8 }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 'bold', marginBottom: 8, color: '#555' }}>Shi-Tomasi Corner Parameters</div>
                    {([
                      ['max_corners', 'Max Corners', 10, 500, 10],
                      ['quality_level', 'Quality Level', 0.001, 0.1, 0.001],
                      ['min_distance', 'Min Distance (px)', 2, 60, 1],
                      ['block_size', 'Block Size', 2, 10, 1],
                    ] as [keyof GridCornerParams, string, number, number, number][]).map(([key, label, min, max, step]) => (
                      <div key={key} className="input-group" style={{ marginBottom: 6 }}>
                        <label style={{ fontSize: '0.8rem' }}>{label}: <strong>{cornerParams[key]}</strong></label>
                        <input type="range" min={min} max={max} step={step}
                          value={cornerParams[key] as number}
                          onChange={e => setCornerParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                          disabled={isAligning || isDetecting} style={{ width: '100%' }} />
                      </div>
                    ))}
                  </div>
                )}

                {/* Grid Line Hough: hough-specific params */}
                {alignMethod === 'grid_line_hough' && (
                  <div style={{ background: '#f5f5f5', padding: 10, borderRadius: 4, marginTop: 8 }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 'bold', marginBottom: 8, color: '#555' }}>Hough Line Parameters</div>
                    {([
                      ['hough_threshold', 'Hough Threshold', 10, 200, 5],
                      ['min_line_length', 'Min Line Length (px)', 20, 300, 10],
                      ['max_line_gap', 'Max Line Gap (px)', 1, 80, 5],
                      ['angle_tolerance', 'Angle Tolerance (°)', 2, 30, 1],
                      ['merge_distance', 'Merge Distance (px)', 5, 80, 5],
                    ] as [keyof GridHoughParams, string, number, number, number][]).map(([key, label, min, max, step]) => (
                      <div key={key} className="input-group" style={{ marginBottom: 6 }}>
                        <label style={{ fontSize: '0.8rem' }}>{label}: <strong>{houghParams[key]}</strong></label>
                        <input type="range" min={min} max={max} step={step}
                          value={houghParams[key] as number}
                          onChange={e => setHoughParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                          disabled={isAligning || isDetecting} style={{ width: '100%' }} />
                      </div>
                    ))}
                  </div>
                )}

                {alignMethod === 'frequency_domain_fft' && (
                  <div style={{ background: '#f0f4ff', border: '1px solid #c5cae9', padding: 10, borderRadius: 4, marginTop: 10 }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 'bold', marginBottom: 8, color: '#303f9f' }}>
                      Frequency Domain (FFT) Parameters
                    </div>

                    <div className="input-group" style={{ marginBottom: 6 }}>
                      <label style={{ fontSize: '0.8rem' }}>FFT Mode:</label>
                      <select value={fftParams.fft_mode}
                        onChange={e => setFftParams(p => ({ ...p, fft_mode: e.target.value as any }))}
                        disabled={isAligning || isDetecting}>
                        <option value="grid_detection">Grid Structure Detection (full pipeline)</option>
                        <option value="simple_correlation">Simple Phase Correlation (fallback)</option>
                      </select>
                    </div>

                    {fftParams.fft_mode === 'simple_correlation' && (
                      <div style={{ background: '#fff8e1', color: '#7a5c00', padding: 6, borderRadius: 4, fontSize: '0.78rem', marginBottom: 6, border: '1px solid #ffe082' }}>
                        ⚠️ Simple correlation only detects image-to-image shift (dx, dy). It does NOT detect grid structure. Grid aliasing risk on repeating patterns.
                      </div>
                    )}

                    <div className="input-group" style={{ marginBottom: 6 }}>
                      <label style={{ fontSize: '0.8rem' }}>Window Function:</label>
                      <select value={fftParams.window_fn}
                        onChange={e => setFftParams(p => ({ ...p, window_fn: e.target.value as any }))}
                        disabled={isAligning || isDetecting || fftParams.fft_mode !== 'grid_detection'}>
                        <option value="hann">Hann (recommended)</option>
                        <option value="hamming">Hamming</option>
                        <option value="none">None (edge artifacts possible)</option>
                      </select>
                    </div>

                    {([
                      ['min_spacing_px', 'Min Grid Spacing (px)', 5, 100, 5],
                      ['max_spacing_px', 'Max Grid Spacing (px)', 50, 500, 10],
                      ['peak_threshold', 'Peak Threshold', 0.05, 0.9, 0.05],
                      ['grid_line_width', 'Grid Line Width (px)', 1, 10, 0.5],
                    ] as [keyof FftParams, string, number, number, number][]).map(([key, label, min, max, step]) => (
                      <div key={key} className="input-group" style={{ marginBottom: 6 }}>
                        <label style={{ fontSize: '0.8rem' }}>
                          {label}: <strong>{fftParams[key] as number}</strong>
                        </label>
                        <input type="range" min={min} max={max} step={step}
                          value={fftParams[key] as number}
                          onChange={e => setFftParams(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                          disabled={isAligning || isDetecting || fftParams.fft_mode !== 'grid_detection'}
                          style={{ width: '100%' }} />
                      </div>
                    ))}

                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8rem', cursor: 'pointer' }}>
                      <input type="checkbox" checked={fftParams.enable_rotation}
                        onChange={e => setFftParams(p => ({ ...p, enable_rotation: e.target.checked }))}
                        disabled={isAligning || isDetecting || fftParams.fft_mode !== 'grid_detection'} />
                      Enable Rotation Detection (wider angle tolerance)
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 5a. Manual Diagonal: interactive box placement */}
          {loaded && alignMethod === 'manual_diagonal' && (
            <div className="sidebar-section">
              <h3>✏️ Diagonal Boxes</h3>
              <div className="sidebar-content">
                <div style={{ fontSize: '0.82rem', color: '#555', marginBottom: 10, lineHeight: 1.5 }}>
                  Draw a diagonal across one grid cell in each channel image.
                  Click two corners → box auto-fits. Adjust per channel, then run alignment.
                </div>

                {/* Per-channel box status */}
                {availableItems.filter(i => selectedChannels.has(i.key)).map(item => {
                  const hasBox = !!diagonalBoxes[item.key];
                  const isRef  = item.key === refChannel;
                  return (
                    <div key={item.key}
                      onClick={() => setSelectedPreviewChannel(item.key)}
                      style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '4px 8px', marginBottom: 3, fontSize: '0.82rem', cursor: 'pointer',
                        background: hasBox ? '#e8f5e9' : '#fff3e0',
                        border: `1px solid ${hasBox ? '#81c784' : '#ffb74d'}`, borderRadius: 4,
                        outline: item.key === selectedPreviewChannel ? '2px solid #1976d2' : 'none' }}>
                      <span>{item.display_label}{isRef && <span style={{ color: '#4caf50', marginLeft: 4 }}>⭐</span>}</span>
                      <span style={{ color: hasBox ? '#2e7d32' : '#e65100', fontWeight: 'bold' }}>
                        {hasBox ? '✅' : '—'}
                      </span>
                    </div>
                  );
                })}

                {/* Action buttons */}
                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                  <button className="sidebar-btn"
                    style={{ fontSize: '0.8rem', padding: '5px 10px' }}
                    disabled={!diagonalBoxes[refChannel]}
                    title="Copy reference box position to all other channels"
                    onClick={() => {
                      const refBox = diagonalBoxes[refChannel];
                      if (!refBox) return;
                      const next: Record<string, DiagonalBox | null> = {};
                      availableItems.forEach(i => { next[i.key] = { ...refBox }; });
                      setDiagonalBoxes(next);
                    }}>
                    Copy Ref → All
                  </button>
                  <button className="sidebar-btn"
                    style={{ fontSize: '0.8rem', padding: '5px 10px', background: '#fff3e0', color: '#b71c1c' }}
                    onClick={() => {
                      const cleared: Record<string, DiagonalBox | null> = {};
                      availableItems.forEach(i => { cleared[i.key] = null; });
                      setDiagonalBoxes(cleared);
                    }}>
                    Reset All
                  </button>
                </div>

                {/* Current channel info */}
                {diagonalBoxes[selectedPreviewChannel] && (() => {
                  const b = diagonalBoxes[selectedPreviewChannel]!;
                  return (
                    <div style={{ background: '#f5f5f5', padding: 8, borderRadius: 4, marginTop: 10,
                      fontSize: '0.8rem', fontFamily: 'monospace', lineHeight: 1.6 }}>
                      <div><strong>{availableItems.find(i => i.key === selectedPreviewChannel)?.display_label}</strong></div>
                      <div>Center: ({Math.round(b.cx)}, {Math.round(b.cy)})</div>
                      <div>Size:   {Math.round(b.width)} × {Math.round(b.height)} px</div>
                      <div>Angle:  {(b.angle * 180 / Math.PI).toFixed(1)}°</div>
                    </div>
                  );
                })()}

                {/* Tips */}
                <div style={{ background: '#e3f2fd', padding: 8, borderRadius: 4, marginTop: 10,
                  fontSize: '0.78rem', color: '#1565c0', lineHeight: 1.5 }}>
                  <strong>Shortcuts (click canvas first):</strong>
                  <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                    <li>Arrows: nudge 1px (Shift = 10px)</li>
                    <li>r / R: rotate ±5°</li>
                    <li>Esc: clear box</li>
                    <li>Drag outside box: redraw</li>
                  </ul>
                </div>

                {readyToAlign && (
                  <div style={{ background: '#e8f5e9', color: '#2e7d32', padding: 8, borderRadius: 4,
                    marginTop: 10, fontSize: '0.82rem' }}>
                    ✅ All channels have boxes — ready to run alignment.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 5. Stage A: Feature Detection */}
          {loaded && alignMethod !== 'manual_diagonal' && (
            <div className="sidebar-section">
              <h3>🔍 Feature Detection</h3>
              <div className="sidebar-content">
                <button className="sidebar-btn" onClick={handleDetectFeatures}
                  disabled={isDetecting || isAligning || !selectedInputStage}
                  style={{ width: '100%', marginBottom: 8 }}>
                  {isDetecting ? 'Detecting...' : 'Detect Features'}
                </button>

                {featureDetectionError && (
                  <div style={{ background: '#ffebee', color: '#c62828', padding: 8, borderRadius: 4, fontSize: '0.82rem', marginBottom: 8 }}>
                    ⚠️ {featureDetectionError}
                  </div>
                )}

                {featureDetected && (
                  <>
                    <div style={{ background: '#e8f5e9', padding: 10, borderRadius: 4, marginBottom: 8, fontSize: '0.82rem', color: '#2e7d32' }}>
                      ✅ Features detected
                      <div style={{ marginTop: 6, color: '#555', lineHeight: 1.4 }}>{featureSummary}</div>
                    </div>

                    {/* Per-channel feature counts */}
                    <div style={{ marginBottom: 8 }}>
                      {availableItems.map(item => (
                        featureCounts[item.key] && (
                          <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', padding: '3px 0', borderBottom: '1px solid #eee' }}>
                            <span>{item.display_label}</span>
                            <span style={{ color: '#555' }}>{featureCountLabel(item.key)}</span>
                          </div>
                        )
                      ))}
                    </div>

                    {/* Preview layer selector */}
                    <div className="input-group" style={{ marginBottom: 8 }}>
                      <label style={{ fontSize: '0.82rem' }}>Preview Layer:</label>
                      <select value={selectedPreviewLayer}
                        onChange={e => setSelectedPreviewLayer(e.target.value as PreviewLayer)}>
                        {alignMethod === 'frequency_domain_fft' ? (<>
                          <option value="fft_spectrum">FFT Spectrum (peaks highlighted)</option>
                          <option value="synthetic_grid">Synthetic Grid Template</option>
                          <option value="intersections">Intersections</option>
                          <option value="all">All Combined</option>
                        </>) : (<>
                          <option value="preprocessed">Preprocessed (inverted)</option>
                          <option value="binary_mask">Binary Mask</option>
                          <option value="detected">{alignMethod === 'grid_line_hough' ? 'Detected Lines' : 'Detected Corners'}</option>
                          {alignMethod === 'grid_line_hough' && <option value="intersections">Intersections</option>}
                        </>)}
                      </select>
                    </div>

                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', cursor: 'pointer' }}>
                      <input type="checkbox" checked={showFeatureOverlay}
                        onChange={e => setShowFeatureOverlay(e.target.checked)} />
                      Show overlay in preview
                    </label>

                    {alignMethod === 'frequency_domain_fft' && fftParams.fft_mode === 'simple_correlation' && (
                      <div style={{ background: '#fff8e1', color: '#7a5c00', padding: 8, borderRadius: 4, fontSize: '0.8rem', marginTop: 8, border: '1px solid #ffe082' }}>
                        ⚠️ <strong>Aliasing risk:</strong> Repeating grid — simple PCC may lock onto wrong unit cell. Use "Grid Structure Detection" mode to detect actual grid spacing.
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* 6. Run Alignment */}
          {loaded && (
            <div className="sidebar-section">
              <h3>🎯 Run Alignment</h3>
              <div className="sidebar-content">
                {alignMethod === 'manual_diagonal' && !readyToAlign && (
                  <div style={{ background: '#f3e5f5', color: '#6a1b9a', padding: 8, borderRadius: 4, fontSize: '0.82rem', marginBottom: 10 }}>
                    Draw a diagonal box on every selected channel, then run alignment.
                  </div>
                )}
                {alignMethod !== 'manual_diagonal' && featureDetected && (
                  <div style={{ background: '#e8f5e9', color: '#2e7d32', padding: 6, borderRadius: 4, fontSize: '0.8rem', marginBottom: 8 }}>
                    ✅ Features detected — alignment will use detected grid structure.
                  </div>
                )}
                {alignMethod !== 'manual_diagonal' && !featureDetected && (
                  <div style={{ background: '#e3f2fd', color: '#1565c0', padding: 6, borderRadius: 4, fontSize: '0.8rem', marginBottom: 8 }}>
                    💡 Tip: run Detect Features first to preview grid quality (optional).
                  </div>
                )}
                <button className="sidebar-btn" onClick={handleRunAlignment}
                  disabled={
                    isAligning || isDetecting ||
                    !selectedInputStage ||
                    !readyToAlign
                  }
                  style={{ width: '100%' }}>
                  {isAligning
                    ? `Aligning ${selectedChannels.size} channels…`
                    : `Run Alignment (${selectedChannels.size} ch)`}
                </button>
              </div>
            </div>
          )}

          {/* 7. Crop Tool (post-alignment) */}
          {loaded && !!stages.aligned && (
            <div className="sidebar-section">
              <h3
                style={{ cursor: 'pointer', userSelect: 'none', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                onClick={() => setCropExpanded(p => !p)}
              >
                <span>✂️ Crop</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  {cropRect && (
                    <span style={{ fontSize: '0.72rem', background: '#e3f2fd', color: '#1565c0',
                      padding: '2px 6px', borderRadius: 10, fontWeight: 'normal' }}>
                      active
                    </span>
                  )}
                  <span style={{ fontSize: '0.8rem', color: '#888' }}>{cropExpanded ? '▲' : '▼'}</span>
                </span>
              </h3>
              {cropExpanded && (
                <div className="sidebar-content">
                  <div style={{ fontSize: '0.82rem', color: '#555', marginBottom: 10, lineHeight: 1.5 }}>
                    Toggle <strong>Crop</strong> in the Preview header, then drag on the
                    preview image to define a crop region (post-alignment only).
                    Non-destructive — affects export only.
                  </div>

                  {/* Numeric inputs */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                    {(['x', 'y', 'w', 'h'] as const).map(field => (
                      <div key={field} className="input-group" style={{ marginBottom: 0 }}>
                        <label style={{ fontSize: '0.78rem', marginBottom: 2 }}>
                          {field === 'x' ? 'X' : field === 'y' ? 'Y' : field === 'w' ? 'W' : 'H'}
                        </label>
                        <input
                          type="number"
                          min={0}
                          value={cropRect ? cropRect[field] : ''}
                          placeholder={field === 'x' || field === 'y' ? '0' : '—'}
                          style={{ width: '100%', padding: '4px 6px', fontSize: '0.82rem',
                            border: '1px solid #ddd', borderRadius: 4 }}
                          onChange={e => {
                            const v = parseInt(e.target.value, 10);
                            if (isNaN(v)) return;
                            setCropRect(prev => prev
                              ? { ...prev, [field]: v }
                              : { x: 0, y: 0, w: 0, h: 0, [field]: v }
                            );
                          }}
                        />
                      </div>
                    ))}
                  </div>

                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      className="sidebar-btn"
                      style={{ fontSize: '0.8rem', padding: '5px 10px',
                        background: cropRect ? '#e8f5e9' : '#f5f5f5',
                        color: cropRect ? '#2e7d32' : '#888',
                        border: cropRect ? '1px solid #81c784' : '1px solid #ddd' }}
                      disabled={!cropRect}
                      title="Crop is applied at export time"
                    >
                      {cropRect ? '✅ Crop applied' : 'No crop set'}
                    </button>
                    <button
                      className="sidebar-btn"
                      style={{ fontSize: '0.8rem', padding: '5px 10px',
                        background: '#fff3e0', color: '#b71c1c' }}
                      onClick={() => setCropRect(null)}
                      disabled={!cropRect}
                      title="Remove crop rectangle">
                      Clear
                    </button>
                  </div>

                  {!stages.aligned && (
                    <div style={{ background: '#fff8e1', color: '#7a5c00', padding: 6,
                      borderRadius: 4, fontSize: '0.78rem', marginTop: 8, border: '1px solid #ffe082' }}>
                      ⚠️ Run alignment first, then select the Aligned stage in the Color Overlay.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* 8. Save / Restore Alignment */}
          {loaded && (
            <div className="sidebar-section">
              <h3>💾 Save / Restore</h3>
              <div className="sidebar-content">
                {savedAt ? (
                  <div style={{ background: '#e8f5e9', color: '#2e7d32', padding: '6px 10px',
                    borderRadius: 4, fontSize: '0.78rem', marginBottom: 10, border: '1px solid #a5d6a7' }}>
                    Saved: {new Date(savedAt).toLocaleString()}
                  </div>
                ) : (
                  <div style={{ background: '#fff8e1', color: '#7a5c00', padding: '6px 10px',
                    borderRadius: 4, fontSize: '0.78rem', marginBottom: 10, border: '1px solid #ffe082' }}>
                    No saved alignment for this position
                  </div>
                )}

                {/* TIFF export preference */}
                <div style={{ background: '#f5f5f5', border: '1px solid #e0e0e0',
                  borderRadius: 4, padding: '8px 10px', marginBottom: 10 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8,
                    fontSize: '0.82rem', cursor: 'pointer', marginBottom: tiffExportPref.autoDownload ? 8 : 0 }}>
                    <input
                      type="checkbox"
                      checked={tiffExportPref.autoDownload}
                      disabled={alignmentRunId === null}
                      onChange={e => setTiffExportPref(p => ({ ...p, autoDownload: e.target.checked }))}
                    />
                    Download aligned TIFFs after save
                  </label>
                  {alignmentRunId === null && (
                    <div style={{ fontSize: '0.75rem', color: '#888', marginTop: 4, marginLeft: 22 }}>
                      (run alignment first)
                    </div>
                  )}
                  {tiffExportPref.autoDownload && (
                    <div style={{ display: 'flex', gap: 16, marginLeft: 22 }}>
                      {(['zip', 'composite'] as const).map(f => (
                        <label key={f} style={{ display: 'flex', alignItems: 'center', gap: 4,
                          fontSize: '0.8rem', cursor: 'pointer' }}>
                          <input
                            type="radio"
                            name="tiffFormat"
                            value={f}
                            checked={tiffExportPref.format === f}
                            onChange={() => setTiffExportPref(p => ({ ...p, format: f }))}
                          />
                          {f === 'zip' ? 'Individual channels' : 'Multi-channel composite'}
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <button className="sidebar-btn"
                    style={{ fontSize: '0.8rem', padding: '5px 10px', background: '#1976d2', color: '#fff' }}
                    onClick={saveAlignment}
                    title="Save current ROI boxes and shift vectors to localStorage">
                    Save
                  </button>
                  <button className="sidebar-btn"
                    style={{ fontSize: '0.8rem', padding: '5px 10px' }}
                    disabled={!localStorage.getItem(snapshotKey(selectedSample, selectedPosition))}
                    onClick={loadSavedAlignment}
                    title="Restore previously saved alignment state">
                    Load Saved
                  </button>
                  <button className="sidebar-btn"
                    style={{ fontSize: '0.8rem', padding: '5px 10px', background: '#fff3e0', color: '#b71c1c' }}
                    onClick={resetAlignment}
                    title="Clear saved alignment and reset ROI boxes">
                    Reset
                  </button>
                </div>

                {/* Export status line */}
                {exportStatus && (
                  <div style={{ marginTop: 8, fontSize: '0.8rem',
                    color: exportStatus.includes('✓') ? '#2e7d32' : exportStatus.includes('failed') || exportStatus.includes('error') ? '#c62828' : '#1565c0',
                    background: exportStatus.includes('✓') ? '#e8f5e9' : exportStatus.includes('failed') || exportStatus.includes('error') ? '#ffebee' : '#e3f2fd',
                    padding: '4px 8px', borderRadius: 4, border: '1px solid',
                    borderColor: exportStatus.includes('✓') ? '#a5d6a7' : exportStatus.includes('failed') || exportStatus.includes('error') ? '#ef9a9a' : '#90caf9' }}>
                    {exportStatus}
                  </div>
                )}
              </div>
            </div>
          )}

        </div>
        {/* ── END LEFT SIDEBAR ── */}

        {/* ── CENTER: Main Panel ── */}
        <div className="main-panel">
          <div className="preview-section">
            <div className="preview-header">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                📸 Preview
                <button
                  className="sidebar-btn"
                  style={{
                    fontSize: '0.75rem',
                    padding: '4px 8px',
                    background: previewCropMode ? '#1565c0' : '#fff',
                    color: previewCropMode ? '#fff' : '#1565c0',
                    border: '1px solid #1565c0',
                  }}
                  disabled={!loaded || currentStage !== 'aligned'}
                  onClick={() => setPreviewCropMode(p => !p)}
                  title={currentStage !== 'aligned' ? 'Crop mode is available on Aligned stage only' : 'Toggle interactive crop mode'}
                >
                  {previewCropMode ? 'Crop: ON' : 'Crop'}
                </button>
              </h3>
              {loaded && (
                <div className="preview-info">
                  <span><strong>Sample:</strong> {selectedSample}</span>
                  <span><strong>Position:</strong> {selectedPosition}</span>
                  <span><strong>Stage:</strong> {currentStage}</span>
                </div>
              )}
            </div>

            <div className="preview-controls">
              <div className="control-group">
                <label>Stage:</label>
                <select value={currentStage} onChange={e => setCurrentStage(e.target.value)} disabled={!loaded}>
                  {stages.raw && <option value="raw">Raw</option>}
                  {stages.processed && <option value="processed">Processed</option>}
                  {featureDetected && alignMethod !== 'manual_diagonal' && <option value="features">Feature Detection</option>}
                  {stages.aligned && <option value="aligned">Aligned</option>}
                </select>
              </div>
              <div className="control-group">
                <label>Channel:</label>
                <select
                  value={selectedPreviewChannel}
                  onChange={e => setSelectedPreviewChannel(e.target.value)}
                  disabled={!loaded || (previewCropMode && currentStage === 'aligned')}
                >
                  {availableItems.map(item => <option key={item.key} value={item.key}>{item.display_label}</option>)}
                </select>
              </div>
              {currentStage === 'features' && featureDetected && (
                <>
                  <div className="control-group">
                    <label>Layer:</label>
                    <select value={selectedPreviewLayer}
                      onChange={e => setSelectedPreviewLayer(e.target.value as PreviewLayer)}>
                      {alignMethod === 'frequency_domain_fft' ? (<>
                        <option value="fft_spectrum">FFT Spectrum</option>
                        <option value="synthetic_grid">Synthetic Grid</option>
                        <option value="intersections">Intersections</option>
                        <option value="all">All</option>
                      </>) : (<>
                        <option value="preprocessed">Preprocessed</option>
                        <option value="binary_mask">Binary Mask</option>
                        <option value="detected">{alignMethod === 'grid_line_hough' ? 'Lines' : 'Corners'}</option>
                        {alignMethod === 'grid_line_hough' && <option value="intersections">Intersections</option>}
                      </>)}
                    </select>
                  </div>
                  <div className="control-group">
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <input type="checkbox" checked={showFeatureOverlay}
                        onChange={e => setShowFeatureOverlay(e.target.checked)} />
                      Show overlay
                    </label>
                  </div>
                </>
              )}
            </div>

            <div className="preview-box" onContextMenu={(e) => e.preventDefault()}>
              {loaded ? (
                // After alignment runs and stage='aligned', show the warped result as a plain
                // image (so the change is visible).  For all other states keep the editor.
                showAlignedPreview ? (() => {
                  const src = getPreviewSrc();
                  const dbgUrl = src ?? '(none)';
                  console.log(
                    `[Alignment preview] channel=${selectedPreviewChannel}` +
                    ` stage=${currentStage}` +
                    ` alignedExists=${!!stages.aligned}` +
                    ` showAlignedPreview=true` +
                    ` src=${dbgUrl}`
                  );
                  return src
                    ? renderPreviewImage(src, `aligned - ${selectedPreviewChannel}`)
                    : <p>No aligned preview for {selectedPreviewChannel}</p>;
                })()
                : alignMethod === 'manual_diagonal' && diagImageSrc ? (
                  <ManualDiagonalEditor
                    imageUrl={diagImageSrc}
                    box={diagonalBoxes[selectedPreviewChannel] ?? null}
                    onBoxChange={newBox => {
                      diagonalBoxesRef.current = {
                        ...diagonalBoxesRef.current,
                        [selectedPreviewChannel]: newBox,
                      };
                      setDiagonalBoxes(prev => ({ ...prev, [selectedPreviewChannel]: newBox }));
                    }}
                    channelLabel={availableItems.find(i => i.key === selectedPreviewChannel)?.display_label}
                  />
                ) : (() => {
                  const src = getPreviewSrc();
                  return src
                    ? renderPreviewImage(src, `${currentStage} - ${selectedPreviewChannel}`)
                    : <p>No preview available for {currentStage} / {selectedPreviewChannel}</p>;
                })()
              ) : (
                <p>Load a sample and position to begin</p>
              )}
            </div>
          </div>

          {/* Alignment Color Overlay QC */}
          <AlignmentColorOverlay
            stages={stages}
            availableItems={availableItems}
            isLoaded={loaded}
            sample={selectedSample}
            position={selectedPosition}
            refChannel={lastAlignmentRefChannel}
            inputStage={lastAlignmentInputStage}
            shiftVectors={shiftVectors}
            onChannelHover={setHighlightedChannel}
            onChannelClick={key => setChannelVisibility(prev => ({ ...prev, [key]: !prev[key] }))}
            highlightedChannel={highlightedChannel}
            cropRect={cropRect}
            onCropRectChange={setCropRect}
            cropInteractive={false}
          />
        </div>

        {/* ── RIGHT: Shift Panel + Channel Info ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <AlignmentShiftPanel
            shiftVectors={shiftVectors as any}
            availableItems={availableItems}
            channels={channelsConfig}
            onChannelHover={setHighlightedChannel}
            onChannelClick={key => setChannelVisibility(prev => ({ ...prev, [key]: !prev[key] }))}
            highlightedChannel={highlightedChannel}
          />
          <ChannelInfoPanel
            stages={stages}
            currentStage={
              // ChannelInfoPanel only accepts known stage names; map 'features' → nearest real stage
              (currentStage === 'features'
                ? (stages.processed ? 'processed' : 'raw')
                : currentStage) as any
            }
            availableItems={availableItems}
            isLoaded={loaded}
            selectedSample={selectedSample}
            selectedPosition={selectedPosition}
            backendStats={backendStats}
          />
        </div>

      </div>
    </div>
  );
};

export default Alignment;
