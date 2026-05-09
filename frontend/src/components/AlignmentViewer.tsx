/**
 * Results Viewer Component
 * Uses Sample -> Position -> Load Position workflow
 * and renders final outputs directly.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { AlignmentResult } from '../types/alignment';
import ResultsVisualization from './ResultsVisualization';
import '../styles/AlignmentViewer.css';
import { getApiBase } from '../lib/apiBase';

const DEFAULT_SAMPLE = 'A2780Cis10';
const DEFAULT_POSITION = 'P1';

const AlignmentViewer: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [selectedPosition, setSelectedPosition] = useState<string>('');
  const [availableSamples, setAvailableSamples] = useState<string[]>([]);
  const [availablePositions, setAvailablePositions] = useState<string[]>([]);
  const [isLoadingPosition, setIsLoadingPosition] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [resultsData, setResultsData] = useState<AlignmentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const autoLoadFiredRef = useRef(false);
  const [pendingAutoLoad, setPendingAutoLoad] = useState(false);

  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await fetch(`${getApiBase()}/api/input/samples`);
        const data = await response.json();
        if (data.success) {
          setAvailableSamples(data.data);
          if (!selectedSample && data.data.includes(DEFAULT_SAMPLE)) {
            setSelectedSample(DEFAULT_SAMPLE);
          }
        }
      } catch (err) {
        console.error('Failed to fetch samples:', err);
      }
    };
    fetchSamples();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedSample) {
      setAvailablePositions([]);
      setSelectedPosition('');
      setLoaded(false);
      setResultsData(null);
      return;
    }

    const fetchPositions = async () => {
      try {
        const response = await fetch(`${getApiBase()}/api/input/samples/${selectedSample}/positions`);
        const data = await response.json();
        if (data.success) {
          setAvailablePositions(data.data);
          if (!selectedPosition && data.data.includes(DEFAULT_POSITION)) {
            setSelectedPosition(DEFAULT_POSITION);
            if (selectedSample === DEFAULT_SAMPLE && !autoLoadFiredRef.current) {
              setPendingAutoLoad(true);
            }
          }
        }
      } catch (err) {
        console.error('Failed to fetch positions:', err);
      }
    };

    fetchPositions();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSample]);

  const loadResultsData = async (sample: string, position: string) => {
    setLoading(true);
    setError(null);

    try {
      // Preferred path: exosome counting outputs for selected sample/position.
      const exosomeResponse = await fetch(
        `${getApiBase()}/api/exosome_results_by_position?sample=${encodeURIComponent(sample)}&position=${encodeURIComponent(position)}`
      );
      if (exosomeResponse.ok) {
        const exosomeResult = await exosomeResponse.json();
        if (exosomeResult.success && exosomeResult.data) {
          setResultsData(exosomeResult.data as AlignmentResult);
          setLoaded(true);
          return;
        }
      } else {
        let exosomeError = `Exosome detection results not found for ${sample}/${position}`;
        try {
          const errPayload = await exosomeResponse.json();
          if (errPayload?.error) exosomeError = errPayload.error;
          if (errPayload?.debug) {
            console.warn('[ResultsViewer] exosome debug trace', errPayload.debug);
          }
        } catch {
          // ignore parse errors and keep default message
        }

        // Optional fallback: alignment-by-position outputs, only when exosome outputs are unavailable.
        const alignmentResponse = await fetch(
          `${getApiBase()}/api/alignment_by_position?sample=${encodeURIComponent(sample)}&position=${encodeURIComponent(position)}`
        );
        if (alignmentResponse.ok) {
          const alignmentResult = await alignmentResponse.json();
          if (alignmentResult.success && alignmentResult.data) {
            setResultsData(alignmentResult.data as AlignmentResult);
            setLoaded(true);
            return;
          }
        }

        throw new Error(exosomeError);
      }
    } catch (err) {
      setLoaded(false);
      setResultsData(null);
      setError(err instanceof Error ? err.message : 'Failed to load exosome detection results');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadPosition = async () => {
    if (!selectedSample || !selectedPosition) {
      alert('Please select both sample and position');
      return;
    }

    setIsLoadingPosition(true);

    try {
      // Keep workflow consistent with other tabs.
      const loadResponse = await fetch(`${getApiBase()}/api/input/load_position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
        }),
      });
      const loadData = await loadResponse.json();
      if (!loadData.success) {
        throw new Error(loadData.error || 'Failed to load position');
      }

      await loadResultsData(selectedSample, selectedPosition);
    } catch (err) {
      setLoaded(false);
      setResultsData(null);
      setError(err instanceof Error ? err.message : 'Failed to load position');
    } finally {
      setIsLoadingPosition(false);
    }
  };

  // Auto-load default sample/position once.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!pendingAutoLoad || autoLoadFiredRef.current) return;
    autoLoadFiredRef.current = true;
    setPendingAutoLoad(false);
    handleLoadPosition();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAutoLoad]);

  // ── Derived data — hooks must be called unconditionally before any early return ──

  const derivedChannelsFromResults = useMemo(() => {
    const channels = new Set<string>();
    for (const item of (resultsData?.finalResults || [])) {
      const m = item.url.match(/\/api\/results\/exosome_detection\/[^/]+\/[^/]+\/([^/]+)\//);
      if (m?.[1]) channels.add(m[1]);
    }
    return Array.from(channels).sort();
  }, [resultsData]);

  const availableChannelOptions = useMemo(() => {
    if (!resultsData) return [];
    if (resultsData.channels && resultsData.channels.length > 0) {
      return resultsData.channels
        .filter((c: any) => c.included !== false)
        .map((c: any) => c.id);
    }
    if (resultsData.metadata?.exosomeChannels && resultsData.metadata.exosomeChannels.length > 0) {
      return resultsData.metadata.exosomeChannels;
    }
    return derivedChannelsFromResults;
  }, [resultsData, derivedChannelsFromResults]);

  const markerMap = useMemo(() => {
    if (!resultsData) return {};
    const m: Record<string, string> = { ...(resultsData.metadata?.channelMarkers || {}) };
    for (const ch of (resultsData.channels || [] as any[])) {
      if (ch.id && ch.biomarker && !m[ch.id]) m[ch.id] = ch.biomarker;
    }
    return m;
  }, [resultsData]);

  // ── Early returns (after all hooks) ──────────────────────────────────────────

  if (loading) {
    return (
      <div className="alignment-viewer loading">
        <div className="spinner"></div>
        <p>Loading results for {selectedSample}/{selectedPosition}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alignment-viewer error">
        <h2>Error Loading Exosome Results</h2>
        <p>{error}</p>
        <button onClick={handleLoadPosition}>Retry</button>
      </div>
    );
  }

  if (!loaded || !resultsData) {
    return (
      <div className="alignment-viewer">
        <header className="viewer-header">
          <h1>SEA Results Viewer</h1>
          <div className="metadata">
            <div className="input-group">
              <label>Sample:</label>
              <select
                value={selectedSample}
                onChange={(e) => setSelectedSample(e.target.value)}
                disabled={availableSamples.length === 0 || isLoadingPosition}
              >
                <option value="">-- Select a sample --</option>
                {availableSamples.map((sample) => (
                  <option key={sample} value={sample}>
                    {sample}
                  </option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>Position:</label>
              <select
                value={selectedPosition}
                onChange={(e) => setSelectedPosition(e.target.value)}
                disabled={availablePositions.length === 0 || isLoadingPosition}
              >
                <option value="">-- Select position --</option>
                {availablePositions.map((position) => (
                  <option key={position} value={position}>
                    {position}
                  </option>
                ))}
              </select>
            </div>
            <button
              className="step-button active"
              onClick={handleLoadPosition}
              disabled={!selectedSample || !selectedPosition || isLoadingPosition}
            >
              {isLoadingPosition ? 'Loading...' : 'Load Position'}
            </button>
          </div>
        </header>
        <div className="step-content">
          <p>Select a sample and position, then click Load Position to view final results.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="alignment-viewer">
      <header className="viewer-header">
        <h1>SEA Results Viewer</h1>
        <div className="metadata">
          <div className="input-group">
            <label>Sample:</label>
            <select
              value={selectedSample}
              onChange={(e) => setSelectedSample(e.target.value)}
              disabled={availableSamples.length === 0 || isLoadingPosition}
            >
              <option value="">-- Select a sample --</option>
              {availableSamples.map((sample) => (
                <option key={sample} value={sample}>
                  {sample}
                </option>
              ))}
            </select>
          </div>
          <div className="input-group">
            <label>Position:</label>
            <select
              value={selectedPosition}
              onChange={(e) => setSelectedPosition(e.target.value)}
              disabled={availablePositions.length === 0 || isLoadingPosition}
            >
              <option value="">-- Select position --</option>
              {availablePositions.map((position) => (
                <option key={position} value={position}>
                  {position}
                </option>
              ))}
            </select>
          </div>
          <button
            className="step-button active"
            onClick={handleLoadPosition}
            disabled={!selectedSample || !selectedPosition || isLoadingPosition}
          >
            {isLoadingPosition ? 'Loading...' : 'Load Position'}
          </button>
        </div>
      </header>

      <main className="step-content">
        <ResultsVisualization 
          results={resultsData.finalResults}
          sampleName={resultsData.sampleName}
          selectedSample={selectedSample}
          selectedPosition={selectedPosition}
          availableChannels={availableChannelOptions}
          channelMarkers={markerMap}
          searchedExosomePath={resultsData.metadata?.searchedExosomePath}
          colocalization={resultsData.colocalization}
          comboAnalysis={resultsData.comboAnalysis}
          referenceColocalization={resultsData.referenceColocalization}
          exosomeColocalization={resultsData.exosomeColocalization}
        />
      </main>
    </div>
  );
};

export default AlignmentViewer;

