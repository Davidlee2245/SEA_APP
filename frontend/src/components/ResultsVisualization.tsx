/**
 * Final Results Visualization
 * Displays output files and reference-centered exosome colocalization analysis.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ResultItem,
  ColocalizationStats,
  ComboAnalysis,
  ReferenceColocalizationRow,
  ExosomeColocalizationResult,
} from '../types/alignment';
import '../styles/ResultsVisualization.css';
import { getApiBase } from '../lib/apiBase';

function geometryChannelPositive(row: Record<string, unknown>, ch: string): boolean {
  const geomKey = `${ch}_positive_geom`;
  if (Object.prototype.hasOwnProperty.call(row, geomKey)) {
    return !!(row[geomKey] as boolean);
  }
  return !!(row[`${ch}_positive`] as boolean);
}

function channelComboLabelFromKeys(positiveChannelKeys: string[]): string {
  if (positiveChannelKeys.length === 0) return 'Negative';
  if (positiveChannelKeys.length === 1) return `${positiveChannelKeys[0]} only`;
  return [...positiveChannelKeys].sort().join(' + ');
}

function effectivePositiveChannels(row: Record<string, unknown>, markerChs: string[]): string[] {
  return markerChs.filter((ch) => geometryChannelPositive(row, ch));
}

function effectiveCombinationLabel(row: Record<string, unknown>, markerChs: string[]): string {
  return channelComboLabelFromKeys(effectivePositiveChannels(row, markerChs));
}

function effectiveTotalPositiveMarkerCount(
  row: Record<string, unknown>,
  markerChs: string[],
): number {
  return markerChs.reduce((sum, ch) => {
    if (!geometryChannelPositive(row, ch)) return sum;
    return sum + Number(row[`${ch}_count`] ?? 0);
  }, 0);
}

interface ResultsVisualizationProps {
  results: ResultItem[];
  sampleName: string;
  selectedSample?: string;
  selectedPosition?: string;
  availableChannels?: string[];
  channelMarkers?: Record<string, string>;
  searchedExosomePath?: string;
  colocalization?: ColocalizationStats;
  comboAnalysis?: ComboAnalysis;
  referenceColocalization?: ReferenceColocalizationRow[];
  exosomeColocalization?: ExosomeColocalizationResult;
}

const ResultsVisualization: React.FC<ResultsVisualizationProps> = ({ 
  results, 
  sampleName,
  selectedSample,
  selectedPosition,
  availableChannels = [],
  channelMarkers = {},
  searchedExosomePath,
  colocalization,
  comboAnalysis,
  referenceColocalization,
  exosomeColocalization,
}) => {
  const [resultItems, setResultItems] = useState<ResultItem[]>(results);
  const [selectedResult, setSelectedResult] = useState<ResultItem | null>(results.length > 0 ? results[0] : null);
  const [imageError, setImageError] = useState<boolean>(false);
  const [analysisMode, setAnalysisMode] = useState<'overlap' | 'nearest_centroid'>('nearest_centroid');
  const [distanceThreshold, setDistanceThreshold] = useState<number>(10);
  const [referenceChannel, setReferenceChannel] = useState<string>(availableChannels[0] || '');
  const [markerChannels, setMarkerChannels] = useState<string[]>(availableChannels.slice(1));
  const [colocData, setColocData] = useState<ExosomeColocalizationResult | null>(exosomeColocalization || null);
  const [isRunningColoc, setIsRunningColoc] = useState<boolean>(false);
  const [colocError, setColocError] = useState<string>('');
  const [colocSuccessMessage, setColocSuccessMessage] = useState<string>('');
  const [filteredReferenceObjectIds, setFilteredReferenceObjectIds] = useState<number[] | null>(null);
  const [isExportingIntensity, setIsExportingIntensity] = useState<boolean>(false);
  const [intensityExportError, setIntensityExportError] = useState<string>('');
  const [showDeltaIntensityColumns, setShowDeltaIntensityColumns] = useState<boolean>(true);
  /** When true, skip syncing `exosomeColocalization` prop over POST-derived colocData until context changes. */
  const colocDataFromRunRef = useRef(false);

  useEffect(() => {
    setResultItems(results);
    setSelectedResult(results.length > 0 ? results[0] : null);
  }, [results]);

  useEffect(() => {
    if (availableChannels.length === 0) return;
    if (!referenceChannel || !availableChannels.includes(referenceChannel)) {
      setReferenceChannel(availableChannels[0]);
    }
  }, [availableChannels, referenceChannel]);

  useEffect(() => {
    colocDataFromRunRef.current = false;
  }, [selectedSample, selectedPosition, referenceChannel]);

  useEffect(() => {
    if (colocDataFromRunRef.current) return;
    setColocData(exosomeColocalization || null);
  }, [exosomeColocalization, selectedSample, selectedPosition, referenceChannel]);

  // If precomputed colocalization results exist, sync the reference channel dropdown
  // to the reference channel used by the backend for that dataset.
  useEffect(() => {
    const refFromResults = colocData?.summary?.reference_channel;
    if (refFromResults && availableChannels.includes(refFromResults)) {
      setReferenceChannel(refFromResults);
    }
  }, [colocData?.summary?.reference_channel, availableChannels]);

  // Bridge Exosome Detection filters → Results Viewer:
  // ExosomeDetection persists filtered *object IDs* into localStorage keyed by (sample, position, channel).
  useEffect(() => {
    if (!selectedSample || !selectedPosition || !referenceChannel) {
      setFilteredReferenceObjectIds(null);
      return;
    }

    const storageKey = `sea_filtered_detections_${selectedSample}_${selectedPosition}_${referenceChannel}`;
    const raw = localStorage.getItem(storageKey);
    if (!raw) {
      setFilteredReferenceObjectIds(null);
      return;
    }

    try {
      const parsed = JSON.parse(raw);
      const enabled = parsed?.enabled ?? true;
      if (!enabled) {
        setFilteredReferenceObjectIds(null);
        return;
      }

      const ids: number[] = Array.isArray(parsed?.objectIds)
        ? parsed.objectIds
            .map((v: any) => Number(v))
            .filter((n: number) => Number.isFinite(n))
        : [];
      setFilteredReferenceObjectIds(ids);

      // Investigation logs (remove after debugging)
      try {
        const rawIds = Array.isArray(parsed?.objectIds) ? parsed.objectIds : [];
        const uniq = new Set(rawIds.map((v: any) => String(v)));
        console.log('[ResultsViewerFilter] key=', storageKey);
        console.log('[ResultsViewerFilter] enabled=', enabled, 'rawObjectIds.length=', rawIds.length, 'numericIds.length=', ids.length);
        console.log('[ResultsViewerFilter] rawIdTypeSample=', rawIds.slice(0, 5).map((v: any) => typeof v), 'rawIdSample=', rawIds.slice(0, 5));
        console.log('[ResultsViewerFilter] uniqueCount(raw)=', uniq.size, 'hasDuplicates(raw)=', uniq.size !== rawIds.length);
      } catch (e) {
        console.log('[ResultsViewerFilter] failed to log parsed payload', e);
      }
    } catch {
      setFilteredReferenceObjectIds(null);
    }
  }, [selectedSample, selectedPosition, referenceChannel]);

  const mapChannelIdsToMarkersInLabel = (label: string): string => {
    // Replace occurrences of e.g. "C1_ch1" with "p62" using metadata provided by the API.
    // If a marker name is missing, we keep the original channel identifier.
    let out = label;
    Object.entries(channelMarkers).forEach(([chId, markerName]) => {
      const marker = (markerName || '').trim();
      if (!marker) return;
      out = out.split(chId).join(marker);
    });
    return out;
  };

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

  const groupedResults = resultItems.reduce((acc, result) => {
    if (!acc[result.type]) {
      acc[result.type] = [];
    }
    acc[result.type].push(result);
    return acc;
  }, {} as Record<string, ResultItem[]>);

  const markerChannelOptions = useMemo(
    () => availableChannels.filter((ch) => ch !== referenceChannel),
    [availableChannels, referenceChannel]
  );

  const baseReferenceRowsForFilter = useMemo(() => {
    return (colocData?.reference_table || referenceColocalization || []) as ReferenceColocalizationRow[];
  }, [colocData, referenceColocalization]);

  // Investigation logs (remove after debugging)
  useEffect(() => {
    const baseN = baseReferenceRowsForFilter.length;
    const idSet = filteredReferenceObjectIds ? new Set(filteredReferenceObjectIds) : null;
    const filteredN = idSet === null
      ? baseReferenceRowsForFilter.length
      : baseReferenceRowsForFilter.filter((row) => idSet.has(row.reference_object_id)).length;
    const sampleIds = baseReferenceRowsForFilter.slice(0, 5).map((r: any) => r?.reference_object_id);
    const sampleTypes = sampleIds.map((v: any) => typeof v);
    console.log('[ResultsViewerFilter] reference_table base=', baseN, 'afterFilter=', filteredN);
    console.log('[ResultsViewerFilter] reference_object_id sample=', sampleIds, 'types=', sampleTypes);
  }, [baseReferenceRowsForFilter, filteredReferenceObjectIds]);

  const referenceRowsForDisplay = useMemo(() => {
    if (filteredReferenceObjectIds === null) return baseReferenceRowsForFilter;
    const idSet = new Set(filteredReferenceObjectIds);
    return baseReferenceRowsForFilter.filter((row) => idSet.has(row.reference_object_id));
  }, [baseReferenceRowsForFilter, filteredReferenceObjectIds]);

  const colocMarkerChs = useMemo(() => {
    const fromSummary = colocData?.summary?.marker_channels;
    if (Array.isArray(fromSummary) && fromSummary.length > 0) {
      return fromSummary;
    }
    return markerChannels;
  }, [colocData?.summary?.marker_channels, markerChannels]);

  const summaryForDisplay = useMemo(() => {
    if (!baseReferenceRowsForFilter || baseReferenceRowsForFilter.length === 0) return null;

    const total = referenceRowsForDisplay.length;
    const chs = colocMarkerChs;
    const positiveCount =
      chs.length > 0
        ? referenceRowsForDisplay.filter((r) =>
            chs.some((ch) => geometryChannelPositive(r as unknown as Record<string, unknown>, ch)),
          ).length
        : referenceRowsForDisplay.filter((r) => r.overall_status === 'Positive').length;
    const negativeCount = total - positiveCount;
    const matchedMarkers =
      chs.length > 0
        ? referenceRowsForDisplay.reduce(
            (sum, r) =>
              sum +
              effectiveTotalPositiveMarkerCount(r as unknown as Record<string, unknown>, chs),
            0,
          )
        : referenceRowsForDisplay.reduce((sum, r) => sum + Number(r.total_positive_marker_count || 0), 0);

    const rate = total > 0 ? (positiveCount / total) * 100.0 : 0.0;

    return {
      total_reference_objects: total,
      marker_positive_reference_objects: positiveCount,
      marker_negative_reference_objects: negativeCount,
      overall_positive_rate: Number(rate.toFixed(3)),
      total_matched_marker_objects: matchedMarkers,
      analysis_mode: colocData?.summary?.analysis_mode ?? analysisMode,
    };
  }, [
    baseReferenceRowsForFilter,
    referenceRowsForDisplay,
    colocData?.summary?.analysis_mode,
    analysisMode,
    colocMarkerChs,
  ]);

  const combinationSummaryForDisplay = useMemo(() => {
    const markerChs = colocMarkerChs;
    const total = referenceRowsForDisplay.length;
    if (total === 0) return colocData?.combination_summary || [];
    if (markerChs.length === 0) {
      return colocData?.combination_summary || [];
    }
    const counts = new Map<string, number>();
    referenceRowsForDisplay.forEach((row) => {
      const key = effectiveCombinationLabel(row as unknown as Record<string, unknown>, markerChs);
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return Array.from(counts.entries())
      .map(([combination, count]) => {
        const rate = total > 0 ? count / total : 0.0;
        return {
          combination,
          count,
          rate,
          percentage: rate * 100.0,
        };
      })
      .sort((a, b) => b.count - a.count);
  }, [
    referenceRowsForDisplay,
    colocData?.combination_summary,
    colocMarkerChs,
  ]);

  const multiChannelComboTableForDisplay = useMemo(() => {
    if (!comboAnalysis?.combo_table) return [];
    // If no filter was applied, show persisted combo results.
    if (filteredReferenceObjectIds === null) return comboAnalysis.combo_table;

    // When filtering, rebuild the multi-channel combo counts from the filtered reference rows.
    const total = referenceRowsForDisplay.length;
    const counts = new Map<string, number>();
    referenceRowsForDisplay.forEach((row) => {
      const key = row.biomarker_combination_label || 'Negative';
      counts.set(key, (counts.get(key) || 0) + 1);
    });

    const combos = Array.from(counts.entries())
      .map(([combination, count]) => {
        const rate = total > 0 ? count / total : 0.0;
        return {
          combo: [combination],
          count,
          rate,
          percentage: rate * 100.0,
        };
      })
      .sort((a, b) => b.count - a.count);
    return combos;
  }, [comboAnalysis, filteredReferenceObjectIds, referenceRowsForDisplay]);

  const channelSummaryForDisplay = useMemo(() => {
    const rows = referenceRowsForDisplay;
    const chs = colocMarkerChs;
    if (!rows.length || !chs.length) {
      return colocData?.channel_summary || [];
    }
    const base = colocData?.channel_summary || [];
    const refTotal = rows.length;
    return chs.map((ch) => {
      const meta = base.find((b) => b.channel === ch);
      const positiveRows = rows.filter((r) =>
        geometryChannelPositive(r as unknown as Record<string, unknown>, ch),
      );
      const posCount = positiveRows.length;
      const avgCount =
        posCount > 0
          ? positiveRows.reduce(
              (s, r) => s + Number((r as unknown as Record<string, unknown>)[`${ch}_count`] ?? 0),
              0,
            ) / posCount
          : 0;
      const nearestVals = positiveRows
        .map((r) => (r as unknown as Record<string, unknown>)[`${ch}_nearest_distance`])
        .filter((v) => v != null && Number.isFinite(Number(v)))
        .map((v) => Number(v));
      const medianNearest =
        nearestVals.length > 0
          ? [...nearestVals].sort((a, b) => a - b)[Math.floor(nearestVals.length / 2)]
          : null;
      return {
        channel: ch,
        biomarker: meta?.biomarker ?? channelMarkers[ch] ?? '',
        total_marker_objects: meta?.total_marker_objects ?? 0,
        positive_reference_objects: posCount,
        positive_rate: refTotal > 0 ? (posCount / refTotal) * 100.0 : 0,
        avg_marker_count_per_positive_reference: avgCount,
        median_nearest_distance: medianNearest,
      };
    });
  }, [referenceRowsForDisplay, colocMarkerChs, colocData?.channel_summary, channelMarkers]);

  const toggleMarkerChannel = (channel: string) => {
    setMarkerChannels((prev) =>
      prev.includes(channel) ? prev.filter((c) => c !== channel) : [...prev, channel]
    );
  };

  const handleRunColocalization = async () => {
    if (!selectedSample || !selectedPosition) {
      setColocError('Sample and position are required to run colocalization.');
      return;
    }
    if (!referenceChannel) {
      setColocError('Please select a reference channel.');
      return;
    }
    if (markerChannels.length === 0) {
      setColocError('Please select at least one marker channel.');
      return;
    }

    setIsRunningColoc(true);
    setColocError('');
    setColocSuccessMessage('');
    try {
      const response = await fetch(`${getApiBase()}/api/colocalization_analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
          reference_channel: referenceChannel,
          marker_channels: markerChannels,
          analysis_mode: analysisMode,
          distance_threshold: distanceThreshold,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        throw new Error(data.error || data.message || 'Colocalization analysis failed');
      }

      setColocData(data.data || null);
      colocDataFromRunRef.current = true;

      const msg =
        typeof data.message === 'string' && data.message.length > 0
          ? data.message
          : `Analysis complete. Results saved to ${typeof data.results_path_display === 'string' ? data.results_path_display : `${selectedSample}/results/${selectedPosition}/`}`;
      setColocSuccessMessage(msg);
    } catch (err) {
      setColocSuccessMessage('');
      setColocError(err instanceof Error ? err.message : 'Colocalization analysis failed');
    } finally {
      setIsRunningColoc(false);
    }
  };

  const handleDownloadExcel = async () => {
    if (!selectedSample || !selectedPosition) return;

    try {
      const response = await fetch(`${getApiBase()}/api/exosome/export_excel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
          colocalization_rows: referenceColocalization ?? [],
        }),
      });

      if (!response.ok) {
        let message = 'Excel export failed';
        try {
          const err = await response.json();
          if (err?.message) message = err.message;
        } catch {
          // Ignore parse failures and keep fallback message.
        }
        alert(`Export failed: ${message}`);
        return;
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedSample}_${selectedPosition}_results.xlsx`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Export error: ${err?.message || 'Unknown error'}`);
    }
  };

  const handleDownloadIntensityExcel = async () => {
    if (!selectedSample || !selectedPosition) return;
    setIsExportingIntensity(true);
    setIntensityExportError('');

    try {
      const res = await fetch(`${getApiBase()}/api/exosome/export_intensity_excel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: selectedSample,
          position: selectedPosition,
        }),
      });

      if (!res.ok) {
        let message = 'Intensity export failed';
        try {
          const err = await res.json();
          if (err?.message) message = err.message;
        } catch {
          // keep fallback message
        }
        setIntensityExportError(`Export failed: ${message}`);
        return;
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedSample}_${selectedPosition}_intensity.xlsx`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setIntensityExportError(`Error: ${err?.message || 'Unknown error'}`);
    } finally {
      setIsExportingIntensity(false);
    }
  };

  return (
    <div className="results-visualization">
      <div className="view-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
        <div>
          <h3>Final Results</h3>
          <p className="description">
            View and download all output files generated from the aligned images.
            Results include overlays, quantification data, and processed images.
          </p>
        </div>
        <button
          className="download-button"
          onClick={handleDownloadExcel}
          disabled={!selectedSample || !selectedPosition}
        >
          ⬇️ Download Excel
        </button>
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
      {multiChannelComboTableForDisplay.length > 0 && (
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
                  {multiChannelComboTableForDisplay
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
                      // Format combination label (then replace channel IDs with marker names)
                      const comboLabelRaw = entry.combo.join(' + ');
                      const comboLabel = mapChannelIdsToMarkersInLabel(comboLabelRaw);
                      
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

      <div className="coloc-controls">
        <h4>Reference-Centered Colocalization</h4>
        <div className="coloc-control-grid">
          <div className="input-group">
            <label>Reference channel</label>
            <select
              value={referenceChannel}
              onChange={(e) => {
                const nextRef = e.target.value;
                setReferenceChannel(nextRef);
                setMarkerChannels((prev) => prev.filter((c) => c !== nextRef));
              }}
            >
              <option value="">-- Select reference channel --</option>
              {availableChannels.map((ch) => (
                <option key={ch} value={ch}>
                  {ch}{channelMarkers[ch] ? ` (${channelMarkers[ch]})` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <label>Analysis mode</label>
            <select
              value={analysisMode}
              onChange={(e) => setAnalysisMode(e.target.value as 'overlap' | 'nearest_centroid')}
            >
              <option value="overlap">Overlap</option>
              <option value="nearest_centroid">Nearest Centroid</option>
            </select>
          </div>

          <div className="input-group">
            <label>Distance threshold (px)</label>
            <input
              type="number"
              min={0}
              step={0.5}
              value={distanceThreshold}
              disabled={analysisMode !== 'nearest_centroid'}
              onChange={(e) => setDistanceThreshold(Number(e.target.value))}
            />
          </div>

          <div className="input-group marker-select">
            <label>Marker channels</label>
            <div className="marker-checkboxes">
              {markerChannelOptions.map((ch) => (
                <label key={ch} className="marker-option">
                  <input
                    type="checkbox"
                    checked={markerChannels.includes(ch)}
                    onChange={() => toggleMarkerChannel(ch)}
                  />
                  {ch}{channelMarkers[ch] ? ` (${channelMarkers[ch]})` : ''}
                </label>
              ))}
            </div>
          </div>
        </div>
        {availableChannels.length === 0 && (
          <p className="coloc-error">
            No exported channels were discovered for this sample/position.
            {searchedExosomePath ? ` Searched path: ${searchedExosomePath}` : ''}
          </p>
        )}
        <button className="download-button" onClick={handleRunColocalization} disabled={isRunningColoc}>
          {isRunningColoc ? 'Running Analysis...' : 'Run Colocalization Analysis'}
        </button>
        {colocError && <p className="coloc-error">{colocError}</p>}
        {colocSuccessMessage && !colocError && <p className="coloc-success">{colocSuccessMessage}</p>}
      </div>

      {(colocData || referenceColocalization) && (
        <div className="exosome-coloc-dashboard">
          <h4>Reference Object Dashboard</h4>

          {summaryForDisplay && (
            <div className="summary-grid">
              <div className="summary-item"><strong>Total reference:</strong> {summaryForDisplay.total_reference_objects}</div>
              <div className="summary-item"><strong>Marker-positive:</strong> {summaryForDisplay.marker_positive_reference_objects}</div>
              <div className="summary-item"><strong>Marker-negative:</strong> {summaryForDisplay.marker_negative_reference_objects}</div>
              <div className="summary-item"><strong>Positive rate:</strong> {summaryForDisplay.overall_positive_rate}%</div>
              <div className="summary-item"><strong>Total matched markers:</strong> {summaryForDisplay.total_matched_marker_objects}</div>
              <div className="summary-item"><strong>Mode:</strong> {summaryForDisplay.analysis_mode}</div>
            </div>
          )}

          {channelSummaryForDisplay && channelSummaryForDisplay.length > 0 && (
            <div className="combo-table-container" style={{ marginTop: '1rem' }}>
              <h5>Biomarker / Channel Summary</h5>
              <table className="combo-table">
                <thead>
                  <tr>
                    <th>Channel</th>
                    <th>Biomarker</th>
                    <th>Total Marker Objects</th>
                    <th>Positive Reference Objects</th>
                    <th>Positive Rate</th>
                    <th>Avg Count / Positive Ref</th>
                    <th>Median Nearest Distance</th>
                  </tr>
                </thead>
                <tbody>
                  {channelSummaryForDisplay.map((row, idx) => (
                    <tr key={`${row.channel}-${idx}`}>
                      <td>{row.channel}</td>
                      <td>{row.biomarker || '-'}</td>
                      <td>{row.total_marker_objects}</td>
                      <td>{row.positive_reference_objects}</td>
                      <td>{row.positive_rate.toFixed(2)}%</td>
                      <td>{row.avg_marker_count_per_positive_reference.toFixed(3)}</td>
                      <td>{row.median_nearest_distance ?? '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {combinationSummaryForDisplay && combinationSummaryForDisplay.length > 0 && (
            <div className="combo-table-container" style={{ marginTop: '1rem' }}>
              <h5>Channel Combination Summary</h5>
              <table className="combo-table">
                <thead>
                  <tr>
                    <th>Combination</th>
                    <th>Count</th>
                    <th>Rate</th>
                    <th>Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {combinationSummaryForDisplay.map((row, idx) => (
                    <tr key={`${row.combination}-${idx}`}>
                      <td>{mapChannelIdsToMarkersInLabel(row.combination)}</td>
                      <td>{row.count}</td>
                      <td>{row.rate.toFixed(4)}</td>
                      <td>{row.percentage.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {baseReferenceRowsForFilter && baseReferenceRowsForFilter.length > 0 && (
            <div className="combo-table-container" style={{ marginTop: '1rem' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                  flexWrap: 'wrap',
                  marginBottom: '8px',
                }}
              >
                <h5 style={{ margin: 0 }}>Reference Object Table</h5>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showDeltaIntensityColumns}
                    onChange={(e) => setShowDeltaIntensityColumns(e.target.checked)}
                  />
                  Show ΔI (background-subtracted intensity)
                </label>
                {!showDeltaIntensityColumns && (
                  <span style={{ color: '#666', fontSize: '13px' }}>ΔI columns hidden</span>
                )}
              </div>
              <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                <table className="combo-table">
                  <thead>
                    <tr>
                      <th>Ref ID</th>
                      <th>X</th>
                      <th>Y</th>
                      <th>Area</th>
                      <th>Perimeter</th>
                      <th>Circularity</th>
                      {showDeltaIntensityColumns &&
                        colocMarkerChs.map((ch) => {
                          const bgVal = colocData?.background_intensities?.[ch];
                          const hasBg = bgVal != null && Number.isFinite(Number(bgVal));
                          const markerLabel = (channelMarkers[ch] || ch).trim() || ch;
                          return (
                            <th
                              key={`int-${ch}`}
                              title={hasBg ? `Background mean: ${Number(bgVal).toFixed(1)}` : undefined}
                            >
                              <div>{markerLabel} ΔI</div>
                              {hasBg && (
                                <div style={{ fontSize: '11px', fontWeight: 400, color: '#555' }}>
                                  BG: {Number(bgVal).toFixed(1)}
                                </div>
                              )}
                            </th>
                          );
                        })}
                      <th>Combination</th>
                      <th>Positive Marker Count</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {referenceRowsForDisplay.slice(0, 500).map((row, idx) => {
                      const rr = row as unknown as Record<string, unknown>;
                      const effCombo = effectiveCombinationLabel(rr, colocMarkerChs);
                      const effCount = effectiveTotalPositiveMarkerCount(rr, colocMarkerChs);
                      const effPos =
                        colocMarkerChs.length > 0
                          ? colocMarkerChs.some((ch) => geometryChannelPositive(rr, ch))
                          : row.overall_status === 'Positive';
                      return (
                        <tr key={`${row.reference_object_id}-${idx}`}>
                          <td>{row.reference_object_id}</td>
                          <td>{row.reference_centroid_x?.toFixed?.(2) ?? row.reference_centroid_x}</td>
                          <td>{row.reference_centroid_y?.toFixed?.(2) ?? row.reference_centroid_y}</td>
                          <td>{row.reference_area?.toFixed?.(2) ?? row.reference_area}</td>
                          <td>{row.reference_perimeter?.toFixed?.(2) ?? row.reference_perimeter}</td>
                          <td>{row.reference_circularity?.toFixed?.(3) ?? row.reference_circularity}</td>
                          {showDeltaIntensityColumns &&
                            colocMarkerChs.map((ch) => {
                              const geomPos = geometryChannelPositive(rr, ch);
                              const raw = rr[`${ch}_intensity_bg_subtracted`];
                              const v =
                                raw === undefined || raw === null || raw === ''
                                  ? null
                                  : Number(raw);
                              const showNumber =
                                geomPos && v != null && Number.isFinite(v);
                              const bgStyle: React.CSSProperties = !geomPos
                                ? { backgroundColor: '#f8f9fa', color: '#868e96' }
                                : !showNumber
                                  ? { backgroundColor: '#f1f3f5', color: '#868e96' }
                                  : v! > 0
                                    ? { backgroundColor: '#d3f9d8' }
                                    : { backgroundColor: '#e9ecef', color: '#495057' };
                              return (
                                <td key={`${ch}-ibg-${idx}`} style={{ ...bgStyle, textAlign: 'right' }}>
                                  {showNumber ? v!.toFixed(1) : '—'}
                                </td>
                              );
                            })}
                          <td>{mapChannelIdsToMarkersInLabel(effCombo)}</td>
                          <td>{effCount}</td>
                          <td>{effPos ? 'Positive' : 'Negative'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <details className="intensity-export-section" style={{ marginTop: '24px' }}>
        <summary style={{ cursor: 'pointer', fontWeight: 600, fontSize: '1.05rem' }}>
          📊 Per-Object Marker Intensity
        </summary>
        <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>
          Download raw marker intensity values extracted from original TIFF images at each
          detected object location. One row per reference object and one intensity column per marker channel.
        </p>
        <button
          className="download-button"
          onClick={handleDownloadIntensityExcel}
          disabled={!selectedSample || !selectedPosition || isExportingIntensity}
          style={{ marginTop: '12px' }}
        >
          {isExportingIntensity ? '⏳ Extracting intensities...' : '⬇️ Download Intensity Excel'}
        </button>
        {intensityExportError && (
          <p style={{ color: 'red', marginTop: '8px' }}>{intensityExportError}</p>
        )}
      </details>

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

              <div className="result-display" onContextMenu={(e) => e.preventDefault()}>
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
                      onContextMenu={(e) => e.preventDefault()}
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
            <strong>Total Files:</strong> {resultItems.length}
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

