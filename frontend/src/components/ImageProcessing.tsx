/**
 * Image Processing Component
 * Dedicated to image preprocessing operations (contrast enhancement, normalization, filtering, etc.)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import ProgressBar from './ProgressBar';
import ChannelInfoPanel from './ChannelInfoPanel';
import '../styles/PipelineControl.css';
import { getApiBase, getWsBase } from '../lib/apiBase';

interface PipelineStatus {
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_step: string | null;
  current_sample: string | null;
  progress: number;
  total_samples: number;
  completed_samples: number;
  error: string | null;
  error_details: string | null;
}

interface PreviewStage {
  [channel: string]: string; // channel -> preview URL
}

interface ChannelItem {
  key: string;  // e.g., "C1_ch1"
  cycle: string;  // e.g., "C1"
  channel: string;  // e.g., "Ch1"
  marker: string | null;  // e.g., "p62"
  display_label: string;  // e.g., "C1_ch1(p62)"
  tiff_path: string;
  preview_url: string | null;
}

interface ChannelPreprocessParams {
  // Contrast enhancement params
  contrastMethod?: 'CLAHE' | 'Linear Stretch';
  claheClipLimit?: number;
  claheTileSize?: number;
  stretchPLow?: number;
  stretchPHigh?: number;
  
  // Step params (to be extended)
  step1Params?: any;
  step2Params?: any;
  step3Params?: any;
  step4Params?: any;
  
  // Current preprocessing stage for this channel
  currentStage?: 'raw' | 'contrast_enhance' | 'step1' | 'step2' | 'step3' | 'step4';
}

interface ImageProcessingState {
  // Selection state
  selectedSample: string;
  selectedPosition: string;
  selectedChannel: string;  // Internal key: e.g., "C1_ch1"
  
  // Available options
  availableSamples: string[];
  availablePositions: string[];
  availableItems: ChannelItem[];  // Replaces availableChannels
  
  // Preview state - simplified to Raw and Processed (Final)
  loaded: boolean;
  stages: {
    raw?: PreviewStage;
    processed?: PreviewStage;  // Final processed output after all steps
  };
  currentStage: 'raw' | 'processed';  // Simplified to just raw and processed
  
  // Internal tracking of intermediate stages (for pipeline execution)
  internalStages: {
    contrast_enhance?: PreviewStage;
    step1?: PreviewStage;
    step2?: PreviewStage;
    step3?: PreviewStage;
    step4?: PreviewStage;
  };
  
  // Preprocessing tracking (critical for pipeline flow)
  currentPreprocessStage: 'raw' | 'contrast_enhance' | 'step1' | 'step2' | 'step3' | 'step4';
  
  // Per-channel preprocessing state
  channelPreprocessParams: {
    [channelKey: string]: ChannelPreprocessParams;
  };
  
  // Global preprocessing params (for backward compatibility and "Apply to All")
  globalContrastMethod: 'CLAHE' | 'Linear Stretch';
  globalClaheClipLimit: number;
  globalClaheTileSize: number;
  globalStretchPLow: number;
  globalStretchPHigh: number;
  
  // Session management
  versionHash: string | null;
  showAdvanced: boolean;  // Toggle for showing intermediate stages

  // Backend 16-bit stats for ChannelInfoPanel (same source as Alignment stage)
  backendStats: Record<string, any>;

  // Busy flags
  isLoadingPosition: boolean;
  isProcessing: boolean;
}

const ImageProcessing: React.FC = () => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  
  // Ref to track if we're currently auto-triggering preprocessing (to prevent loops)
  const autoTriggeringRef = useRef<string>('');
  
  // Ref to track if we just reset (to prevent auto-fetch from restoring processed images)
  const justResetRef = useRef<string>('');

  // Unified image processing state
  const [processingState, setProcessingState] = useState<ImageProcessingState>({
    selectedSample: '',
    selectedPosition: '',
    selectedChannel: '',
    availableSamples: [],
    availablePositions: [],
    availableItems: [],
    loaded: false,
    stages: {},
    currentStage: 'processed',  // Default to processed (final)
    internalStages: {},
    currentPreprocessStage: 'raw',
    channelPreprocessParams: {},
    globalContrastMethod: 'CLAHE',
    globalClaheClipLimit: 2.0,
    globalClaheTileSize: 8,
    globalStretchPLow: 1,
    globalStretchPHigh: 99,
    versionHash: null,
    showAdvanced: false,
    backendStats: {},
    isLoadingPosition: false,
    isProcessing: false,
  });

  // Fetch status function - defined first
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('${getApiBase()}/api/pipeline/status');
      const data = await response.json();
      if (data.success) {
        setStatus(data.data);
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  }, []);

  // WebSocket message handler - memoized to prevent reconnections
  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === 'status') {
      setStatus(message.data);
    } else if (message.type === 'progress') {
      fetchStatus();
    } else if (message.type === 'log') {
      console.log('[Pipeline Log]', message.message || '');
    }
  }, [fetchStatus]);

  // WebSocket connection for real-time updates
  const { isConnected } = useWebSocket({
    url: `${getWsBase()}/ws/pipeline`,
    reconnect: false,
    onMessage: handleWebSocketMessage,
    onError: (error) => {
      console.error('WebSocket error in ImageProcessing:', error);
    },
  });

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);
  
  // Contrast Enhancement settings
  const [contrastMethod, setContrastMethod] = useState<string>('CLAHE');
  const [claheClipLimit, setClaheClipLimit] = useState<number>(2.0);
  const [claheTileSize, setClaheTileSize] = useState<number>(8);
  const [stretchPLow, setStretchPLow] = useState<number>(1);
  const [stretchPHigh, setStretchPHigh] = useState<number>(99);
  
  // Background Subtraction and Gaussian Blur settings
  const [backgroundSubtractionStrength, setBackgroundSubtractionStrength] = useState<number>(50);
  const [gaussianBlurSigma, setGaussianBlurSigma] = useState<number>(1.0);
  
  // Apply to all channels or just selected channel
  const [applyToAllChannels, setApplyToAllChannels] = useState<boolean>(false);

  // Fetch input samples on mount
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await fetch('${getApiBase()}/api/input/samples');
        const data = await response.json();
        if (data.success) {
          setProcessingState(prev => ({ ...prev, availableSamples: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch input samples:', err);
      }
    };
    fetchSamples();
  }, []);

  // Fetch positions when sample changes
  useEffect(() => {
    if (!processingState.selectedSample) {
      setProcessingState(prev => ({ 
        ...prev, 
        availablePositions: [], 
        selectedPosition: '',
        loaded: false,
        stages: {},
      }));
      return;
    }

    const fetchPositions = async () => {
      try {
        const response = await fetch(
          `${getApiBase()}/api/input/samples/${processingState.selectedSample}/positions`
        );
        const data = await response.json();
        if (data.success) {
          setProcessingState(prev => ({ ...prev, availablePositions: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch positions:', err);
      }
    };
    fetchPositions();
  }, [processingState.selectedSample]);

  // Track when we've fetched processed final to prevent loops
  const fetchedProcessedRef = useRef<string>('');
  
  // Auto-check and load preview when channel changes (always show processed/final)
  useEffect(() => {
    if (!processingState.loaded || !processingState.selectedChannel) {
      return;
    }

    console.log(`[ImageProcessing] Channel changed: ${processingState.selectedChannel}`);
    
    // Always check for processed (final) stage preview
    const processedPreviewUrl = processingState.stages.processed?.[processingState.selectedChannel];
    const rawPreviewUrl = processingState.stages.raw?.[processingState.selectedChannel];
    
    console.log(`[ImageProcessing] Preview check for processed/${processingState.selectedChannel}:`, processedPreviewUrl ? 'EXISTS' : 'MISSING');
    
    // If processed preview exists, we're done
    if (processedPreviewUrl) {
      console.log(`[ImageProcessing] Processed preview exists, will render automatically`);
      return;
    }

    // If no processed preview but raw exists, that's fine (no processing done yet)
    if (rawPreviewUrl) {
      console.log(`[ImageProcessing] No processed preview, but raw exists - will show raw`);
      return;
    }

    console.log(`[ImageProcessing] Preview missing for processed/${processingState.selectedChannel}, fetching from backend...`);
    
    const fetchMissingPreview = async () => {
      try {
        // Guard against repeated calls with a ref
        const fetchKey = `${processingState.selectedSample}/${processingState.selectedPosition}/processed`;
        if (fetchedProcessedRef.current === fetchKey) {
          console.log(`[ImageProcessing] Already fetched processed for ${fetchKey}, skipping`);
          return;
        }
        
        // Check if we just reset this channel - don't auto-fetch if so
        const resetKey = `${processingState.selectedSample}/${processingState.selectedPosition}/${processingState.selectedChannel}`;
        if (justResetRef.current === resetKey) {
          console.log(`[ImageProcessing] Just reset channel ${processingState.selectedChannel}, skipping auto-fetch`);
          justResetRef.current = ''; // Clear the flag after checking
          return;
        }
        
        // Fetch processed final from backend
        console.log(`[ImageProcessing] Fetching processed final from backend`);
        fetchedProcessedRef.current = fetchKey;
        await fetchProcessedFinal();
      } catch (err) {
        console.error(`[ImageProcessing] Failed to fetch processed preview:`, err);
      }
    };
    
    if (!processingState.isProcessing) {
      fetchMissingPreview();
    }
  }, [
    processingState.selectedChannel, 
    processingState.loaded, 
    processingState.selectedSample, 
    processingState.selectedPosition, 
    processingState.isProcessing,
    processingState.stages.processed, // Watch for processed stage updates
  ]);

  // Fetch processed final preview and stats
  const fetchProcessedFinal = async () => {
    if (!processingState.selectedSample || !processingState.selectedPosition) {
      return;
    }

    try {
      const response = await fetch(
        `${getApiBase()}/api/input/preprocess/final?sample=${encodeURIComponent(processingState.selectedSample)}&position=${encodeURIComponent(processingState.selectedPosition)}`
      );

      if (!response.ok) {
        // If 404, it means position not loaded or no preprocessing done yet
        if (response.status === 404) {
          console.log(`[ImageProcessing] Processed final not available (position not loaded or no preprocessing done)`);
          // Fallback to raw if processed not available
          setProcessingState(prev => ({
            ...prev,
            stages: {
              ...prev.stages,
              processed: prev.stages.raw,  // Use raw as fallback
            },
          }));
          return;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.success) {
        const finalPreviews = data.data.previews || {};
        const finalStage = data.data.final_stage || 'raw';
        
        console.log(`[ImageProcessing] Fetched processed final previews:`, Object.keys(finalPreviews));
        console.log(`[ImageProcessing] Processed final preview URLs:`, finalPreviews);
        console.log(`[ImageProcessing] Final stage: ${finalStage}`);
        
        // If no previews, fallback to raw
        if (Object.keys(finalPreviews).length === 0) {
          console.log(`[ImageProcessing] No processed previews, using raw as fallback`);
          setProcessingState(prev => ({
            ...prev,
            stages: {
              ...prev.stages,
              processed: prev.stages.raw,
            },
            versionHash: data.data.version_hash || prev.versionHash,
          }));
        } else {
          // Force a new stages object to trigger React re-render
          // Also update channelPreprocessParams to reflect the final stage
          setProcessingState(prev => {
            const newStages = {
              raw: prev.stages.raw,
              processed: { ...finalPreviews },  // Create new object to force update
            };
            
            // Update channelPreprocessParams to set currentStage to final_stage for all channels
            // that have processed previews
            const updatedChannelParams = { ...prev.channelPreprocessParams };
            for (const channelKey of Object.keys(finalPreviews)) {
              if (!updatedChannelParams[channelKey]) {
                updatedChannelParams[channelKey] = {
                  contrastMethod: 'CLAHE',
                  claheClipLimit: 2.0,
                  claheTileSize: 8,
                  stretchPLow: 1,
                  stretchPHigh: 99,
                  currentStage: 'raw',
                };
              }
              // Update currentStage to the final processed stage
              updatedChannelParams[channelKey].currentStage = finalStage as any;
            }
            
            // Update currentPreprocessStage to final_stage
            const newCurrentPreprocessStage = finalStage !== 'raw' ? finalStage as any : prev.currentPreprocessStage;
            
            console.log(`[ImageProcessing] Updated stages object, processed preview for selected channel:`, 
              newStages.processed[prev.selectedChannel]);
            console.log(`[ImageProcessing] Updated channel params currentStage to: ${finalStage}`);
            console.log(`[ImageProcessing] Updated currentPreprocessStage to: ${newCurrentPreprocessStage}`);
            
            return {
              ...prev,
              stages: newStages,
              channelPreprocessParams: updatedChannelParams,
              currentPreprocessStage: newCurrentPreprocessStage,
              versionHash: data.data.version_hash || prev.versionHash,
              backendStats: data.data.stats || prev.backendStats,
            };
          });
        }

        console.log(`[ImageProcessing] Processed final preview updated, histogram should recompute`);
      } else {
        console.warn(`[ImageProcessing] Failed to fetch processed final: ${data.error}`);
        // Fallback to raw
        setProcessingState(prev => ({
          ...prev,
          stages: {
            ...prev.stages,
            processed: prev.stages.raw,
          },
        }));
      }
    } catch (err) {
      console.error(`[ImageProcessing] Error fetching processed final:`, err);
      // On error, fallback to raw
      setProcessingState(prev => ({
        ...prev,
        stages: {
          ...prev.stages,
          processed: prev.stages.raw,
        },
      }));
    }
  };

  // Load preprocessing session and restore state
  const loadPreprocessingSession = async () => {
    if (!processingState.selectedSample || !processingState.selectedPosition) {
      return;
    }

    // Wait a bit to ensure availableItems are loaded
    if (processingState.availableItems.length === 0) {
      console.log(`[ImageProcessing] Waiting for availableItems to load...`);
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // If still no items, skip session restoration
      if (processingState.availableItems.length === 0) {
        console.log(`[ImageProcessing] No availableItems, skipping session restoration`);
        return;
      }
    }

    try {
      const response = await fetch(
        `${getApiBase()}/api/input/preprocess/session?sample=${encodeURIComponent(processingState.selectedSample)}&position=${encodeURIComponent(processingState.selectedPosition)}`
      );

      const data = await response.json();
      
      if (!data.success || !data.data.session) {
        // No session exists - this is fine, just use defaults
        console.log(`[ImageProcessing] No preprocessing session found, using defaults`);
        return;
      }

      const session = data.data.session;
      console.log(`[ImageProcessing] Loaded session with ${session.num_steps} steps:`, session);
      
      // Update version hash
      setProcessingState(prev => ({
        ...prev,
        versionHash: session.version_hash || null,
      }));

      // If no preprocessing steps, just show raw (already loaded)
      if (!session.pipeline || session.pipeline.length === 0) {
        console.log(`[ImageProcessing] No preprocessing steps in session, showing raw images`);
        return;
      }

      // Restore UI controls from session data
      // Extract channel params from pipeline steps
      const restoredChannelParams: { [channelKey: string]: ChannelPreprocessParams } = {};
      const globalParams = session.global_params || {};
      
      // Get current available items (they should be loaded by now)
      const currentItems = processingState.availableItems;
      
      // Initialize all channels with defaults
      currentItems.forEach(item => {
        restoredChannelParams[item.key] = {
          contrastMethod: 'CLAHE',
          claheClipLimit: 2.0,
          claheTileSize: 8,
          stretchPLow: 1,
          stretchPHigh: 99,
          currentStage: 'raw',
        };
      });

      // Extract parameters from pipeline steps
      for (const step of session.pipeline) {
        const stepParams = step.step_params || {};
        const channelParams = step.channel_params || {};
        
        // Update global params if present
        if (step.step_name === 'contrast_enhance') {
          if (stepParams.method === 'CLAHE') {
            setGlobalContrastMethod('CLAHE');
            if (stepParams.clip_limit !== undefined) {
              setGlobalClaheClipLimit(stepParams.clip_limit);
            }
            if (stepParams.tile_grid_size !== undefined) {
              setGlobalClaheTileSize(stepParams.tile_grid_size);
            }
          } else if (stepParams.method === 'Linear Stretch') {
            setGlobalContrastMethod('Linear Stretch');
            if (stepParams.p_low !== undefined) {
              setGlobalStretchPLow(stepParams.p_low);
            }
            if (stepParams.p_high !== undefined) {
              setGlobalStretchPHigh(stepParams.p_high);
            }
          }
        }

        // Update per-channel params
        for (const [channelKey, params] of Object.entries(channelParams)) {
          if (!restoredChannelParams[channelKey]) {
            restoredChannelParams[channelKey] = {
              contrastMethod: 'CLAHE',
              claheClipLimit: 2.0,
              claheTileSize: 8,
              stretchPLow: 1,
              stretchPHigh: 99,
              currentStage: 'raw',
            };
          }

          if (step.step_name === 'contrast_enhance') {
            if (params.method === 'CLAHE') {
              restoredChannelParams[channelKey].contrastMethod = 'CLAHE';
              if (params.clip_limit !== undefined) {
                restoredChannelParams[channelKey].claheClipLimit = params.clip_limit;
              }
              if (params.tile_grid_size !== undefined) {
                restoredChannelParams[channelKey].claheTileSize = params.tile_grid_size;
              }
            } else if (params.method === 'Linear Stretch') {
              restoredChannelParams[channelKey].contrastMethod = 'Linear Stretch';
              if (params.p_low !== undefined) {
                restoredChannelParams[channelKey].stretchPLow = params.p_low;
              }
              if (params.p_high !== undefined) {
                restoredChannelParams[channelKey].stretchPHigh = params.p_high;
              }
            }
          }

          // Update current stage for this channel
          restoredChannelParams[channelKey].currentStage = step.output_stage as any;
        }
      }

      // Update channel params in state
      setProcessingState(prev => ({
        ...prev,
        channelPreprocessParams: restoredChannelParams,
        currentPreprocessStage: session.final_stage || 'raw',
      }));

      // Automatically re-apply the preprocessing pipeline in order
      console.log(`[ImageProcessing] Re-applying ${session.pipeline.length} preprocessing steps...`);
      
      for (let i = 0; i < session.pipeline.length; i++) {
        const step = session.pipeline[i];
        const stepName = step.step_name;
        const stepParams = step.step_params || {};
        const channelParams = step.channel_params || {};
        
        console.log(`[ImageProcessing] Re-applying step ${i + 1}/${session.pipeline.length}: ${stepName}`);
        
        // Build channel params for this step
        const stepChannelParams: { [channelKey: string]: any } = {};
        for (const [channelKey, params] of Object.entries(channelParams)) {
          stepChannelParams[channelKey] = params;
        }
        
        // If no channel-specific params, use global params for all channels
        // Use current state to get available items
        const currentItems = processingState.availableItems;
        if (Object.keys(stepChannelParams).length === 0) {
          for (const item of currentItems) {
            stepChannelParams[item.key] = stepParams;
          }
        }

        // Determine step name for display
        let displayName = stepName;
        if (stepName === 'contrast_enhance') {
          const method = stepParams.method || 'CLAHE';
          displayName = `Contrast Enhancement (${method})`;
        } else if (stepName === 'step1') {
          displayName = 'Background Subtraction';
        } else if (stepName === 'step2') {
          displayName = 'Clip';
        } else if (stepName === 'step3') {
          displayName = 'Gaussian Blur';
        }

        // Re-apply the step
        await handleApplyStep(
          stepName,
          displayName,
          stepParams,
          true, // applyToAll
          stepChannelParams
        );

        // Small delay to ensure state updates
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      console.log(`[ImageProcessing] ✓ All preprocessing steps re-applied`);
      
      // Fetch final processed preview to ensure UI is up to date
      await fetchProcessedFinal();
      
    } catch (err) {
      console.error(`[ImageProcessing] Error loading session:`, err);
      // On error, just continue with raw images
    }
  };

  // Handlers
  const handleLoadPosition = async () => {
    if (!processingState.selectedSample || !processingState.selectedPosition) {
      alert('Please select both sample and position');
      return;
    }

    setProcessingState(prev => ({ ...prev, isLoadingPosition: true }));
    console.log(`Loading position ${processingState.selectedSample}/${processingState.selectedPosition}...`);

    try {
      const response = await fetch('${getApiBase()}/api/input/load_position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: processingState.selectedSample,
          position: processingState.selectedPosition,
        }),
      });

      const data = await response.json();
      if (data.success) {
        const items: ChannelItem[] = data.data.items || [];
        
        const previews: PreviewStage = {};
        items.forEach(item => {
          if (item.preview_url) {
            previews[item.key] = item.preview_url;
          }
        });
        
        const initialChannelParams: { [key: string]: ChannelPreprocessParams } = {};
        items.forEach(item => {
          initialChannelParams[item.key] = {
            contrastMethod: 'CLAHE',
            claheClipLimit: 2.0,
            claheTileSize: 8,
            stretchPLow: 1,
            stretchPHigh: 99,
            currentStage: 'raw',
          };
        });
        
        setProcessingState(prev => ({
          ...prev,
          availableItems: items,
          selectedChannel: items[0]?.key || '',
          loaded: items.length > 0,
          stages: { raw: previews },
          currentStage: 'processed',  // Default to processed (will show raw if no processing done)
          internalStages: {},
          currentPreprocessStage: 'raw',
          channelPreprocessParams: initialChannelParams,
          isLoadingPosition: false,
        }));

        console.log(`✓ Position loaded. Items: ${items.map(i => i.display_label).join(', ')}`);
        
        // Load preprocessing session and fetch processed final
        // This will restore UI controls and try to fetch existing processed images
        await loadPreprocessingSession();
        
        // Also try to fetch processed final immediately (in case files exist but session restore didn't work)
        // This ensures we show processed images if they exist on disk
        try {
          await fetchProcessedFinal();
        } catch (err) {
          console.log(`[ImageProcessing] Could not fetch processed final (may not exist yet):`, err);
        }
      } else {
        console.log(`✗ Failed to load position: ${data.error}`);
        alert(data.error || 'Failed to load position');
        setProcessingState(prev => ({ ...prev, isLoadingPosition: false }));
      }
    } catch (err) {
      console.error('Failed to load position:', err);
      alert('Failed to load position: ' + err);
      setProcessingState(prev => ({ ...prev, isLoadingPosition: false }));
    }
  };

  // Helper: Get channel-specific preprocessing params
  const getChannelParams = (channelKey: string, stepKey: string): any => {
    const channelParams = processingState.channelPreprocessParams[channelKey] || {};
    
    if (stepKey === 'contrast_enhance') {
      const method = channelParams.contrastMethod || processingState.globalContrastMethod;
      if (method === 'CLAHE') {
        return {
          method: 'CLAHE',
          clip_limit: channelParams.claheClipLimit ?? processingState.globalClaheClipLimit,
          tile_grid_size: channelParams.claheTileSize ?? processingState.globalClaheTileSize,
        };
      } else {
        return {
          method: 'Linear Stretch',
          p_low: channelParams.stretchPLow ?? processingState.globalStretchPLow,
          p_high: channelParams.stretchPHigh ?? processingState.globalStretchPHigh,
        };
      }
    }
    
    const stepParamsKey = `${stepKey}Params` as keyof ChannelPreprocessParams;
    return channelParams[stepParamsKey] || {};
  };

  // Helper: Build per-channel params object
  const buildChannelParams = (stepKey: string): { [channelKey: string]: any } => {
    const channelParams: { [channelKey: string]: any } = {};
    
    for (const item of processingState.availableItems) {
      channelParams[item.key] = getChannelParams(item.key, stepKey);
    }
    
    return channelParams;
  };

  const handleApplyStep = async (
    stepKey: string, 
    stepName: string, 
    params?: any,
    applyToAll: boolean = true,
    overrideChannelParams?: { [channelKey: string]: any }
  ) => {
    if (!processingState.loaded) {
      alert('Please load a position first');
      return;
    }

    setProcessingState(prev => ({ ...prev, isProcessing: true }));
    console.log(`Applying ${stepName} (from ${processingState.currentPreprocessStage})...`);

    try {
      let channelParams: { [channelKey: string]: any };
      if (overrideChannelParams) {
        channelParams = overrideChannelParams;
      } else if (applyToAll) {
        channelParams = buildChannelParams(stepKey);
      } else {
        const selectedCh = processingState.selectedChannel;
        channelParams = { [selectedCh]: getChannelParams(selectedCh, stepKey) };
      }
      
      const channelFromStages: { [channelKey: string]: string } = {};
      
      const getChannelFromStage = (channelKey: string): string => {
        const channelState = processingState.channelPreprocessParams[channelKey];
        if (channelState?.currentStage && channelState.currentStage !== 'raw') {
          return channelState.currentStage;
        }
        return 'raw';
      };
      
      if (!applyToAll && overrideChannelParams) {
        const channelKey = Object.keys(overrideChannelParams)[0];
        channelFromStages[channelKey] = getChannelFromStage(channelKey);
      } else if (!applyToAll) {
        channelFromStages[processingState.selectedChannel] = getChannelFromStage(processingState.selectedChannel);
      } else {
        for (const channelKey of Object.keys(channelParams)) {
          channelFromStages[channelKey] = getChannelFromStage(channelKey);
        }
      }
      
      const requestBody: any = {
        sample: processingState.selectedSample,
        position: processingState.selectedPosition,
        from_stage: processingState.currentPreprocessStage,
        channel_from_stages: channelFromStages,
        step: stepKey,
        channel_params: channelParams,
      };
      
      if (params) {
        requestBody.params = params;
      }
      
      const response = await fetch('${getApiBase()}/api/input/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      
      if (data.success) {
        const stepPreviews = data.data.previews || {};
        const stageKey = stepKey as keyof typeof processingState.stages;
        
        console.log(`[ImageProcessing] Processing complete for ${stepName}`);
        console.log(`[ImageProcessing] Stage key: ${stageKey}`);
        console.log(`[ImageProcessing] Previews received:`, Object.keys(stepPreviews));
        console.log(`[ImageProcessing] Preview URLs:`, stepPreviews);
        
        const updatedChannelParams = { ...processingState.channelPreprocessParams };
        if (data.data.channel_states) {
          for (const [channelKey, state] of Object.entries(data.data.channel_states)) {
            if (!updatedChannelParams[channelKey]) {
              updatedChannelParams[channelKey] = {};
            }
            updatedChannelParams[channelKey].currentStage = stepKey as any;
          }
        }
        
        // Store intermediate stage previews internally (for advanced mode)
        const existingInternalPreviews = processingState.internalStages[stageKey] || {};
        const mergedInternalPreviews = { ...existingInternalPreviews, ...stepPreviews };
        
        console.log(`[ImageProcessing] Merged previews for ${stageKey}:`, Object.keys(mergedInternalPreviews));
        console.log(`[ImageProcessing] Selected channel: ${processingState.selectedChannel}`);
        console.log(`[ImageProcessing] Preview for selected channel:`, mergedInternalPreviews[processingState.selectedChannel]);
        
        // Update state with intermediate stage AND the processed stage
        setProcessingState(prev => {
          const newInternalStages = { ...prev.internalStages, [stageKey]: mergedInternalPreviews };
          
          // Also update "processed" stage directly from the step previews
          // This avoids needing to fetch again
          return {
            ...prev,
            internalStages: newInternalStages,
            stages: {
              ...prev.stages,
              processed: mergedInternalPreviews,  // Update processed stage with current step output
            },
            currentPreprocessStage: stepKey as any,
            channelPreprocessParams: updatedChannelParams,
            isProcessing: false,
            versionHash: data.data.version_hash || prev.versionHash,
          };
        });

        // Fetch processed final to update backendStats (stats are not in the step response)
        fetchProcessedFinal();
        console.log(`[ImageProcessing] Updated processed stage with step output; fetching stats from /preprocess/final`);

        console.log(`✓ ${stepName} complete`);
      } else {
        console.log(`✗ ${stepName} failed: ${data.error}`);
        alert(data.error || `${stepName} failed`);
        setProcessingState(prev => ({ ...prev, isProcessing: false }));
      }
    } catch (err) {
      console.error('Preprocessing error:', err);
      alert(`${stepName} failed: ` + err);
      setProcessingState(prev => ({ ...prev, isProcessing: false }));
    }
  };

  const handleApplyContrastEnhancement = () => {
    const params = contrastMethod === 'CLAHE'
      ? { method: 'CLAHE', clip_limit: claheClipLimit, tile_grid_size: claheTileSize }
      : { method: 'Linear Stretch', p_low: stretchPLow, p_high: stretchPHigh };
    
    if (!applyToAllChannels && processingState.selectedChannel) {
      const targetChannel = processingState.selectedChannel;
      
      setProcessingState(prev => ({
        ...prev,
        channelPreprocessParams: {
          ...prev.channelPreprocessParams,
          [targetChannel]: {
            ...prev.channelPreprocessParams[targetChannel],
            contrastMethod: contrastMethod as 'CLAHE' | 'Linear Stretch',
            claheClipLimit: claheClipLimit,
            claheTileSize: claheTileSize,
            stretchPLow: stretchPLow,
            stretchPHigh: stretchPHigh,
          }
        }
      }));
      
      handleApplyStep(
        'contrast_enhance', 
        `Contrast Enhancement (${contrastMethod})`, 
        params, 
        false,
        { [targetChannel]: params }
      );
    } else {
      handleApplyStep('contrast_enhance', `Contrast Enhancement (${contrastMethod})`, params, true);
    }
  };

  const handleApplyBackgroundSubtraction = () => {
    if (!processingState.selectedChannel) {
      alert('Please select a channel first');
      return;
    }
    
    const params = {
      strength: backgroundSubtractionStrength,
    };
    
    handleApplyStep(
      'step1',
      'Background Subtraction',
      params,
      false,
      { [processingState.selectedChannel]: params }
    );
  };

  const handleApplyGaussianBlur = () => {
    if (!processingState.selectedChannel) {
      alert('Please select a channel first');
      return;
    }
    
    const params = {
      sigma: gaussianBlurSigma,
    };
    
    handleApplyStep(
      'step3',
      'Gaussian Blur',
      params,
      false,
      { [processingState.selectedChannel]: params }
    );
  };

  const handleResetToOriginal = () => {
    if (!processingState.loaded || !processingState.selectedChannel) {
      console.warn('[Reset] No channel selected, cannot reset');
      return;
    }

    const channelKey = processingState.selectedChannel;
    console.log(`[Reset] Resetting channel ${channelKey} to original (raw) stage`);

    setProcessingState(prev => {
      const updatedChannelParams = { ...prev.channelPreprocessParams };
      
      if (updatedChannelParams[channelKey]) {
        updatedChannelParams[channelKey] = {
          ...updatedChannelParams[channelKey],
          currentStage: 'raw',
        };
      } else {
        updatedChannelParams[channelKey] = {
          contrastMethod: 'CLAHE',
          claheClipLimit: 2.0,
          claheTileSize: 8,
          stretchPLow: 1,
          stretchPHigh: 99,
          currentStage: 'raw',
        };
      }
      
      // Update stages: clear processed stage for this channel and use raw instead
      const updatedStages = { ...prev.stages };
      const rawPreview = prev.stages.raw?.[channelKey];
      
      // Clear processed stage for this channel - replace with raw preview
      // Create a completely new processed object to force React re-render
      if (rawPreview) {
        // Replace processed preview with raw preview for this channel
        // Create a new object to ensure React detects the change
        const processedData: PreviewStage = {};
        
        // Copy all other channels' processed previews (if any)
        if (prev.stages.processed) {
          for (const [ch, url] of Object.entries(prev.stages.processed)) {
            if (ch !== channelKey) {
              processedData[ch] = url;
            }
          }
        }
        
        // Set this channel to raw preview
        processedData[channelKey] = rawPreview;
        updatedStages.processed = processedData;
        
        console.log(`[Reset] Updated processed stage: channel ${channelKey} now shows raw preview`);
        console.log(`[Reset] Processed stage now contains channels:`, Object.keys(processedData));
      } else if (updatedStages.processed) {
        // Remove this channel from processed if no raw preview available
        const processedData = { ...updatedStages.processed };
        delete processedData[channelKey];
        updatedStages.processed = Object.keys(processedData).length > 0 ? processedData : prev.stages.raw;
      } else {
        // No processed stage exists, use raw as fallback
        updatedStages.processed = prev.stages.raw;
      }
      
      // Clear internal stages for this channel
      const updatedInternalStages = { ...prev.internalStages };
      const stagesToClear = ['contrast_enhance', 'step1', 'step2', 'step3', 'step4'] as const;
      
      for (const stage of stagesToClear) {
        if (updatedInternalStages[stage]) {
          const stageData = { ...updatedInternalStages[stage] };
          delete stageData[channelKey];
          updatedInternalStages[stage] = Object.keys(stageData).length > 0 ? stageData : undefined;
        }
      }

      // Determine if all channels are now raw
      const allChannelsRaw = Object.values(updatedChannelParams).every(
        params => params.currentStage === 'raw'
      );

      // If viewing processed stage and this is the selected channel, switch to raw to show the reset
      const shouldSwitchToRaw = prev.currentStage === 'processed' && prev.selectedChannel === channelKey && rawPreview;

      // Clear the fetchedProcessedRef to allow re-fetching if needed (but won't happen since we set processed to raw)
      fetchedProcessedRef.current = '';
      
      // Set flag to prevent auto-fetch from restoring processed images immediately after reset
      const resetKey = `${prev.selectedSample}/${prev.selectedPosition}/${channelKey}`;
      justResetRef.current = resetKey;
      // Clear the flag after a short delay
      setTimeout(() => {
        if (justResetRef.current === resetKey) {
          justResetRef.current = '';
        }
      }, 1000);

      return {
        ...prev,
        channelPreprocessParams: updatedChannelParams,
        stages: updatedStages,
        internalStages: updatedInternalStages,
        currentPreprocessStage: allChannelsRaw ? 'raw' : prev.currentPreprocessStage,
        currentStage: shouldSwitchToRaw ? 'raw' : prev.currentStage,
        // Clear versionHash to indicate state has changed
        versionHash: allChannelsRaw ? null : prev.versionHash,
      };
    });

    console.log(`[Reset] ✓ Channel ${channelKey} reset to raw`);
    
    // Note: Backend processed files remain on disk, but UI now shows raw
    // User can re-apply preprocessing if needed
  };

  return (
    <div className="pipeline-control">
      {/* 3-COLUMN LAYOUT: Sidebar + Main + Channel Info */}
      <div className="pipeline-layout">
        {/* LEFT: Controls Sidebar */}
        <div className="sidebar-panel">
          {/* File Operations */}
          <div className="sidebar-section">
            <h3>📁 File Operations</h3>
            <div className="sidebar-content">
              <div className="input-group">
                <label>Sample:</label>
                <select 
                  value={processingState.selectedSample} 
                  onChange={(e) => setProcessingState(prev => ({ 
                    ...prev, 
                    selectedSample: e.target.value 
                  }))}
                  disabled={processingState.availableSamples.length === 0}
                >
                  <option value="">-- Select a sample --</option>
                  {processingState.availableSamples.map((sample) => (
                    <option key={sample} value={sample}>
                      {sample}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>Position:</label>
                <select 
                  value={processingState.selectedPosition} 
                  onChange={(e) => setProcessingState(prev => ({ 
                    ...prev, 
                    selectedPosition: e.target.value 
                  }))}
                  disabled={processingState.availablePositions.length === 0}
                >
                  <option value="">-- Select position --</option>
                  {processingState.availablePositions.map((position) => (
                    <option key={position} value={position}>
                      {position}
                    </option>
                  ))}
                </select>
              </div>
              <button 
                className="sidebar-btn" 
                onClick={handleLoadPosition} 
                disabled={!processingState.selectedSample || !processingState.selectedPosition || processingState.isLoadingPosition}
              >
                {processingState.isLoadingPosition ? 'Loading...' : 'Load Position'}
              </button>

              <button className="sidebar-btn btn-image" onClick={() => console.log('TODO: Load Image File')}>
                Load Image File
              </button>
            </div>
          </div>

          {/* Image Processing Pipeline */}
          <div className="sidebar-section">
            <h3>⚙️ Image Processing</h3>
            <div className="sidebar-content">
              {processingState.loaded && (
                <div className="preprocess-stage-indicator">
                  <strong>Current output stage:</strong> {{
                    raw: 'Raw',
                    contrast_enhance: 'Contrast Enhanced',
                    step1: 'Bkg Subtracted',
                    step2: 'Clipped',
                    step3: 'Gaussian Blurred',
                    step4: 'Connected',
                  }[processingState.currentPreprocessStage] ?? processingState.currentPreprocessStage}
                </div>
              )}

              <button className="sidebar-btn btn-reset" onClick={handleResetToOriginal}>
                Reset to Original
              </button>

              <div className="step-box">
                <label>Contrast Enhancement</label>
                <div className="input-group" style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '0.85rem' }}>Method:</label>
                  <select 
                    value={contrastMethod} 
                    onChange={(e) => setContrastMethod(e.target.value)}
                    disabled={!processingState.loaded || processingState.isProcessing}
                  >
                    <option value="CLAHE">CLAHE</option>
                    <option value="Linear Stretch">Linear Stretch</option>
                  </select>
                </div>

                {contrastMethod === 'CLAHE' ? (
                  <>
                    <div style={{ marginTop: '8px' }}>
                      <label style={{ fontSize: '0.85rem' }}>Clip Limit: {claheClipLimit.toFixed(1)}</label>
                      <input 
                        type="range" 
                        min="1.0" 
                        max="6.0" 
                        step="0.1"
                        value={claheClipLimit}
                        onChange={(e) => setClaheClipLimit(Number(e.target.value))}
                        disabled={!processingState.loaded || processingState.isProcessing}
                      />
                    </div>
                    <div style={{ marginTop: '4px' }}>
                      <label style={{ fontSize: '0.85rem' }}>Tile Grid Size: {claheTileSize}</label>
                      <input 
                        type="range" 
                        min="4" 
                        max="16" 
                        step="1"
                        value={claheTileSize}
                        onChange={(e) => setClaheTileSize(Number(e.target.value))}
                        disabled={!processingState.loaded || processingState.isProcessing}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ marginTop: '8px' }}>
                      <label style={{ fontSize: '0.85rem' }}>Low Percentile: {stretchPLow}%</label>
                      <input 
                        type="range" 
                        min="0" 
                        max="10" 
                        step="1"
                        value={stretchPLow}
                        onChange={(e) => setStretchPLow(Number(e.target.value))}
                        disabled={!processingState.loaded || processingState.isProcessing}
                      />
                    </div>
                    <div style={{ marginTop: '4px' }}>
                      <label style={{ fontSize: '0.85rem' }}>High Percentile: {stretchPHigh}%</label>
                      <input 
                        type="range" 
                        min="90" 
                        max="100" 
                        step="1"
                        value={stretchPHigh}
                        onChange={(e) => setStretchPHigh(Number(e.target.value))}
                        disabled={!processingState.loaded || processingState.isProcessing}
                      />
                    </div>
                  </>
                )}
                
                <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    id="applyToAllChannels"
                    checked={applyToAllChannels}
                    onChange={(e) => setApplyToAllChannels(e.target.checked)}
                    disabled={!processingState.loaded || processingState.isProcessing}
                  />
                  <label 
                    htmlFor="applyToAllChannels" 
                    style={{ fontSize: '0.85rem', cursor: 'pointer' }}
                  >
                    Apply to All Channels
                  </label>
                </div>
                
                <button 
                  className="sidebar-btn" 
                  onClick={handleApplyContrastEnhancement}
                  disabled={!processingState.loaded || processingState.isProcessing}
                  style={{ marginTop: '8px' }}
                >
                  {processingState.isProcessing 
                    ? 'Processing...' 
                    : applyToAllChannels 
                      ? 'Apply to All Channels' 
                      : `Apply to ${processingState.availableItems.find(i => i.key === processingState.selectedChannel)?.display_label || 'Selected Channel'}`}
                </button>
              </div>

              <div className="step-box">
                <label>Background Subtraction</label>
                <div className="input-group" style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '0.85rem' }}>Strength:</label>
                  <input
                    type="number"
                    min="0"
                    max="200"
                    step="1"
                    value={backgroundSubtractionStrength}
                    onChange={(e) => setBackgroundSubtractionStrength(Number(e.target.value))}
                    disabled={!processingState.loaded || processingState.isProcessing}
                    style={{ width: '80px', padding: '4px' }}
                  />
                </div>
                <button 
                  className="sidebar-btn" 
                  onClick={handleApplyBackgroundSubtraction}
                  disabled={!processingState.loaded || processingState.isProcessing || !processingState.selectedChannel}
                  style={{ marginTop: '8px' }}
                >
                  {processingState.isProcessing ? 'Processing...' : `Apply to ${processingState.availableItems.find(i => i.key === processingState.selectedChannel)?.display_label || 'Channel'}`}
                </button>
              </div>

              <div className="step-box">
                <label>Gaussian Blur</label>
                <div className="input-group" style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '0.85rem' }}>Sigma:</label>
                  <input
                    type="number"
                    min="0.1"
                    max="10.0"
                    step="0.1"
                    value={gaussianBlurSigma}
                    onChange={(e) => setGaussianBlurSigma(Number(e.target.value))}
                    disabled={!processingState.loaded || processingState.isProcessing}
                    style={{ width: '80px', padding: '4px' }}
                  />
                </div>
                <button 
                  className="sidebar-btn" 
                  onClick={handleApplyGaussianBlur}
                  disabled={!processingState.loaded || processingState.isProcessing || !processingState.selectedChannel}
                  style={{ marginTop: '8px' }}
                >
                  {processingState.isProcessing ? 'Processing...' : `Apply to ${processingState.availableItems.find(i => i.key === processingState.selectedChannel)?.display_label || 'Channel'}`}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* CENTER: Main Panel */}
        <div className="main-panel">
          {/* Status Display */}
          {status && status.status !== 'idle' && (
            <div className={`status-panel status-${status.status}`}>
              <h3>Status: {status.status.toUpperCase()}</h3>
              
              {status.current_step && (
                <p>
                  <strong>Current Step:</strong> {status.current_step}
                </p>
              )}
              
              {status.current_sample && (
                <p>
                  <strong>Processing:</strong> {status.current_sample}
                </p>
              )}

              <p>
                <strong>Progress:</strong> {status.completed_samples} / {status.total_samples} samples
              </p>

              <ProgressBar progress={status.progress} />

              {status.error && (
                <div className="error-panel">
                  <h4>❌ Error:</h4>
                  <p>{status.error}</p>
                  {status.error_details && (
                    <details>
                      <summary>Show Details</summary>
                      <pre>{status.error_details}</pre>
                    </details>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Preview Section */}
          <div className="preview-section">
            <div className="preview-header">
              <h3>📸 Image Preview</h3>
              {processingState.loaded && (
                <div className="preview-info">
                  <span><strong>Sample:</strong> {processingState.selectedSample}</span>
                  <span><strong>Position:</strong> {processingState.selectedPosition}</span>
                  <span><strong>Stage:</strong> Processed (Final)</span>
                  <span><strong>Channel:</strong> {
                    processingState.availableItems.find(item => item.key === processingState.selectedChannel)?.display_label || processingState.selectedChannel
                  }</span>
                </div>
              )}
            </div>

            <div className="preview-controls">
              <div className="control-group">
                <label>Channel:</label>
                <select 
                  value={processingState.selectedChannel}
                  onChange={(e) => setProcessingState(prev => ({ 
                    ...prev, 
                    selectedChannel: e.target.value 
                  }))}
                  disabled={!processingState.loaded}
                >
                  {processingState.availableItems.map(item => (
                    <option key={item.key} value={item.key}>{item.display_label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="preview-box">
              {processingState.loaded ? (
                (() => {
                  // Always show processed (final) stage, fallback to raw if no processing done
                  let previewUrl: string | undefined;
                  
                  // Try processed first
                  previewUrl = processingState.stages.processed?.[processingState.selectedChannel];
                  
                  // If no processed preview exists yet, fallback to raw
                  if (!previewUrl) {
                    previewUrl = processingState.stages.raw?.[processingState.selectedChannel];
                  }
                  
                  if (previewUrl) {
                    const absoluteUrl = previewUrl.startsWith('http') ? previewUrl : `${getApiBase()}${previewUrl}`;
                    
                    // Use stable URL without timestamp to prevent flickering
                    const finalUrl = absoluteUrl;
                    
                    // Use a stable key based on the actual image URL
                    const stableKey = `processed-${processingState.selectedChannel}-${previewUrl}`;
                    
                    return (
                      <img 
                        key={stableKey}
                        src={finalUrl}
                        alt={`Processed - ${processingState.selectedChannel}`}
                        style={{ maxWidth: '100%', height: 'auto' }}
                        onError={(e) => {
                          console.error(`[ImageProcessing] Failed to load preview image:`, e);
                          console.error(`[ImageProcessing] URL:`, previewUrl);
                        }}
                        onLoad={() => {
                          console.log(`[ImageProcessing] Preview image loaded for processed/${processingState.selectedChannel}`);
                        }}
                      />
                    );
                  } else {
                    return (
                      <p>No preview available for {processingState.selectedChannel}</p>
                    );
                  }
                })()
              ) : (
                <p>Load a sample and position to preview images</p>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Channel Info Panel */}
        <ChannelInfoPanel
          stages={processingState.stages}
          currentStage="processed"
          availableItems={processingState.availableItems}
          isLoaded={processingState.loaded}
          selectedSample={processingState.selectedSample}
          selectedPosition={processingState.selectedPosition}
          backendStats={processingState.backendStats}
        />
      </div>
    </div>
  );
};

export default ImageProcessing;

