/**
 * Main Pipeline Control Component
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import ProgressBar from './ProgressBar';
import AlignmentColorOverlay from './AlignmentColorOverlay';
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

interface PipelineState {
  // Selection state
  selectedSample: string;
  selectedPosition: string;
  selectedChannel: string;  // Internal key: e.g., "C1_ch1"
  
  // Available options
  availableSamples: string[];
  availablePositions: string[];
  availableItems: ChannelItem[];  // Replaces availableChannels
  
  // Preview state
  loaded: boolean;
  stages: {
    raw?: PreviewStage;
    contrast_enhance?: PreviewStage;
    step1?: PreviewStage;
    step2?: PreviewStage;
    step3?: PreviewStage;
    step4?: PreviewStage;
    aligned?: PreviewStage;
  };
  currentStage: 'raw' | 'contrast_enhance' | 'step1' | 'step2' | 'step3' | 'step4' | 'aligned';
  
  // Preprocessing tracking (NEW - critical for pipeline flow)
  currentPreprocessStage: 'raw' | 'contrast_enhance' | 'step1' | 'step2' | 'step3' | 'step4';
  
  // NEW: Per-channel preprocessing state
  channelPreprocessParams: {
    [channelKey: string]: ChannelPreprocessParams;
  };
  
  // Global preprocessing params (for backward compatibility and "Apply to All")
  globalContrastMethod: 'CLAHE' | 'Linear Stretch';
  globalClaheClipLimit: number;
  globalClaheTileSize: number;
  globalStretchPLow: number;
  globalStretchPHigh: number;
  
  // Busy flags
  isLoadingPosition: boolean;
  isAligning: boolean;
  isProcessing: boolean;
}

const PipelineControl: React.FC = () => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  
  // Ref to track if we're currently auto-triggering preprocessing (to prevent loops)
  const autoTriggeringRef = useRef<string>('');

  // Unified pipeline state
  const [pipelineState, setPipelineState] = useState<PipelineState>({
    selectedSample: '',
    selectedPosition: '',
    selectedChannel: '',
    availableSamples: [],
    availablePositions: [],
    availableItems: [],
    loaded: false,
    stages: {},
    currentStage: 'raw',
    currentPreprocessStage: 'raw',
    channelPreprocessParams: {},  // NEW: per-channel params
    globalContrastMethod: 'CLAHE',  // Global defaults
    globalClaheClipLimit: 2.0,
    globalClaheTileSize: 8,
    globalStretchPLow: 1,
    globalStretchPHigh: 99,
    isLoadingPosition: false,
    isAligning: false,
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
      // Progress update (status will be updated separately)
      fetchStatus();
    } else if (message.type === 'log') {
      // Log messages no longer displayed in UI
      console.log('[Pipeline Log]', message.message || '');
    }
  }, [fetchStatus]);

  // WebSocket connection for real-time updates
  const { isConnected } = useWebSocket({
    url: `${getWsBase()}/ws/pipeline`,
    reconnect: false, // Disable auto-reconnect to prevent spam
    onMessage: handleWebSocketMessage,
    onError: (error) => {
      console.error('WebSocket error in PipelineControl:', error);
      // Don't crash the component on WebSocket errors
    },
  });

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000); // Poll every 2s as backup
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Alignment/preprocessing settings
  const [refChannel, setRefChannel] = useState<string>('ch1');
  const [alignMethod, setAlignMethod] = useState<string>('phase_cross_correlation');
  const [transformType, setTransformType] = useState<string>('EuclideanTransform');
  
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
  const [applyToAllChannels, setApplyToAllChannels] = useState<boolean>(false);  // Default: apply to selected only

  // Fetch input samples on mount
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await fetch('${getApiBase()}/api/input/samples');
        const data = await response.json();
        if (data.success) {
          setPipelineState(prev => ({ ...prev, availableSamples: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch input samples:', err);
      }
    };
    fetchSamples();
  }, []);

  // Fetch positions when sample changes
  useEffect(() => {
    if (!pipelineState.selectedSample) {
      setPipelineState(prev => ({ 
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
          `${getApiBase()}/api/input/samples/${pipelineState.selectedSample}/positions`
        );
        const data = await response.json();
        if (data.success) {
          setPipelineState(prev => ({ ...prev, availablePositions: data.data }));
        }
      } catch (err) {
        console.error('Failed to fetch positions:', err);
      }
    };
    fetchPositions();
  }, [pipelineState.selectedSample]);

  // Auto-check and load preview when channel or stage changes
  useEffect(() => {
    if (!pipelineState.loaded || !pipelineState.selectedChannel || !pipelineState.currentStage) {
      return;
    }

    console.log(`[Frontend] Channel/Stage changed: ${pipelineState.selectedChannel} / ${pipelineState.currentStage}`);
    
    const currentStageData = pipelineState.stages[pipelineState.currentStage];
    const previewUrl = currentStageData?.[pipelineState.selectedChannel];
    
    console.log(`[Frontend] Preview check for ${pipelineState.currentStage}/${pipelineState.selectedChannel}:`, previewUrl ? 'EXISTS' : 'MISSING');
    
    // If preview exists, nothing to do - it will render automatically
    if (previewUrl) {
      console.log(`[Frontend] Preview exists, will render automatically`);
      return;
    }

    // Preview doesn't exist for current stage - try to fetch from backend
    console.log(`[Frontend] Preview missing for ${pipelineState.currentStage}/${pipelineState.selectedChannel}`);
    console.log(`[Frontend] Attempting to fetch preview from backend...`);
    
    const fetchMissingPreview = async () => {
      try {
        // Check channel's current preprocessing state
        const channelState = pipelineState.channelPreprocessParams[pipelineState.selectedChannel];
        const channelCurrentStage = channelState?.currentStage || 'raw';
        
        console.log(`[Frontend] Channel ${pipelineState.selectedChannel} current stage: ${channelCurrentStage}`);
        console.log(`[Frontend] Requested stage: ${pipelineState.currentStage}`);
        
        // If channel hasn't been processed to the requested stage, auto-trigger preprocessing
        if (channelCurrentStage !== pipelineState.currentStage && pipelineState.currentStage !== 'raw') {
          const triggerKey = `${pipelineState.selectedChannel}/${pipelineState.currentStage}`;
          
          // Prevent duplicate triggers
          if (autoTriggeringRef.current === triggerKey) {
            console.log(`[Frontend] Already auto-triggering ${triggerKey}, skipping`);
            return;
          }
          
          console.log(`[Frontend] Auto-triggering preprocessing for ${pipelineState.selectedChannel} to reach ${pipelineState.currentStage}`);
          autoTriggeringRef.current = triggerKey;
          
          // Determine which step to apply based on currentStage
          let stepKey: string = pipelineState.currentStage;
          let stepName: string = '';
          
          if (pipelineState.currentStage === 'contrast_enhance') {
            stepKey = 'contrast_enhance';
            
            // Use channel-specific params if available, otherwise use global UI values
            const channelParams = channelState || {};
            const method = channelParams.contrastMethod || contrastMethod;
            stepName = `Contrast Enhancement (${method})`;
            
            const params = method === 'CLAHE'
              ? { 
                  method: 'CLAHE', 
                  clip_limit: channelParams.claheClipLimit ?? claheClipLimit, 
                  tile_grid_size: channelParams.claheTileSize ?? claheTileSize 
                }
              : { 
                  method: 'Linear Stretch', 
                  p_low: channelParams.stretchPLow ?? stretchPLow, 
                  p_high: channelParams.stretchPHigh ?? stretchPHigh 
                };
            
            console.log(`[Frontend] Auto-applying contrast enhancement with params:`, params);
            
            // Apply to this channel only
            handleApplyStep(
              stepKey,
              stepName,
              params,
              false, // applyToAll = false
              { [pipelineState.selectedChannel]: params } // overrideChannelParams
            ).then(() => {
              // Clear trigger ref after processing completes
              if (autoTriggeringRef.current === triggerKey) {
                autoTriggeringRef.current = '';
              }
            }).catch(() => {
              // Clear trigger ref on error too
              if (autoTriggeringRef.current === triggerKey) {
                autoTriggeringRef.current = '';
              }
            });
          } else {
            // For other steps, use default params
            stepName = `Step ${stepKey}`;
            console.log(`[Frontend] Auto-triggering ${stepKey} for ${pipelineState.selectedChannel}`);
            handleApplyStep(stepKey, stepName, undefined, false).then(() => {
              if (autoTriggeringRef.current === triggerKey) {
                autoTriggeringRef.current = '';
              }
            }).catch(() => {
              if (autoTriggeringRef.current === triggerKey) {
                autoTriggeringRef.current = '';
              }
            });
          }
        } else if (channelCurrentStage === pipelineState.currentStage) {
          // Channel was processed to this stage, but preview is missing
          // This shouldn't happen, but if it does, we need to fetch the preview
          console.warn(`[Frontend] Channel ${pipelineState.selectedChannel} was processed to ${pipelineState.currentStage}, but preview is missing`);
          console.warn(`[Frontend] This suggests a state synchronization issue`);
        } else {
          // Channel is at a later stage than requested, or we're viewing raw
          console.log(`[Frontend] Channel is at ${channelCurrentStage}, requested ${pipelineState.currentStage}`);
          if (pipelineState.currentStage === 'raw') {
            // Raw preview should always exist
            const rawPreviewUrl = pipelineState.stages.raw?.[pipelineState.selectedChannel];
            if (!rawPreviewUrl) {
              console.warn(`[Frontend] WARNING: Raw preview missing for ${pipelineState.selectedChannel}`);
            }
          }
        }
      } catch (err) {
        console.error(`[Frontend] Failed to fetch/trigger missing preview:`, err);
      }
    };
    
    // Only fetch if we're not currently processing (to avoid race conditions)
    if (!pipelineState.isProcessing) {
      fetchMissingPreview();
    }
  }, [
    pipelineState.selectedChannel, 
    pipelineState.currentStage, 
    pipelineState.loaded, 
    pipelineState.stages, 
    pipelineState.selectedSample, 
    pipelineState.selectedPosition, 
    pipelineState.isProcessing,
    pipelineState.channelPreprocessParams,
    contrastMethod,
    claheClipLimit,
    claheTileSize,
    stretchPLow,
    stretchPHigh
  ]);

  // Handlers
  const handleLoadPosition = async () => {
    if (!pipelineState.selectedSample || !pipelineState.selectedPosition) {
      alert('Please select both sample and position');
      return;
    }

    setPipelineState(prev => ({ ...prev, isLoadingPosition: true }));
    console.log(`Loading position ${pipelineState.selectedSample}/${pipelineState.selectedPosition}...`);

    try {
      const response = await fetch('${getApiBase()}/api/input/load_position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: pipelineState.selectedSample,
          position: pipelineState.selectedPosition,
        }),
      });

      const data = await response.json();
      if (data.success) {
        const items: ChannelItem[] = data.data.items || [];
        
        // Build previews object from items: {key: preview_url}
        const previews: PreviewStage = {};
        items.forEach(item => {
          if (item.preview_url) {
            previews[item.key] = item.preview_url;
          }
        });
        
        // Initialize per-channel preprocessing params
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
        
        setPipelineState(prev => ({
          ...prev,
          availableItems: items,
          selectedChannel: items[0]?.key || '',
          loaded: items.length > 0,
          stages: { raw: previews },
          currentStage: 'raw',
          currentPreprocessStage: 'raw',
          channelPreprocessParams: initialChannelParams,  // NEW: initialize per-channel params
          isLoadingPosition: false,
        }));

        // Set first item as default reference
        if (items.length > 0) {
          setRefChannel(items[0].key);
        }

        console.log(`✓ Position loaded. Items: ${items.map(i => i.display_label).join(', ')}`);
      } else {
        console.log(`✗ Failed to load position: ${data.error}`);
        alert(data.error || 'Failed to load position');
        setPipelineState(prev => ({ ...prev, isLoadingPosition: false }));
      }
    } catch (err) {
      console.error('Failed to load position:', err);
      console.log(`✗ Error: ${err}`);
      alert('Failed to load position: ' + err);
      setPipelineState(prev => ({ ...prev, isLoadingPosition: false }));
    }
  };

  const handleRunAlignment = async () => {
    if (!pipelineState.loaded) {
      alert('Please load a position first');
      return;
    }

    setPipelineState(prev => ({ ...prev, isAligning: true }));
    const inputStage = pipelineState.currentPreprocessStage;
    console.log(`Running alignment on stage "${inputStage}" with reference: ${refChannel}...`);

    try {
      const response = await fetch('${getApiBase()}/api/input/align', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample: pipelineState.selectedSample,
          position: pipelineState.selectedPosition,
          input_stage: inputStage,
          ref_channel: refChannel,
          method: alignMethod,
          transform: transformType,
        }),
      });

      const data = await response.json();
      if (data.success) {
        const alignedPreviews = data.data.previews || {};
        
        console.log('[Alignment] Received previews from backend:', alignedPreviews);
        console.log('[Alignment] Preview keys:', Object.keys(alignedPreviews));
        
        setPipelineState(prev => ({
          ...prev,
          stages: { ...prev.stages, aligned: alignedPreviews },
          currentStage: 'aligned',
          isAligning: false,
        }));

        console.log(`✓ Alignment complete. Input: ${inputStage}, Method: ${alignMethod}`);
      } else {
        console.log(`✗ Alignment failed: ${data.error}`);
        alert(data.error || 'Alignment failed');
        setPipelineState(prev => ({ ...prev, isAligning: false }));
      }
    } catch (err) {
      console.error('Alignment error:', err);
      console.log(`✗ Error: ${err}`);
      alert('Alignment failed: ' + err);
      setPipelineState(prev => ({ ...prev, isAligning: false }));
    }
  };

  // Helper: Get channel-specific preprocessing params
  const getChannelParams = (channelKey: string, stepKey: string): any => {
    const channelParams = pipelineState.channelPreprocessParams[channelKey] || {};
    
    if (stepKey === 'contrast_enhance') {
      const method = channelParams.contrastMethod || pipelineState.globalContrastMethod;
      if (method === 'CLAHE') {
        return {
          method: 'CLAHE',
          clip_limit: channelParams.claheClipLimit ?? pipelineState.globalClaheClipLimit,
          tile_grid_size: channelParams.claheTileSize ?? pipelineState.globalClaheTileSize,
        };
      } else {
        return {
          method: 'Linear Stretch',
          p_low: channelParams.stretchPLow ?? pipelineState.globalStretchPLow,
          p_high: channelParams.stretchPHigh ?? pipelineState.globalStretchPHigh,
        };
      }
    }
    
    // For other steps, return channel-specific params if available
    const stepParamsKey = `${stepKey}Params` as keyof ChannelPreprocessParams;
    return channelParams[stepParamsKey] || {};
  };

  // Helper: Build per-channel params object
  const buildChannelParams = (stepKey: string): { [channelKey: string]: any } => {
    const channelParams: { [channelKey: string]: any } = {};
    
    for (const item of pipelineState.availableItems) {
      channelParams[item.key] = getChannelParams(item.key, stepKey);
    }
    
    return channelParams;
  };

  const handleApplyStep = async (
    stepKey: string, 
    stepName: string, 
    params?: any,
    applyToAll: boolean = true,  // NEW: if false, apply only to selected channel
    overrideChannelParams?: { [channelKey: string]: any }  // NEW: override params for specific channels
  ) => {
    if (!pipelineState.loaded) {
      alert('Please load a position first');
      return;
    }

    setPipelineState(prev => ({ ...prev, isProcessing: true }));
    console.log(`Applying ${stepName} (from ${pipelineState.currentPreprocessStage})...`);

    try {
      // Build per-channel parameters
      let channelParams: { [channelKey: string]: any };
      if (overrideChannelParams) {
        // Use provided override params
        channelParams = overrideChannelParams;
        console.log(`[Frontend] Using override params:`, overrideChannelParams);
      } else if (applyToAll) {
        // Apply to all channels with their specific params
        channelParams = buildChannelParams(stepKey);
      } else {
        // Apply to selected only
        const selectedCh = pipelineState.selectedChannel;
        console.log(`[Frontend] Applying to selected channel: ${selectedCh}`);
        channelParams = { [selectedCh]: getChannelParams(selectedCh, stepKey) };
      }
      
      console.log(`[Frontend] Applying ${stepName}`);
      console.log(`[Frontend] Apply to all: ${applyToAll}`);
      console.log(`[Frontend] Selected channel: ${pipelineState.selectedChannel}`);
      console.log(`[Frontend] Channels to process:`, Object.keys(channelParams));
      console.log(`[Frontend] Channel params:`, channelParams);
      
      // Determine per-channel from_stage based on each channel's current preprocessing state
      // If a channel hasn't been processed yet, use 'raw' as its from_stage
      const channelFromStages: { [channelKey: string]: string } = {};
      
      const getChannelFromStage = (channelKey: string): string => {
        const channelState = pipelineState.channelPreprocessParams[channelKey];
        // If channel has a processed stage (not undefined and not 'raw'), use it
        // Otherwise, use 'raw' (channel hasn't been processed yet)
        if (channelState?.currentStage && channelState.currentStage !== 'raw') {
          return channelState.currentStage;
        }
        return 'raw';
      };
      
      if (!applyToAll && overrideChannelParams) {
        // Single channel: use its current stage, or fallback to 'raw'
        const channelKey = Object.keys(overrideChannelParams)[0];
        const fromStage = getChannelFromStage(channelKey);
        channelFromStages[channelKey] = fromStage;
        console.log(`[Frontend] Channel ${channelKey} from_stage: ${fromStage} (currentStage: ${pipelineState.channelPreprocessParams[channelKey]?.currentStage || 'raw'})`);
      } else if (!applyToAll) {
        // Single channel without override: use its current stage
        const fromStage = getChannelFromStage(pipelineState.selectedChannel);
        channelFromStages[pipelineState.selectedChannel] = fromStage;
        console.log(`[Frontend] Channel ${pipelineState.selectedChannel} from_stage: ${fromStage} (currentStage: ${pipelineState.channelPreprocessParams[pipelineState.selectedChannel]?.currentStage || 'raw'})`);
      } else {
        // All channels: determine each channel's current stage
        for (const channelKey of Object.keys(channelParams)) {
          channelFromStages[channelKey] = getChannelFromStage(channelKey);
        }
        console.log(`[Frontend] Per-channel from_stages:`, channelFromStages);
      }
      
      const requestBody: any = {
        sample: pipelineState.selectedSample,
        position: pipelineState.selectedPosition,
        from_stage: pipelineState.currentPreprocessStage,  // Global fallback
        channel_from_stages: channelFromStages,  // NEW: per-channel from_stage
        step: stepKey,
        channel_params: channelParams,  // NEW: per-channel params
      };
      
      // Legacy support: also include global params for backward compatibility
      if (params) {
        requestBody.params = params;
      }
      
      console.log(`[Frontend] Request body:`, JSON.stringify(requestBody, null, 2));

      const response = await fetch('${getApiBase()}/api/input/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      console.log(`[Frontend] STEP 7: Response received`);
      console.log(`[Frontend]   Response success: ${data.success}`);
      console.log(`[Frontend]   Full response data:`, JSON.stringify(data, null, 2));
      
      if (data.success) {
        const stepPreviews = data.data.previews || {};
        const stageKey = stepKey as keyof typeof pipelineState.stages;
        
        console.log(`[Frontend] STEP 8: Parse response`);
        console.log(`[Frontend]   Step key: ${stageKey}`);
        console.log(`[Frontend]   Previews object:`, stepPreviews);
        console.log(`[Frontend]   Processed channels (preview keys):`, Object.keys(stepPreviews));
        console.log(`[Frontend]   Preview URLs:`, stepPreviews);
        console.log(`[Frontend]   Channel states returned:`, data.data.channel_states || {});
        console.log(`[Frontend]   Channel states keys:`, Object.keys(data.data.channel_states || {}));
        
        if (Object.keys(stepPreviews).length === 0) {
          console.error(`[Frontend] ERROR: No previews in response!`);
          console.error(`[Frontend]   This means the backend processed 0 channels`);
          console.error(`[Frontend]   Check backend logs for why channels were skipped`);
        }
        
        // Update per-channel preprocessing states
        const updatedChannelParams = { ...pipelineState.channelPreprocessParams };
        if (data.data.channel_states) {
          for (const [channelKey, state] of Object.entries(data.data.channel_states)) {
            if (!updatedChannelParams[channelKey]) {
              updatedChannelParams[channelKey] = {};
            }
            updatedChannelParams[channelKey].currentStage = stepKey as any;
          }
        }
        
        // Only update previews for channels that were actually processed
        // Merge with existing previews to preserve other channels
        const existingPreviews = pipelineState.stages[stageKey] || {};
        const mergedPreviews = { ...existingPreviews, ...stepPreviews };
        
        console.log(`[Frontend] STEP 9: Merge previews`);
        console.log(`[Frontend]   Stage key: ${stageKey}`);
        console.log(`[Frontend]   Existing previews for ${stageKey}:`, existingPreviews);
        console.log(`[Frontend]   Existing preview keys:`, Object.keys(existingPreviews));
        console.log(`[Frontend]   New previews to merge:`, stepPreviews);
        console.log(`[Frontend]   New preview keys:`, Object.keys(stepPreviews));
        console.log(`[Frontend]   Merged previews:`, mergedPreviews);
        console.log(`[Frontend]   Merged preview keys:`, Object.keys(mergedPreviews));
        
        setPipelineState(prev => {
          const newStages = { ...prev.stages, [stageKey]: mergedPreviews };
          console.log(`[Frontend] STEP 10: Update state`);
          console.log(`[Frontend]   New stages object:`, newStages);
          console.log(`[Frontend]   New stages keys:`, Object.keys(newStages));
          console.log(`[Frontend]   Stage ${stageKey} previews:`, newStages[stageKey]);
          console.log(`[Frontend]   Stage ${stageKey} preview keys:`, Object.keys(newStages[stageKey] || {}));
          console.log(`[Frontend]   Selected channel: ${prev.selectedChannel}`);
          console.log(`[Frontend]   Preview for ${stageKey}/${prev.selectedChannel}:`, newStages[stageKey]?.[prev.selectedChannel]);
          
          return {
            ...prev,
            stages: newStages,
            currentStage: stageKey,
            currentPreprocessStage: stepKey as any,
            channelPreprocessParams: updatedChannelParams,
            isProcessing: false,
          };
        });

        console.log(`✓ ${stepName} complete`);
        console.log(`[Frontend] Updated previews for stage ${stageKey}:`, Object.keys(mergedPreviews));
      } else {
        console.log(`✗ ${stepName} failed: ${data.error}`);
        alert(data.error || `${stepName} failed`);
        setPipelineState(prev => ({ ...prev, isProcessing: false }));
      }
    } catch (err) {
      console.error('Preprocessing error:', err);
      console.log(`✗ Error: ${err}`);
      alert(`${stepName} failed: ` + err);
      setPipelineState(prev => ({ ...prev, isProcessing: false }));
    }
  };

  const handleApplyContrastEnhancement = () => {
    const params = contrastMethod === 'CLAHE'
      ? { method: 'CLAHE', clip_limit: claheClipLimit, tile_grid_size: claheTileSize }
      : { method: 'Linear Stretch', p_low: stretchPLow, p_high: stretchPHigh };
    
    console.log(`[Frontend] handleApplyContrastEnhancement called`);
    console.log(`[Frontend] applyToAllChannels: ${applyToAllChannels}`);
    console.log(`[Frontend] selectedChannel: ${pipelineState.selectedChannel}`);
    console.log(`[Frontend] selectedChannel display: ${pipelineState.availableItems.find(i => i.key === pipelineState.selectedChannel)?.display_label}`);
    
    // If applying to selected channel only, use current UI values directly
    if (!applyToAllChannels && pipelineState.selectedChannel) {
      const targetChannel = pipelineState.selectedChannel;  // Capture current value
      console.log(`[Frontend] Applying to single channel: ${targetChannel}`);
      
      // Update the channel's params in state for future reference
      setPipelineState(prev => ({
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
      
      // Pass current UI values directly as override params
      handleApplyStep(
        'contrast_enhance', 
        `Contrast Enhancement (${contrastMethod})`, 
        params, 
        false,
        { [targetChannel]: params }  // Use captured value, not pipelineState.selectedChannel
      );
    } else {
      // Apply to all channels using their stored params
      console.log(`[Frontend] Applying to all channels`);
      handleApplyStep('contrast_enhance', `Contrast Enhancement (${contrastMethod})`, params, true);
    }
  };

  // Background Subtraction handler (per-channel)
  const handleApplyBackgroundSubtraction = () => {
    if (!pipelineState.selectedChannel) {
      alert('Please select a channel first');
      return;
    }
    
    const params = {
      strength: backgroundSubtractionStrength,
    };
    
    // Apply only to selected channel
    handleApplyStep(
      'step1',
      'Background Subtraction',
      params,
      false, // applyToAll = false
      { [pipelineState.selectedChannel]: params }
    );
  };

  // Gaussian Blur handler (per-channel)
  const handleApplyGaussianBlur = () => {
    if (!pipelineState.selectedChannel) {
      alert('Please select a channel first');
      return;
    }
    
    const params = {
      sigma: gaussianBlurSigma,
    };
    
    // Apply only to selected channel
    handleApplyStep(
      'step3',
      'Gaussian Blur',
      params,
      false, // applyToAll = false
      { [pipelineState.selectedChannel]: params }
    );
  };

  const handleResetToOriginal = () => {
    if (!pipelineState.loaded || !pipelineState.selectedChannel) {
      console.warn('[Reset] No channel selected, cannot reset');
      return;
    }

    const channelKey = pipelineState.selectedChannel;
    console.log(`[Reset] Resetting channel ${channelKey} to original (raw) stage`);

    // Reset only the selected channel's preprocessing state
    setPipelineState(prev => {
      const updatedChannelParams = { ...prev.channelPreprocessParams };
      
      // Reset this channel's preprocessing state to raw
      if (updatedChannelParams[channelKey]) {
        updatedChannelParams[channelKey] = {
          ...updatedChannelParams[channelKey],
          currentStage: 'raw',
        };
      } else {
        // Initialize if doesn't exist
        updatedChannelParams[channelKey] = {
          contrastMethod: 'CLAHE',
          claheClipLimit: 2.0,
          claheTileSize: 8,
          stretchPLow: 1,
          stretchPHigh: 99,
          currentStage: 'raw',
        };
      }

      // Check BEFORE deletion if the reset channel has a preview for the current stage
      // This determines if we should switch to raw stage after reset
      const hadPreviewForCurrentStage = prev.currentStage !== 'raw' && 
        prev.stages[prev.currentStage]?.[channelKey] !== undefined;
      
      // Remove this channel's previews from all processed stages (keep raw)
      // CRITICAL: Never touch the raw stage - it should always remain intact for all channels
      // IMPORTANT: Only remove previews for the reset channel, keep other channels' previews intact
      const updatedStages = { ...prev.stages };
      const stagesToClear = ['contrast_enhance', 'step1', 'step2', 'step3', 'step4', 'aligned'] as const;
      
      console.log(`[Reset] ========== RESET CHANNEL ${channelKey} ==========`);
      console.log(`[Reset] BEFORE: Stages contain:`, Object.keys(updatedStages).map(s => ({
        stage: s,
        channels: Object.keys(updatedStages[s] || {})
      })));
      console.log(`[Reset] Removing previews for channel ${channelKey} from processed stages ONLY`);
      console.log(`[Reset] RAW stage will NOT be touched`);
      
      for (const stage of stagesToClear) {
        if (updatedStages[stage]) {
          const stageData = { ...updatedStages[stage] };
          const hadChannel = channelKey in stageData;
          const otherChannels = Object.keys(stageData).filter(k => k !== channelKey);
          
          if (hadChannel) {
            console.log(`[Reset]   Stage ${stage}: Removing ${channelKey}, keeping ${otherChannels.length} other channel(s): ${otherChannels.join(', ')}`);
            delete stageData[channelKey];
            // Keep the stage object if other channels still have previews, otherwise set to undefined
            updatedStages[stage] = Object.keys(stageData).length > 0 ? stageData : undefined;
          } else {
            console.log(`[Reset]   Stage ${stage}: ${channelKey} not present, no change needed`);
          }
        }
      }
      
      // CRITICAL: Ensure raw stage is preserved exactly as-is
      if (updatedStages.raw) {
        console.log(`[Reset] RAW stage preserved with ${Object.keys(updatedStages.raw).length} channel(s): ${Object.keys(updatedStages.raw).join(', ')}`);
      }
      
      console.log(`[Reset] After removal - stages:`, Object.keys(updatedStages).map(s => ({
        stage: s,
        channels: Object.keys(updatedStages[s] || {})
      })));

      // CRITICAL FIX: Do NOT switch currentStage to 'raw' globally
      // This would cause ALL channels to show raw, not just the reset channel
      // Instead, keep currentStage as-is. If the reset channel is selected and viewing a processed stage,
      // the preview will show "No preview available" (correct behavior). User can manually switch to 'raw' if needed.
      // Other channels will continue to show their processed previews correctly.
      let newCurrentStage = prev.currentStage;
      
      console.log(`[Reset] Keeping current stage ${prev.currentStage} (NOT switching to raw)`);
      console.log(`[Reset]   Reset channel: ${channelKey}`);
      console.log(`[Reset]   Selected channel: ${prev.selectedChannel}`);
      console.log(`[Reset]   Reset channel had preview for current stage: ${hadPreviewForCurrentStage}`);
      console.log(`[Reset]   If reset channel is selected, it will show "No preview available" for ${prev.currentStage}`);
      console.log(`[Reset]   User can manually switch to 'raw' stage to see the reset channel's raw image`);
      console.log(`[Reset]   Other channels will continue showing their ${prev.currentStage} previews`);

      // Log final state for debugging
      console.log(`[Reset] ========== FINAL STATE ==========`);
      console.log(`[Reset]   currentStage: ${newCurrentStage} (was ${prev.currentStage})`);
      console.log(`[Reset]   selectedChannel: ${channelKey}`);
      console.log(`[Reset]   Reset channel's currentStage: ${updatedChannelParams[channelKey]?.currentStage}`);
      console.log(`[Reset]   Stages after reset:`, Object.keys(updatedStages).map(s => ({
        stage: s,
        channelCount: Object.keys(updatedStages[s] || {}).length,
        channels: Object.keys(updatedStages[s] || {})
      })));
      
      // VERIFICATION: Ensure raw stage still has all channels
      if (prev.stages.raw) {
        const rawChannelsBefore = Object.keys(prev.stages.raw);
        const rawChannelsAfter = Object.keys(updatedStages.raw || {});
        if (rawChannelsBefore.length !== rawChannelsAfter.length) {
          console.error(`[Reset] ERROR: Raw stage channel count changed! Before: ${rawChannelsBefore.length}, After: ${rawChannelsAfter.length}`);
          console.error(`[Reset]   Before: ${rawChannelsBefore.join(', ')}`);
          console.error(`[Reset]   After: ${rawChannelsAfter.join(', ')}`);
          // Restore raw stage to prevent corruption
          updatedStages.raw = prev.stages.raw;
          console.error(`[Reset]   Restored raw stage to original state`);
        } else {
          console.log(`[Reset] ✓ Raw stage preserved: ${rawChannelsAfter.length} channel(s)`);
        }
      }

      return {
        ...prev,
        channelPreprocessParams: updatedChannelParams,
        stages: updatedStages,
        // Only change currentStage if reset channel is selected and had preview for current stage
        currentStage: newCurrentStage,
        // Update global preprocess stage only if all channels are at raw
        currentPreprocessStage: Object.values(updatedChannelParams).every(
          params => params.currentStage === 'raw'
        ) ? 'raw' : prev.currentPreprocessStage,
      };
    });

    console.log(`[Reset] ✓ Channel ${channelKey} reset to raw`);
  };

  const handleRunAllSteps = async () => {
    if (!pipelineState.loaded) {
      alert('Please load a position first');
      return;
    }

    console.log('Running all preprocessing steps + alignment...');
    
    // Run preprocessing steps sequentially
    try {
      // Reset to raw first
      handleResetToOriginal();
      
      // Apply contrast enhancement
      await handleApplyContrastEnhancement();
      
      // Wait a bit for state to update (in production, use proper Promise chaining)
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Apply Background Subtraction and Gaussian Blur (if needed)
      // Note: These are now per-channel, so "Run All Steps" applies to all channels
      // For now, skip these in "Run All Steps" - user can apply them per-channel manually
      // await handleApplyBackgroundSubtraction();
      // await new Promise(resolve => setTimeout(resolve, 500));
      
      // await handleApplyGaussianBlur();
      // await new Promise(resolve => setTimeout(resolve, 500));
      
      // Finally, run alignment on the preprocessed result
      await handleRunAlignment();
      
      console.log('✓ All steps complete!');
    } catch (err) {
      console.error('Run all steps failed:', err);
      alert('Run all steps failed: ' + err);
    }
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
                  value={pipelineState.selectedSample} 
                  onChange={(e) => setPipelineState(prev => ({ 
                    ...prev, 
                    selectedSample: e.target.value 
                  }))}
                  disabled={pipelineState.availableSamples.length === 0}
                >
                  <option value="">-- Select a sample --</option>
                  {pipelineState.availableSamples.map((sample) => (
                    <option key={sample} value={sample}>
                      {sample}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>Position:</label>
                <select 
                  value={pipelineState.selectedPosition} 
                  onChange={(e) => setPipelineState(prev => ({ 
                    ...prev, 
                    selectedPosition: e.target.value 
                  }))}
                  disabled={pipelineState.availablePositions.length === 0}
                >
                  <option value="">-- Select position --</option>
                  {pipelineState.availablePositions.map((position) => (
                    <option key={position} value={position}>
                      {position}
                    </option>
                  ))}
                </select>
              </div>
              <button 
                className="sidebar-btn" 
                onClick={handleLoadPosition} 
                disabled={!pipelineState.selectedSample || !pipelineState.selectedPosition || pipelineState.isLoadingPosition}
              >
                {pipelineState.isLoadingPosition ? 'Loading...' : 'Load Position'}
              </button>

              <button className="sidebar-btn btn-image" onClick={() => console.log('TODO: Load Image File')}>
                Load Image File
              </button>
            </div>
          </div>

          {/* Pre-processing Pipeline */}
          <div className="sidebar-section">
            <h3>⚙️ Pre-processing Pipeline</h3>
            <div className="sidebar-content">
              {pipelineState.loaded && (
                <div className="preprocess-stage-indicator">
                  <strong>Current output stage:</strong> {pipelineState.currentPreprocessStage}
                </div>
              )}

              <button className="sidebar-btn btn-reset" onClick={handleResetToOriginal}>
                Reset to Original
              </button>

              <button 
                className="sidebar-btn" 
                onClick={handleRunAllSteps}
                disabled={!pipelineState.loaded || pipelineState.isProcessing}
                style={{ background: '#9b59b6', marginBottom: '8px' }}
              >
                {pipelineState.isProcessing ? 'Processing...' : '▶️ Run All Steps + Align'}
              </button>

              <div className="step-box">
                <label>Contrast Enhancement</label>
                <div className="input-group" style={{ marginTop: '8px' }}>
                  <label style={{ fontSize: '0.85rem' }}>Method:</label>
                  <select 
                    value={contrastMethod} 
                    onChange={(e) => setContrastMethod(e.target.value)}
                    disabled={!pipelineState.loaded || pipelineState.isProcessing}
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
                        disabled={!pipelineState.loaded || pipelineState.isProcessing}
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
                        disabled={!pipelineState.loaded || pipelineState.isProcessing}
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
                        disabled={!pipelineState.loaded || pipelineState.isProcessing}
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
                        disabled={!pipelineState.loaded || pipelineState.isProcessing}
                      />
                    </div>
                  </>
                )}
                
                {/* Apply to All vs Selected Channel Toggle */}
                <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    id="applyToAllChannels"
                    checked={applyToAllChannels}
                    onChange={(e) => setApplyToAllChannels(e.target.checked)}
                    disabled={!pipelineState.loaded || pipelineState.isProcessing}
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
                  disabled={!pipelineState.loaded || pipelineState.isProcessing}
                  style={{ marginTop: '8px' }}
                >
                  {pipelineState.isProcessing 
                    ? 'Processing...' 
                    : applyToAllChannels 
                      ? 'Apply to All Channels' 
                      : `Apply to ${pipelineState.availableItems.find(i => i.key === pipelineState.selectedChannel)?.display_label || 'Selected Channel'}`}
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
                    disabled={!pipelineState.loaded || pipelineState.isProcessing}
                    style={{ width: '80px', padding: '4px' }}
                  />
                </div>
                <button 
                  className="sidebar-btn" 
                  onClick={handleApplyBackgroundSubtraction}
                  disabled={!pipelineState.loaded || pipelineState.isProcessing || !pipelineState.selectedChannel}
                  style={{ marginTop: '8px' }}
                >
                  {pipelineState.isProcessing ? 'Processing...' : `Apply to ${pipelineState.availableItems.find(i => i.key === pipelineState.selectedChannel)?.display_label || 'Channel'}`}
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
                    disabled={!pipelineState.loaded || pipelineState.isProcessing}
                    style={{ width: '80px', padding: '4px' }}
                  />
                </div>
                <button 
                  className="sidebar-btn" 
                  onClick={handleApplyGaussianBlur}
                  disabled={!pipelineState.loaded || pipelineState.isProcessing || !pipelineState.selectedChannel}
                  style={{ marginTop: '8px' }}
                >
                  {pipelineState.isProcessing ? 'Processing...' : `Apply to ${pipelineState.availableItems.find(i => i.key === pipelineState.selectedChannel)?.display_label || 'Channel'}`}
                </button>
              </div>
            </div>
          </div>

          {/* Alignment */}
          <div className="sidebar-section">
            <h3>🎯 Alignment</h3>
            <div className="sidebar-content">
              {pipelineState.loaded && (
                <div className="input-stage-indicator">
                  <strong>Input stage:</strong> {pipelineState.currentPreprocessStage}
                </div>
              )}

              <div className="input-group">
                <label>Reference Channel:</label>
                {pipelineState.availableItems.length > 0 ? (
                  <select 
                    value={refChannel} 
                    onChange={(e) => setRefChannel(e.target.value)}
                  >
                    {pipelineState.availableItems.map((item) => (
                      <option key={item.key} value={item.key}>
                        {item.display_label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <select disabled>
                    <option value="">Load a position to detect channels</option>
                  </select>
                )}
              </div>

              <div className="input-group">
                <label>Method:</label>
                <select 
                  value={alignMethod} 
                  onChange={(e) => setAlignMethod(e.target.value)}
                  disabled={!pipelineState.loaded}
                >
                  <option value="phase_cross_correlation">Phase Cross Correlation</option>
                  <option value="feature_based">Feature Based</option>
                </select>
              </div>

              <div className="input-group">
                <label>Transform:</label>
                <select 
                  value={transformType} 
                  onChange={(e) => setTransformType(e.target.value)}
                  disabled={!pipelineState.loaded}
                >
                  <option value="EuclideanTransform">Euclidean Transform</option>
                  <option value="AffineTransform">Affine Transform</option>
                </select>
              </div>

              <button 
                className="sidebar-btn" 
                onClick={handleRunAlignment} 
                disabled={!pipelineState.loaded || pipelineState.isAligning}
              >
                {pipelineState.isAligning ? 'Aligning...' : 'Align Using Preprocessed'}
              </button>
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
              {pipelineState.loaded && (
                <div className="preview-info">
                  <span><strong>Sample:</strong> {pipelineState.selectedSample}</span>
                  <span><strong>Position:</strong> {pipelineState.selectedPosition}</span>
                  <span><strong>Stage:</strong> {pipelineState.currentStage}</span>
                  <span><strong>Channel:</strong> {
                    pipelineState.availableItems.find(item => item.key === pipelineState.selectedChannel)?.display_label || pipelineState.selectedChannel
                  }</span>
                </div>
              )}
            </div>

            <div className="preview-controls">
              <div className="control-group">
                <label>Stage:</label>
                <select 
                  value={pipelineState.currentStage}
                  onChange={(e) => setPipelineState(prev => ({ 
                    ...prev, 
                    currentStage: e.target.value as any 
                  }))}
                  disabled={!pipelineState.loaded}
                >
                  {pipelineState.stages.raw && <option value="raw">Raw</option>}
                  {pipelineState.stages.contrast_enhance && <option value="contrast_enhance">Contrast Enhanced</option>}
                  {pipelineState.stages.step1 && <option value="step1">Background Subtracted</option>}
                  {pipelineState.stages.step3 && <option value="step3">Gaussian Blurred</option>}
                  {pipelineState.stages.aligned && <option value="aligned">Aligned</option>}
                </select>
              </div>

              <div className="control-group">
                <label>Channel:</label>
                <select 
                  value={pipelineState.selectedChannel}
                  onChange={(e) => setPipelineState(prev => ({ 
                    ...prev, 
                    selectedChannel: e.target.value 
                  }))}
                  disabled={!pipelineState.loaded}
                >
                  {pipelineState.availableItems.map(item => (
                    <option key={item.key} value={item.key}>{item.display_label}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="preview-box">
              {pipelineState.loaded ? (
                pipelineState.stages[pipelineState.currentStage]?.[pipelineState.selectedChannel] ? (
                  <img 
                    src={(() => {
                      const url = pipelineState.stages[pipelineState.currentStage]![pipelineState.selectedChannel];
                      const absoluteUrl = url.startsWith('http') ? url : `${getApiBase()}${url}`;
                      const separator = absoluteUrl.includes('?') ? '&' : '?';
                      return `${absoluteUrl}${separator}t=${Date.now()}`;
                    })()}
                    alt={`${pipelineState.currentStage} - ${pipelineState.selectedChannel}`}
                    style={{ maxWidth: '100%', height: 'auto' }}
                  />
                ) : (
                  <p>No preview available for {pipelineState.currentStage} / {pipelineState.selectedChannel}</p>
                )
              ) : (
                <p>Load a sample and position to preview images</p>
              )}
            </div>
          </div>

          {/* Alignment Color Overlay QC */}
          <AlignmentColorOverlay
            stages={pipelineState.stages}
            availableItems={pipelineState.availableItems}
            isLoaded={pipelineState.loaded}
          />
        </div>

        {/* RIGHT: Channel Info Panel */}
        <ChannelInfoPanel
          stages={pipelineState.stages}
          currentStage={pipelineState.currentStage}
          availableItems={pipelineState.availableItems}
          isLoaded={pipelineState.loaded}
          selectedSample={pipelineState.selectedSample}
          selectedPosition={pipelineState.selectedPosition}
        />
      </div>
    </div>
  );
};

export default PipelineControl;

