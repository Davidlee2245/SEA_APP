# Per-Channel Preprocessing Design

## Overview

The preprocessing pipeline has been redesigned to support **per-channel preprocessing parameters** instead of applying uniform settings to all channels. This allows each channel (biomarker) to be optimized independently based on its unique signal characteristics.

## Architecture Changes

### Backend Changes

#### 1. API Endpoint: `/api/input/preprocess` (Updated)

**New Request Format:**
```json
{
  "sample": "A2780Cis10",
  "position": "P1",
  "from_stage": "raw",
  "step": "contrast_enhance",
  "channel_params": {
    "C1_ch1": {
      "method": "CLAHE",
      "clip_limit": 2.0,
      "tile_grid_size": 8
    },
    "C1_ch2": {
      "method": "Linear Stretch",
      "p_low": 1,
      "p_high": 99
    },
    "C1_ch3": {
      "method": "CLAHE",
      "clip_limit": 3.5,
      "tile_grid_size": 12
    }
  },
  "params": { ... }  // Legacy: fallback for channels not in channel_params
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "previews": { ... },
    "channel_states": {
      "C1_ch1": {
        "step": "contrast_enhance",
        "from_stage": "raw",
        "params": { ... },
        "output_path": "..."
      },
      ...
    }
  }
}
```

#### 2. New API Endpoint: `/api/input/preprocess/state`

Get per-channel preprocessing state for a position.

**Query Params:**
- `sample`: Sample name
- `position`: Position name

**Response:**
```json
{
  "success": true,
  "data": {
    "channel_states": {
      "C1_ch1": {
        "contrast_enhance": { ... },
        "step1": { ... },
        ...
      },
      ...
    }
  }
}
```

#### 3. Alignment Endpoint: `/api/input/align` (Updated)

**New Request Format:**
```json
{
  "sample": "A2780Cis10",
  "position": "P1",
  "input_stage": "contrast_enhance",
  "channel_stages": {  // NEW: per-channel stages
    "C1_ch1": "contrast_enhance",
    "C1_ch2": "raw",
    "C1_ch3": "contrast_enhance"
  },
  "ref_channel": "C1_ch1",
  ...
}
```

**Behavior:**
- If `channel_stages` is provided, each channel uses its specified preprocessed stage
- If not provided, all channels use `input_stage` (backward compatible)
- Alignment estimates transform on preprocessed images (better features)
- Alignment applies transform to raw images (preserves sharpness)

### Frontend Changes

#### 1. State Structure (Updated)

```typescript
interface ChannelPreprocessParams {
  contrastMethod?: 'CLAHE' | 'Linear Stretch';
  claheClipLimit?: number;
  claheTileSize?: number;
  stretchPLow?: number;
  stretchPHigh?: number;
  currentStage?: 'raw' | 'contrast_enhance' | ...;
  // ... other step params
}

interface PipelineState {
  // ... existing fields ...
  channelPreprocessParams: {
    [channelKey: string]: ChannelPreprocessParams;
  };
  globalContrastMethod: 'CLAHE' | 'Linear Stretch';  // For "Apply to All"
  globalClaheClipLimit: number;
  // ... global defaults
}
```

#### 2. Handler Functions (Updated)

- `handleApplyStep()`: Now accepts `applyToAll` parameter
  - If `true`: Applies preprocessing to all channels with their specific params
  - If `false`: Applies only to selected channel
- `getChannelParams()`: Helper to get channel-specific params
- `buildChannelParams()`: Helper to build per-channel params object

#### 3. UI Components (To Be Added)

- **Per-Channel Preprocessing Panel**: Expandable section showing controls for each channel
- **Channel Selector**: Dropdown to select channel for individual tuning
- **Apply to All / Apply to Selected**: Toggle buttons
- **Per-Channel Preview**: Preview preprocessing results per channel

## Benefits of Per-Channel Preprocessing

### 1. **Improved Signal Quality**

**Problem with Global Preprocessing:**
- Channels with different SNR require different denoising strengths
- High-SNR channels may be over-processed, losing detail
- Low-SNR channels may be under-processed, retaining noise

**Solution:**
- Each channel can have optimized denoising parameters
- High-SNR channels: minimal denoising, preserve detail
- Low-SNR channels: stronger denoising, reduce noise

### 2. **Better Contrast Enhancement**

**Problem with Global Preprocessing:**
- Different biomarkers have different intensity distributions
- Global CLAHE settings may over-enhance some channels, under-enhance others
- Grid visibility varies by channel (some channels show grid clearly, others don't)

**Solution:**
- Per-channel CLAHE parameters:
  - High-contrast channels: Lower clip_limit (2.0), preserve natural contrast
  - Low-contrast channels: Higher clip_limit (4.0-6.0), enhance visibility
  - Grid-visible channels: Larger tile size (12x12), preserve grid structure
  - Grid-invisible channels: Smaller tile size (8x8), enhance local contrast

### 3. **Optimized Background Removal**

**Problem with Global Preprocessing:**
- Different channels have different background distributions
- Some channels have uniform background, others have gradient
- Global background removal radius may remove signal or leave background

**Solution:**
- Per-channel background removal:
  - Uniform background channels: Small radius (5-10px)
  - Gradient background channels: Larger radius (15-20px)
  - High-background channels: Stronger subtraction
  - Low-background channels: Minimal subtraction

### 4. **Improved Alignment Accuracy**

**Problem with Global Preprocessing:**
- Alignment features may be better detected in some preprocessed channels than others
- Some channels may need different preprocessing for optimal feature detection

**Solution:**
- Channel-specific preprocessed images for alignment:
  - Use best-preprocessed version of each channel for feature detection
  - Reference channel can use its optimal preprocessing
  - Other channels can use their optimal preprocessing
  - Transform is estimated on preprocessed (better features)
  - Transform is applied to raw (preserves sharpness)

### 5. **Better Downstream Analysis**

**Problem with Global Preprocessing:**
- Analysis steps (segmentation, quantification) may require different preprocessing per channel
- Some biomarkers need strong contrast enhancement, others need subtle enhancement

**Solution:**
- Each channel is preprocessed optimally for its analysis requirements
- Segmentation can use channel-specific thresholds
- Quantification uses channel-specific normalization

## Usage Workflow

### Step 1: Load Position
- Load position → All channels initialized with default params

### Step 2: Configure Per-Channel Parameters
- Select a channel from dropdown
- Adjust preprocessing parameters for that channel:
  - Contrast method (CLAHE / Linear Stretch)
  - CLAHE clip limit and tile size
  - Linear stretch percentiles
- Preview results for that channel

### Step 3: Apply Preprocessing
- **Option A: Apply to All Channels**
  - Uses each channel's specific parameters
  - Processes all channels in parallel
  
- **Option B: Apply to Selected Channel**
  - Applies only to currently selected channel
  - Allows iterative tuning per channel

### Step 4: Alignment
- Alignment automatically uses:
  - Channel-specific preprocessed images for feature detection
  - Raw images for final warping (preserves sharpness)
- Reference channel can be selected from any preprocessed stage

## Example Use Cases

### Use Case 1: High-SNR Channel with Grid
- **Channel**: C1_ch1 (p62) - High signal, clear grid
- **Settings**:
  - Method: CLAHE
  - Clip Limit: 2.0 (preserve natural contrast)
  - Tile Size: 12x12 (preserve grid structure)
- **Result**: Grid remains sharp, signal enhanced without over-processing

### Use Case 2: Low-SNR Channel
- **Channel**: C1_ch2 (CD63) - Low signal, noisy
- **Settings**:
  - Method: CLAHE
  - Clip Limit: 4.0 (stronger enhancement)
  - Tile Size: 8x8 (enhance local contrast)
- **Result**: Noise reduced, signal enhanced, better visibility

### Use Case 3: High-Background Channel
- **Channel**: C2_ch1 (CD44) - High background, gradient
- **Settings**:
  - Method: Linear Stretch
  - Percentiles: P1-P99 (remove outliers)
  - Background Removal: Large radius (15px)
- **Result**: Background removed, signal enhanced

## Implementation Status

✅ **Completed:**
- Backend API supports per-channel parameters
- Preprocessing cache tracks per-channel state
- Alignment supports channel-specific preprocessed images
- Frontend state structure updated
- Handler functions updated

🔄 **In Progress:**
- Frontend UI for per-channel controls
- Per-channel preview/tuning

📋 **Planned:**
- Per-channel preprocessing panel UI
- Channel selector dropdown
- "Apply to All" vs "Apply to Selected" toggle
- Per-channel preview visualization

## Migration Guide

### For Existing Code:
- Legacy `params` field still supported (applied to all channels)
- If `channel_params` not provided, falls back to `params`
- Existing workflows continue to work

### For New Code:
- Use `channel_params` for per-channel preprocessing
- Use `channel_stages` in alignment for channel-specific stages
- Query `/api/input/preprocess/state` to get current per-channel state

## Technical Notes

### Performance Considerations:
- Per-channel preprocessing processes channels in parallel (backend)
- No significant performance overhead vs global preprocessing
- Cache structure efficiently tracks per-channel state

### Memory Considerations:
- Per-channel state stored in memory (preprocessing_cache)
- No significant memory increase (only metadata, not images)
- Images still stored once per stage

### Backward Compatibility:
- All existing API calls continue to work
- Global `params` field still supported
- Frontend can gradually migrate to per-channel UI

