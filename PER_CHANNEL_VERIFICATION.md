# Per-Channel Preprocessing Verification Report

## Executive Summary

✅ **Mostly Independent**: The preprocessing pipeline correctly applies per-channel parameters and stores results independently. However, **one critical limitation** was identified that prevents true per-channel independence when channels are at different preprocessing stages.

## ✅ Verified: Independent Per-Channel Processing

### 1. **Per-Channel Parameters Are Maintained** ✅

**Backend Implementation:**
```python
# Line 1407: Each channel gets its own parameters
channel_specific_params = channel_params.get(channel_name, global_params)

# Line 1420-1421: Parameters are applied per channel
method = channel_specific_params.get('method', global_params.get('method', 'CLAHE'))
processed = apply_contrast_enhancement(img_array, method, channel_specific_params)
```

**Verification:**
- ✅ Each channel retrieves its own parameters from `channel_params[channel_name]`
- ✅ If channel-specific params not provided, falls back to `global_params` (intentional)
- ✅ Parameters are passed directly to processing function per channel

**Frontend Implementation:**
```typescript
// Line 356-378: getChannelParams() retrieves channel-specific params
const getChannelParams = (channelKey: string, stepKey: string): any => {
  const channelParams = pipelineState.channelPreprocessParams[channelKey] || {};
  // Returns channel-specific params, falls back to global if not set
}
```

**Verification:**
- ✅ Each channel has its own entry in `channelPreprocessParams[channelKey]`
- ✅ Parameters are stored per channel in state
- ✅ Changing one channel's params doesn't affect others

### 2. **Changing Settings Affects Only That Channel** ✅

**State Management:**
```typescript
// Line 83-85: Per-channel state structure
channelPreprocessParams: {
  [channelKey: string]: ChannelPreprocessParams;
}
```

**Verification:**
- ✅ Each channel has independent state object
- ✅ Modifying `channelPreprocessParams['C1_ch1']` doesn't affect `channelPreprocessParams['C1_ch2']`
- ✅ State is preserved when switching between channels in UI

### 3. **Preprocessing Results Stored Per Channel** ✅

**Backend Storage:**
```python
# Line 1438-1440: Each channel gets its own output file
output_path = output_dir / f"{channel_name}.tif"
tifffile.imwrite(str(output_path), processed)
output_files[channel_name] = output_path

# Line 1443-1448: Per-channel state stored separately
channel_states[channel_name] = {
    'step': step,
    'from_stage': from_stage,
    'params': channel_specific_params,  # Channel-specific params stored
    'output_path': str(output_path)
}

# Line 1459-1462: Cache stores per-channel state per step
for channel_name, state in channel_states.items():
    if channel_name not in preprocessing_cache[position_key]['channel_states']:
        preprocessing_cache[position_key]['channel_states'][channel_name] = {}
    preprocessing_cache[position_key]['channel_states'][channel_name][step] = state
```

**Verification:**
- ✅ Each channel has unique output file: `{channel_name}.tif`
- ✅ Each channel's state is stored in `channel_states[channel_name][step]`
- ✅ No file sharing or overwriting between channels
- ✅ Cache structure: `preprocessing_cache[position_key]['channel_states'][channel_name][step]`

### 4. **State Preserved When Switching Channels** ✅

**Frontend State:**
```typescript
// Line 280: State initialized per channel on load
channelPreprocessParams: initialChannelParams,  // Each channel gets its own params

// Line 437-445: State updated per channel after preprocessing
const updatedChannelParams = { ...pipelineState.channelPreprocessParams };
if (data.data.channel_states) {
  for (const [channelKey, state] of Object.entries(data.data.channel_states)) {
    if (!updatedChannelParams[channelKey]) {
      updatedChannelParams[channelKey] = {};
    }
    updatedChannelParams[channelKey].currentStage = stepKey as any;
  }
}
```

**Verification:**
- ✅ State is initialized per channel when position is loaded
- ✅ State updates are per-channel (doesn't overwrite other channels)
- ✅ Switching channels in UI preserves each channel's state

## ⚠️ CRITICAL ISSUE FOUND: Global `from_stage`

### Problem: All Channels Must Use Same Input Stage

**Location:** `api_server_extended.py` lines 1376-1395

**Current Implementation:**
```python
from_stage = data.get('from_stage', 'raw')  # Global for all channels

if from_stage == 'raw':
    input_files = preprocessing_cache[position_key]['raw']
else:
    input_files = preprocessing_cache[position_key][from_stage]
```

**Impact:**
- ❌ **Cannot process channels at different stages in one call**
- ❌ If C1_ch1 is at "raw" and C1_ch2 is at "contrast_enhance", you cannot apply step1 to both in one request
- ❌ All channels must be at the same preprocessing stage to process together

**Example Scenario:**
1. Apply contrast_enhance to C1_ch1 → C1_ch1 now at "contrast_enhance"
2. C1_ch2 remains at "raw"
3. Try to apply step1 to both:
   - If `from_stage="raw"`: C1_ch1 will load from wrong stage (raw instead of contrast_enhance)
   - If `from_stage="contrast_enhance"`: C1_ch2 will fail (no contrast_enhance stage)

### Solution: Support Per-Channel `from_stage`

**Recommended Fix:**
```python
# Accept per-channel from_stage
channel_from_stages = data.get('channel_from_stages', {})  # NEW
global_from_stage = data.get('from_stage', 'raw')  # Fallback

# Determine input files per channel
input_files = {}
for channel_name in all_channels:
    # Get channel-specific from_stage
    channel_from_stage = channel_from_stages.get(channel_name, global_from_stage)
    
    # Load from appropriate stage
    if channel_from_stage == 'raw':
        input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
    else:
        # Get from channel_states to find the correct file
        channel_states = preprocessing_cache[position_key].get('channel_states', {})
        if channel_name in channel_states and channel_from_stage in channel_states[channel_name]:
            state = channel_states[channel_name][channel_from_stage]
            input_files[channel_name] = Path(state['output_path'])
        else:
            # Fallback to global stage cache
            if channel_from_stage in preprocessing_cache[position_key]:
                input_files[channel_name] = preprocessing_cache[position_key][channel_from_stage][channel_name]
```

## ✅ No Issues Found: Other Areas

### No Shared Buffers or Global Variables
- ✅ `img_array` is loaded fresh per channel (line 1410)
- ✅ `processed` is computed independently per channel (line 1421)
- ✅ `output_files` dictionary stores separate paths per channel
- ✅ No global image buffers that could leak between channels

### No Cross-Channel File Reuse
- ✅ Each channel writes to unique file: `{channel_name}.tif`
- ✅ File paths stored separately in `output_files[channel_name]`
- ✅ Cache stores per-channel paths: `channel_states[channel_name][step]['output_path']`

### Parameter Isolation
- ✅ `channel_specific_params` retrieved per channel (line 1407)
- ✅ Parameters passed directly to processing function (line 1421)
- ✅ No parameter mutation or sharing between channels

## 🔍 Debugging Steps to Verify Independence

### Step 1: Verify Per-Channel Parameters in Request

**Add logging to backend:**
```python
# In preprocess_position(), after line 1407
print(f"[Preprocess] Channel {channel_name} params: {channel_specific_params}")
print(f"[Preprocess] Channel {channel_name} method: {channel_specific_params.get('method')}")
print(f"[Preprocess] Channel {channel_name} clip_limit: {channel_specific_params.get('clip_limit')}")
```

**Expected Output:**
```
[Preprocess] Channel C1_ch1 params: {'method': 'CLAHE', 'clip_limit': 2.0, 'tile_grid_size': 8}
[Preprocess] Channel C1_ch2 params: {'method': 'Linear Stretch', 'p_low': 1, 'p_high': 99}
```

### Step 2: Verify Per-Channel File Outputs

**Add logging:**
```python
# After line 1439
print(f"[Preprocess] Channel {channel_name} output: {output_path}")
print(f"[Preprocess] Channel {channel_name} file exists: {output_path.exists()}")
```

**Check file system:**
```bash
ls -la data/processing/{sample}/{position}/contrast_enhance/
# Should see: C1_ch1.tif, C1_ch2.tif, C1_ch3.tif (separate files)
```

### Step 3: Verify Per-Channel State Storage

**Query state endpoint:**
```bash
curl "http://localhost:5000/api/input/preprocess/state?sample=A2780Cis10&position=P1"
```

**Expected Response:**
```json
{
  "channel_states": {
    "C1_ch1": {
      "contrast_enhance": {
        "params": {"method": "CLAHE", "clip_limit": 2.0},
        "output_path": ".../C1_ch1.tif"
      }
    },
    "C1_ch2": {
      "contrast_enhance": {
        "params": {"method": "Linear Stretch", "p_low": 1, "p_high": 99},
        "output_path": ".../C1_ch2.tif"
      }
    }
  }
}
```

### Step 4: Verify Frontend State Isolation

**Add console logging:**
```typescript
// In getChannelParams()
console.log(`[Frontend] Channel ${channelKey} params:`, channelParams);
console.log(`[Frontend] Channel ${channelKey} method:`, channelParams.contrastMethod);

// In buildChannelParams()
console.log(`[Frontend] Built channel params:`, channelParams);
```

**Expected Output:**
```
[Frontend] Channel C1_ch1 params: {contrastMethod: 'CLAHE', claheClipLimit: 2.0, ...}
[Frontend] Channel C1_ch2 params: {contrastMethod: 'Linear Stretch', stretchPLow: 1, ...}
[Frontend] Built channel params: {C1_ch1: {...}, C1_ch2: {...}}
```

### Step 5: Test Parameter Independence

**Test Case:**
1. Load position with 3 channels
2. Set C1_ch1: CLAHE, clip_limit=2.0
3. Set C1_ch2: CLAHE, clip_limit=4.0
4. Set C1_ch3: Linear Stretch, p_low=5, p_high=95
5. Apply contrast_enhance to all
6. Verify each channel used its own parameters

**Verification:**
- Check backend logs for per-channel params
- Check output files have different characteristics
- Query state endpoint to verify stored params match

### Step 6: Test State Preservation

**Test Case:**
1. Apply contrast_enhance to C1_ch1 only
2. Switch selected channel to C1_ch2
3. Apply contrast_enhance to C1_ch2 with different params
4. Switch back to C1_ch1
5. Verify C1_ch1 still has its original params

**Verification:**
- Check `pipelineState.channelPreprocessParams['C1_ch1']` unchanged
- Check `pipelineState.channelPreprocessParams['C1_ch2']` has new params

## 📋 Summary of Findings

| Aspect | Status | Notes |
|--------|--------|-------|
| **Per-channel parameters** | ✅ Independent | Each channel maintains its own params |
| **Parameter application** | ✅ Independent | Params applied per channel in loop |
| **File storage** | ✅ Independent | Each channel writes to unique file |
| **State storage** | ✅ Independent | Per-channel state in cache structure |
| **State preservation** | ✅ Independent | State maintained when switching channels |
| **Input stage selection** | ⚠️ **GLOBAL** | **All channels must use same `from_stage`** |
| **Shared buffers** | ✅ None | No global image buffers |
| **File reuse** | ✅ None | Each channel has unique output file |

## 🎯 Recommendations

### Immediate Fix Required

1. **Implement per-channel `from_stage` support** (see Solution above)
   - Allows channels at different stages to be processed together
   - Enables true per-channel preprocessing independence

### Optional Enhancements

2. **Add validation** to ensure channel-specific params are provided
3. **Add UI indicator** showing current preprocessing stage per channel
4. **Add preview comparison** to visualize per-channel preprocessing differences

## ✅ Conclusion

The preprocessing pipeline **correctly implements per-channel independence** for:
- Parameter storage and retrieval
- Processing application
- File output and storage
- State management

However, the **global `from_stage` limitation** prevents processing channels at different stages in a single call. This should be fixed to achieve true per-channel independence.

