# 🔧 Pipeline Log Capture Fix

## Problem

The Pipeline Control tab was only showing high-level summary messages (e.g., "Phase 3: Detection and Quantification", "✅ Sample 1 completed") but **not the detailed logs** from each phase.

**What was missing:**
- Inspector phase logs (SNR calculations, denoising decisions, etc.)
- Aligner phase logs (feature detection, matching, transformation details)
- Analyst phase logs (StarDist detection, filtering, colocalization calculations)

**Root Cause:**
- Pipeline components use `loguru` logger which writes to stderr
- PipelineRunner only emitted its own high-level messages via `_emit_log()`
- Detailed logs from Inspector/Aligner/Analyst were not being captured

---

## Solution

### **1. Capture stdout/stderr** ✅

Added `_capture_output()` context manager that:
- Intercepts all stdout/stderr writes
- Forwards lines to log callbacks
- Still writes to original streams (so terminal still works)

### **2. Intercept loguru logs** ✅

Added custom loguru sink that:
- Captures all loguru messages (from Inspector, Aligner, Analyst)
- Extracts just the message text (no timestamps/formatting)
- Forwards to WebSocket via `_emit_log()`
- Added **AFTER** pipeline initialization (so it doesn't get removed by `logger.remove()`)

### **3. Wrap entire execution** ✅

Wrapped the entire pipeline execution in the output capture context, so all logs are captured from:
- Pipeline initialization
- Sample processing
- All three phases (Inspector, Aligner, Analyst)
- Error messages

---

## Implementation Details

### **Context Manager for stdout/stderr:**

```python
@contextlib.contextmanager
def _capture_output(self, stdout_callback, stderr_callback):
    """Capture stdout/stderr and forward to callbacks"""
    class StreamForwarder:
        def write(self, text):
            # Write to original stream AND forward to callback
            self.original_stream.write(text)
            # Process lines and forward
```

### **Loguru Sink:**

```python
def loguru_sink(message):
    """Forward loguru messages to log callback"""
    record = message.record
    log_msg = record["message"]  # Just the message, no formatting
    self._emit_log(log_msg.rstrip())
```

**Key:** Added **AFTER** `ExosomeAnalysisPipeline(config_path)` so it doesn't get removed by `logger.remove()`.

---

## What You'll See Now

### **Before (Missing logs):**
```
📊 Phase 3: Detection and Quantification
✅ Sample 1 completed successfully
```

### **After (Full logs):**
```
📊 Phase 3: Detection and Quantification
Analyzing sample: 1
Detecting objects in channel: r01c01f01p01-ch2sk1fk1fl1
Loading StarDist model: 2D_versatile_fluo
StarDist model loaded successfully
Filtered 310 -> 193 detections (size: 0, intensity: 115, morphology: 0)
Channel r01c01f01p01-ch2sk1fk1fl1: 193 exosomes detected
Detecting objects in channel: r01c01f01p01-ch3sk1fk1fl1
...
Multi-channel combination analysis:
  Total objects grouped: 222
  Unique combinations: 7
    ch1+ch2+ch3+ch4: 18 objects (8.1%)
    ch1+ch2+ch3: 30 objects (13.5%)
    ...
✅ Sample 1 completed successfully
```

---

## Files Modified

1. ✅ `core/pipeline_runner.py`
   - Added `_capture_output()` context manager
   - Added loguru sink (after pipeline init)
   - Wrapped entire execution in output capture
   - Cleanup loguru handler in finally block

---

## Testing

### **Test 1: Run Pipeline**
```bash
# Start backend
python api_server_extended.py

# In GUI: Pipeline Control tab → Start Pipeline
```

### **Expected:**
- ✅ See detailed logs from Inspector phase
- ✅ See detailed logs from Aligner phase  
- ✅ See detailed logs from Analyst phase
- ✅ See all loguru messages (INFO, WARNING, ERROR)
- ✅ See stdout/stderr messages

### **Test 2: Check Log Viewer**
- Logs should appear in real-time
- Auto-scroll should work
- All phases should show detailed progress

---

## Notes

- **Loguru handler:** Added after pipeline init to avoid being removed
- **stdout/stderr:** Still written to terminal (dual output)
- **Performance:** Minimal overhead (just string forwarding)
- **Cleanup:** Handler removed in finally block

---

**Status:** ✅ **Fixed - All logs now captured and displayed!**

**Date:** January 26, 2026

