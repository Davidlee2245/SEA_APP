# 🔬 How Colocalization is Counted

## Overview

Colocalization measures how many objects (exosomes/cells) from different fluorescence channels are located **close to each other** in the same image. This indicates that the same biological structure is labeled by multiple markers.

---

## Step-by-Step Algorithm

### **Step 1: Extract Coordinates**

For each channel, we get the center coordinates of all detected objects:

```python
# Channel 1: 83 detections at coordinates (x1, y1), (x2, y2), ...
# Channel 2: 67 detections at coordinates (x1, y1), (x2, y2), ...
```

**Note:** Coordinates are stored as `(y, x)` in the detection data, but converted to `(x, y)` for distance calculation.

---

### **Step 2: Calculate Pairwise Distances**

For **every pair** of objects (one from ch1, one from ch2), calculate the Euclidean distance:

```python
from scipy.spatial.distance import cdist

# Calculate distance matrix: distances[i, j] = distance between 
# detection i in ch1 and detection j in ch2
distances = cdist(coords1, coords2)
```

**Example:**
- ch1 detection at (100, 200)
- ch2 detection at (105, 202)
- Distance = √[(105-100)² + (202-200)²] = √[25 + 4] = **5.39 pixels**

---

### **Step 3: Find Candidate Pairs**

Find all pairs where the distance is **≤ threshold**:

```python
distance_threshold = 10.0  # pixels (from config.yaml)

candidates = []
for i, det1 in enumerate(detections_ch1):
    for j, det2 in enumerate(detections_ch2):
        dist = distances[i, j]
        if dist <= distance_threshold:
            candidates.append((i, j, dist, id1, id2))
```

**Example:**
- If threshold = 10 pixels
- Pairs with distance ≤ 10 are considered "colocalized"
- Pairs with distance > 10 are **not** colocalized

---

### **Step 4: Greedy Matching (Avoid Double-Counting)**

**Problem:** One object in ch1 might be close to multiple objects in ch2. We need to avoid counting the same object twice.

**Solution:** Use **greedy matching** - match closest pairs first, and mark objects as "used" so they can't be matched again.

```python
# Sort candidates by distance (closest first)
candidates.sort(key=lambda x: x[2])

# Match closest pairs, avoiding duplicates
matched_ch1 = set()  # Track which ch1 objects are already matched
matched_ch2 = set()  # Track which ch2 objects are already matched

for i, j, dist, id1, id2 in candidates:
    if i not in matched_ch1 and j not in matched_ch2:
        # This pair is colocalized!
        colocalized_pairs.append({
            'ch1_id': id1,
            'ch2_id': id2,
            'distance': dist
        })
        matched_ch1.add(i)  # Mark as used
        matched_ch2.add(j)  # Mark as used
```

**Example:**
```
ch1 object A is 5px from ch2 object X
ch1 object A is 8px from ch2 object Y
ch1 object B is 3px from ch2 object X

Sorted by distance:
1. A-X: 5px  → Match! (A and X are now used)
2. B-X: 3px  → Skip (X already matched)
3. A-Y: 8px  → Skip (A already matched)

Result: 1 colocalized pair (A-X)
```

---

### **Step 5: Calculate Statistics**

```python
colocalized_count = len(colocalized_pairs)
colocalization_rate_ch1 = colocalized_count / len(detections_ch1)  # % of ch1 objects colocalized
colocalization_rate_ch2 = colocalized_count / len(detections_ch2)  # % of ch2 objects colocalized
```

**Example:**
- ch1: 83 detections
- ch2: 67 detections
- Colocalized pairs: 50
- Rate ch1: 50/83 = **60.2%**
- Rate ch2: 50/67 = **74.6%**

---

## Multi-Channel Colocalization

For images with **3+ channels**, we check **all unique pairs**:

```python
channels = [ch1, ch2, ch3, ch4]

# Check all pairs:
ch1 ↔ ch2: 50 pairs
ch1 ↔ ch3: 40 pairs
ch1 ↔ ch4: 30 pairs
ch2 ↔ ch3: 45 pairs
ch2 ↔ ch4: 25 pairs
ch3 ↔ ch4: 35 pairs

Total: 225 colocalized pairs
```

**Note:** Each pair is counted once. An object can be colocalized with multiple channels, but each match is counted separately.

---

## Configuration

The distance threshold is configurable in `config/config.yaml`:

```yaml
analyst:
  colocalization_distance: 10.0  # Maximum distance (pixels)
```

**What this means:**
- **Lower value (e.g., 3.0)**: Stricter - only very close objects count
- **Higher value (e.g., 10.0)**: More lenient - accounts for:
  - Chromatic aberration (different wavelengths focus at slightly different positions)
  - Small alignment errors
  - Biological variation

**Current setting: 10.0 pixels** - Good balance for multi-channel fluorescence microscopy.

---

## CSV Output

Each detection in the CSV has a `colocalized` column:

```csv
channel,object_id,x_coord,y_coord,colocalized,...
ch1,1,100,200,True,...
ch1,2,150,250,False,...
ch2,1,105,202,True,...
```

- `colocalized = True`: This object is part of a colocalized pair
- `colocalized = False`: This object is not colocalized with any other channel

---

## Visual Example

```
Channel 1 (Red):     Channel 2 (Green):    Colocalized:
  ●                    ●                      ● (both)
  ●                    ●                      ● (both)
  ●                    ○                      ○ (only red)
  ○                    ●                      ○ (only green)
```

**Result:**
- 2 colocalized pairs (both channels have objects at same location)
- 1 object only in ch1
- 1 object only in ch2

---

## Key Points

1. **Distance-based**: Uses Euclidean distance between object centers
2. **Threshold**: Objects within 10 pixels are considered colocalized
3. **One-to-one matching**: Each object can only be matched once (greedy algorithm)
4. **All channel pairs**: Checks all unique combinations in multi-channel images
5. **Configurable**: Distance threshold can be adjusted in config.yaml

---

## Why This Method?

- **Simple and fast**: O(n×m) distance calculation, then O(k log k) sorting
- **Robust**: Handles cases where one channel has more detections than another
- **No double-counting**: Greedy matching ensures each object is counted once
- **Interpretable**: Distance threshold has clear biological meaning

---

## Example Calculation

**Input:**
- ch1: 3 detections at (100, 100), (200, 200), (300, 300)
- ch2: 2 detections at (105, 102), (295, 298)
- Threshold: 10 pixels

**Distances:**
```
        ch2[0]    ch2[1]
ch1[0]    5.39     282.84
ch1[1]  141.42     141.42
ch1[2]  282.84       5.39
```

**Candidates (≤ 10px):**
- ch1[0] ↔ ch2[0]: 5.39px ✓
- ch1[2] ↔ ch2[1]: 5.39px ✓

**Result:**
- **2 colocalized pairs**
- ch1[1] is not colocalized (too far from both ch2 objects)

---

**This is how colocalization is counted in the SEA pipeline!** 🔬

