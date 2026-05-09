"""
Aivia-style Spatial Relation Analysis for Exosome-Marker Colocalization.

Supported modes:
  overlap_all      - Any pixel overlap between exosome mask and marker mask.
  overlap_full     - Marker mask fully contained within exosome mask.
  overlap_partial  - Partial overlap (any overlap that is NOT full containment).
  overlap_min_pct  - Intersection area / marker area >= min_overlap_pct (%).
  nearest_centroid - Euclidean distance between centroids <= distance_threshold (px).

Per exosome, this reports:
  - relation_count: number of markers that satisfy the spatial criterion.
  - marker_ids:     list of marker IDs that matched.
  - percent_overlaps: list of % overlap values (intersection / marker area * 100).
  - distances:      list of centroid distances (pixels).
"""

import numpy as np
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(det: Dict) -> np.ndarray:
    """Return [x, y] centroid as float64 array."""
    c = det.get("centroid", [0.0, 0.0])
    return np.array([float(c[0]), float(c[1])], dtype=np.float64)


def _mask_array(det: Dict, h: int, w: int) -> Optional[np.ndarray]:
    """
    Reconstruct a 2-D boolean mask from a detection dict.
    Supports:
      - det['mask']:  2-D list/array  (H x W)
      - det['bbox']:  [x1, y1, x2, y2]  → filled rectangle (fallback)
    Returns None if no mask info is available.
    """
    raw = det.get("mask")
    if raw is not None:
        arr = np.asarray(raw, dtype=bool)
        if arr.ndim == 2:
            return arr
    # Fallback: filled bounding-box rectangle
    bbox = det.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        m = np.zeros((h, w), dtype=bool)
        m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        return m
    return None


def _intersection_over_marker(m_exo: np.ndarray, m_marker: np.ndarray) -> float:
    """Intersection area / marker area (0.0 – 1.0)."""
    inter = float(np.sum(m_exo & m_marker))
    marker_area = float(np.sum(m_marker))
    return inter / marker_area if marker_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_spatial_relations(
    primary_detections: List[Dict[str, Any]],
    secondary_detections: Dict[str, List[Dict[str, Any]]],
    mode: str = "overlap_all",
    min_overlap_pct: float = 10.0,
    distance_threshold: float = 5.0,
    image_shape: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """
    Compute spatial relations between primary (exosome) detections and
    per-channel secondary (marker) detections.

    Args:
        primary_detections:   List of exosome detection dicts.
            Each dict must have 'id', 'centroid', and optionally 'mask'/'bbox'.
        secondary_detections: Dict mapping channel_name -> list of marker dicts.
            Same structure as primary_detections.
        mode: One of 'overlap_all', 'overlap_full', 'overlap_partial',
              'overlap_min_pct', 'nearest_centroid'.
        min_overlap_pct:  Threshold for 'overlap_min_pct' mode (%).
        distance_threshold: Threshold for 'nearest_centroid' mode (pixels).
        image_shape: (H, W) used when constructing bbox-based masks.
                     If None, inferred from bbox extents.

    Returns:
        List of per-exosome result dicts:
        {
            "exosome_id": int,
            "centroid": [x, y],
            "area": float,
            "relations": {
                "<channel_name>": {
                    "relation_count": int,
                    "marker_ids": [int, ...],
                    "percent_overlaps": [float, ...],
                    "distances": [float, ...]
                },
                ...
            }
        }
    """
    if not primary_detections:
        return []

    # Determine image dimensions for mask reconstruction
    if image_shape:
        H, W = int(image_shape[0]), int(image_shape[1])
    else:
        # Infer from bounding boxes
        all_bboxes = [
            d.get("bbox", [0, 0, 0, 0])
            for d in primary_detections
        ] + [
            d.get("bbox", [0, 0, 0, 0])
            for dets in secondary_detections.values()
            for d in dets
        ]
        H = max((int(b[3]) for b in all_bboxes if b), default=512)
        W = max((int(b[2]) for b in all_bboxes if b), default=512)
        H = max(H, 1)
        W = max(W, 1)

    results = []

    for exo in primary_detections:
        exo_id   = int(exo.get("id", 0))
        exo_cent = _centroid(exo)
        exo_area = float(exo.get("area", 0))
        exo_mask = _mask_array(exo, H, W)

        per_channel: Dict[str, Dict] = {}

        for ch_name, markers in secondary_detections.items():
            matched_ids: List[int] = []
            pct_overlaps: List[float] = []
            distances: List[float] = []

            for marker in markers:
                m_id   = int(marker.get("id", 0))
                m_cent = _centroid(marker)
                dist   = float(np.linalg.norm(exo_cent - m_cent))

                # --- nearest_centroid mode ---
                if mode == "nearest_centroid":
                    if dist <= distance_threshold:
                        matched_ids.append(m_id)
                        distances.append(round(dist, 2))
                        pct_overlaps.append(0.0)
                    continue

                # --- overlap-based modes ---
                m_mask = _mask_array(marker, H, W)

                if exo_mask is None or m_mask is None:
                    # Fall back to bbox overlap check
                    exo_bb = exo.get("bbox", [0, 0, 0, 0])
                    m_bb   = marker.get("bbox", [0, 0, 0, 0])
                    has_overlap = (
                        exo_bb[0] < m_bb[2] and exo_bb[2] > m_bb[0] and
                        exo_bb[1] < m_bb[3] and exo_bb[3] > m_bb[1]
                    )
                    pct = 100.0 if has_overlap else 0.0
                else:
                    pct = _intersection_over_marker(exo_mask, m_mask) * 100.0

                matched = False
                if mode == "overlap_all":
                    matched = pct > 0.0
                elif mode == "overlap_full":
                    matched = pct >= 99.9  # marker fully inside exosome
                elif mode == "overlap_partial":
                    matched = 0.0 < pct < 99.9
                elif mode == "overlap_min_pct":
                    matched = pct >= min_overlap_pct

                if matched:
                    matched_ids.append(m_id)
                    pct_overlaps.append(round(pct, 2))
                    distances.append(round(dist, 2))

            per_channel[ch_name] = {
                "relation_count":  len(matched_ids),
                "marker_ids":      matched_ids,
                "percent_overlaps": pct_overlaps,
                "distances":       distances,
            }

        results.append({
            "exosome_id": exo_id,
            "centroid":   [round(float(exo_cent[0]), 2), round(float(exo_cent[1]), 2)],
            "area":       exo_area,
            "relations":  per_channel,
        })

    return results


def summarize_spatial_results(
    spatial_results: List[Dict[str, Any]],
    channel_names: List[str],
) -> Dict[str, Any]:
    """
    Aggregate per-exosome spatial results into summary statistics.

    Returns:
    {
        "total_exosomes": int,
        "channels": {
            "<ch>": {
                "exosomes_with_relation": int,
                "total_relations": int,
                "mean_markers_per_exosome": float,
                "mean_max_overlap_pct": float,
            }
        }
    }
    """
    summary: Dict[str, Any] = {
        "total_exosomes": len(spatial_results),
        "channels": {},
    }

    for ch in channel_names:
        exo_with_rel = 0
        total_rel    = 0
        overlap_vals: List[float] = []

        for row in spatial_results:
            rel = row["relations"].get(ch, {})
            count = rel.get("relation_count", 0)
            total_rel += count
            if count > 0:
                exo_with_rel += 1
                pcts = rel.get("percent_overlaps", [])
                if pcts:
                    overlap_vals.append(max(pcts))

        n = len(spatial_results)
        summary["channels"][ch] = {
            "exosomes_with_relation":   exo_with_rel,
            "total_relations":          total_rel,
            "mean_markers_per_exosome": round(total_rel / n, 3) if n > 0 else 0.0,
            "mean_max_overlap_pct":     round(float(np.mean(overlap_vals)), 2) if overlap_vals else 0.0,
        }

    return summary
