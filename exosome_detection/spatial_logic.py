"""
Reference-centered colocalization logic for exosome analysis.

This module treats one channel as the reference (anchor) and evaluates
marker-channel presence per reference object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class DetectionObject:
    object_id: int
    area: float
    perimeter: float
    circularity: float
    centroid_x: float
    centroid_y: float
    bbox: Tuple[float, float, float, float]
    score: float = 0.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _bbox_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _build_centroid_array(objects: List[DetectionObject]) -> np.ndarray:
    if not objects:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[o.centroid_x, o.centroid_y] for o in objects], dtype=np.float64)


def _channel_combo_label(positive_channels: List[str]) -> str:
    if not positive_channels:
        return "Negative"
    if len(positive_channels) == 1:
        return f"{positive_channels[0]} only"
    return " + ".join(sorted(positive_channels))


def _compute_overlap_matches(
    ref_objects: List[DetectionObject],
    marker_objects: List[DetectionObject],
    ref_masks: Optional[np.ndarray],
    marker_masks: Optional[np.ndarray],
) -> Dict[int, Dict[str, Any]]:
    """
    Returns mapping:
      ref_id -> {"count": int, "matched_ids": [int], "nearest_distance": float|None}
    """
    out: Dict[int, Dict[str, Any]] = {
        r.object_id: {"count": 0, "matched_ids": [], "nearest_distance": None}
        for r in ref_objects
    }
    ref_by_id = {r.object_id: r for r in ref_objects}

    # Fast path: label masks are available and shape-compatible.
    if (
        ref_masks is not None
        and marker_masks is not None
        and ref_masks.shape == marker_masks.shape
        and ref_masks.ndim == 2
    ):
        overlap_pixels = np.where((ref_masks > 0) & (marker_masks > 0))
        if overlap_pixels[0].size > 0:
            ref_ids = ref_masks[overlap_pixels]
            marker_ids = marker_masks[overlap_pixels]
            overlap_pairs = set(
                (int(rid), int(mid))
                for rid, mid in zip(ref_ids.tolist(), marker_ids.tolist())
                if int(rid) > 0 and int(mid) > 0
            )
            marker_by_id = {m.object_id: m for m in marker_objects}
            for rid, mid in overlap_pairs:
                if rid not in out or mid not in marker_by_id:
                    continue
                r = ref_by_id[rid]
                m = marker_by_id[mid]
                d = float(np.hypot(r.centroid_x - m.centroid_x, r.centroid_y - m.centroid_y))
                out[rid]["matched_ids"].append(mid)
                out[rid]["count"] += 1
                prev = out[rid]["nearest_distance"]
                out[rid]["nearest_distance"] = d if prev is None else min(prev, d)
            return out

    # Fallback: bbox overlap + centroid distance.
    for r in ref_objects:
        nearest_d: Optional[float] = None
        matched: List[int] = []
        for m in marker_objects:
            if not _bbox_overlap(r.bbox, m.bbox):
                continue
            d = float(np.hypot(r.centroid_x - m.centroid_x, r.centroid_y - m.centroid_y))
            matched.append(m.object_id)
            nearest_d = d if nearest_d is None else min(nearest_d, d)
        out[r.object_id] = {
            "count": len(matched),
            "matched_ids": matched,
            "nearest_distance": nearest_d,
        }
    return out


def _compute_nearest_matches(
    ref_objects: List[DetectionObject],
    marker_objects: List[DetectionObject],
    distance_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    """
    Assign each marker to at most one nearest reference object within threshold.
    One reference object can receive multiple markers.
    """
    out: Dict[int, Dict[str, Any]] = {
        r.object_id: {"count": 0, "matched_ids": [], "nearest_distance": None}
        for r in ref_objects
    }
    if not ref_objects or not marker_objects:
        return out

    ref_coords = _build_centroid_array(ref_objects)
    marker_coords = _build_centroid_array(marker_objects)
    tree = cKDTree(ref_coords)

    # cKDTree already returns one nearest ref per marker.
    dists, idxs = tree.query(marker_coords, k=1, distance_upper_bound=distance_threshold)

    for marker_idx, (dist, ref_idx) in enumerate(zip(dists.tolist(), idxs.tolist())):
        if np.isinf(dist):
            continue
        if ref_idx < 0 or ref_idx >= len(ref_objects):
            continue
        ref_id = ref_objects[ref_idx].object_id
        marker_id = marker_objects[marker_idx].object_id
        out[ref_id]["matched_ids"].append(marker_id)
        out[ref_id]["count"] += 1
        prev = out[ref_id]["nearest_distance"]
        out[ref_id]["nearest_distance"] = dist if prev is None else min(prev, dist)
    return out


def run_reference_colocalization(
    reference_channel: str,
    marker_channels: List[str],
    detections_by_channel: Dict[str, List[DetectionObject]],
    analysis_mode: str,
    distance_threshold: float,
    marker_names: Optional[Dict[str, str]] = None,
    label_masks_by_channel: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Perform reference-centered colocalization analysis.
    """
    marker_names = marker_names or {}
    label_masks_by_channel = label_masks_by_channel or {}

    ref_objects = detections_by_channel.get(reference_channel, [])
    if not ref_objects:
        return {
            "reference_table": [],
            "summary": {
                "total_reference_objects": 0,
                "marker_positive_reference_objects": 0,
                "marker_negative_reference_objects": 0,
                "overall_positive_rate": 0.0,
                "total_matched_marker_objects": 0,
                "analysis_mode": analysis_mode,
                "distance_threshold": distance_threshold,
                "reference_channel": reference_channel,
            },
            "channel_summary": [],
            "combination_summary": [],
            "overlay": {"positive_reference_ids": []},
        }

    per_channel_match: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for ch in marker_channels:
        marker_objs = detections_by_channel.get(ch, [])
        if analysis_mode == "nearest_centroid":
            per_channel_match[ch] = _compute_nearest_matches(
                ref_objects=ref_objects,
                marker_objects=marker_objs,
                distance_threshold=distance_threshold,
            )
        else:
            per_channel_match[ch] = _compute_overlap_matches(
                ref_objects=ref_objects,
                marker_objects=marker_objs,
                ref_masks=label_masks_by_channel.get(reference_channel),
                marker_masks=label_masks_by_channel.get(ch),
            )

    reference_table: List[Dict[str, Any]] = []
    combo_counts: Dict[str, int] = {}
    positive_reference_ids: List[int] = []
    total_matched_markers = 0

    for r in ref_objects:
        row: Dict[str, Any] = {
            "reference_object_id": r.object_id,
            "reference_channel": reference_channel,
            "reference_centroid_x": r.centroid_x,
            "reference_centroid_y": r.centroid_y,
            "reference_area": r.area,
            "reference_perimeter": r.perimeter,
            "reference_circularity": r.circularity,
        }
        positive_channels: List[str] = []
        marker_counts_sum = 0

        for ch in marker_channels:
            m = per_channel_match[ch].get(r.object_id, {"count": 0, "matched_ids": [], "nearest_distance": None})
            count = int(m["count"])
            matched_ids = [int(x) for x in m["matched_ids"]]
            nearest_d = m["nearest_distance"]
            positive = count > 0

            row[f"{ch}_positive"] = positive
            row[f"{ch}_count"] = count
            row[f"{ch}_nearest_distance"] = round(float(nearest_d), 3) if nearest_d is not None else None
            row[f"{ch}_matched_ids"] = matched_ids

            if positive:
                positive_channels.append(ch)
            marker_counts_sum += count

        combo = _channel_combo_label(positive_channels)
        row["biomarker_combination_label"] = combo
        row["total_positive_marker_count"] = marker_counts_sum
        row["overall_status"] = "Positive" if positive_channels else "Negative"
        row["positive_channels"] = sorted(positive_channels)
        row["positive_biomarkers"] = [marker_names.get(ch, "") for ch in sorted(positive_channels)]

        reference_table.append(row)
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
        total_matched_markers += marker_counts_sum
        if positive_channels:
            positive_reference_ids.append(r.object_id)

    total_ref = len(ref_objects)
    total_positive_ref = len(positive_reference_ids)
    total_negative_ref = total_ref - total_positive_ref

    channel_summary: List[Dict[str, Any]] = []
    for ch in marker_channels:
        total_marker_objects = len(detections_by_channel.get(ch, []))
        positive_rows = [row for row in reference_table if bool(row.get(f"{ch}_positive"))]
        pos_count = len(positive_rows)
        avg_count = (
            float(np.mean([row.get(f"{ch}_count", 0) for row in positive_rows]))
            if positive_rows
            else 0.0
        )
        nearest_values = [
            row.get(f"{ch}_nearest_distance")
            for row in positive_rows
            if row.get(f"{ch}_nearest_distance") is not None
        ]
        median_nearest = float(np.median(nearest_values)) if nearest_values else None
        channel_summary.append({
            "channel": ch,
            "biomarker": marker_names.get(ch, ""),
            "total_marker_objects": total_marker_objects,
            "positive_reference_objects": pos_count,
            "positive_rate": round((pos_count / total_ref * 100.0) if total_ref > 0 else 0.0, 3),
            "avg_marker_count_per_positive_reference": round(avg_count, 3),
            "median_nearest_distance": round(median_nearest, 3) if median_nearest is not None else None,
        })

    combo_summary = [
        {
            "combination": k,
            "count": v,
            "rate": (v / total_ref) if total_ref > 0 else 0.0,
            "percentage": (v / total_ref * 100.0) if total_ref > 0 else 0.0,
        }
        for k, v in combo_counts.items()
    ]
    combo_summary.sort(key=lambda x: x["count"], reverse=True)

    summary = {
        "total_reference_objects": total_ref,
        "marker_positive_reference_objects": total_positive_ref,
        "marker_negative_reference_objects": total_negative_ref,
        "overall_positive_rate": round((total_positive_ref / total_ref * 100.0) if total_ref > 0 else 0.0, 3),
        "total_matched_marker_objects": total_matched_markers,
        "analysis_mode": analysis_mode,
        "distance_threshold": distance_threshold,
        "reference_channel": reference_channel,
        "marker_channels": marker_channels,
    }

    return {
        "reference_table": reference_table,
        "summary": summary,
        "channel_summary": channel_summary,
        "combination_summary": combo_summary,
        "overlay": {
            "positive_reference_ids": positive_reference_ids,
            "reference_channel": reference_channel,
        },
    }
