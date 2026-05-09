"""
Object-Level Random Forest Classifier for Exosome Detection.

Classifies already-detected objects (from blob/SAM/RF pixel detection) as
True Exosome (1) vs Noise (0) using per-object geometric and intensity features.

This is distinct from random_forest_segmentation.py which operates at the
pixel level. This module operates at the object level.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
import cv2


# ---------------------------------------------------------------------------
# Feature definitions (must stay in sync with the frontend checklist)
# ---------------------------------------------------------------------------

ALL_FEATURES = {
    # Intensity
    "mean_intensity":   {"group": "intensity",  "label": "Mean Intensity"},
    "max_intensity":    {"group": "intensity",  "label": "Max Intensity"},
    "std_intensity":    {"group": "intensity",  "label": "Std Dev Intensity"},
    # Geometry
    "area":             {"group": "geometry",   "label": "Area"},
    "perimeter":        {"group": "geometry",   "label": "Perimeter"},
    "circularity":      {"group": "geometry",   "label": "Circularity"},
    "elongation":       {"group": "geometry",   "label": "Elongation (Aspect Ratio)"},
    "eccentricity":     {"group": "geometry",   "label": "Eccentricity"},
    "solidity":         {"group": "geometry",   "label": "Solidity"},
}

DEFAULT_FEATURES = {k: True for k in ALL_FEATURES}


def compute_object_features(
    mask: np.ndarray,
    image: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full enriched feature dict for a single object.

    Args:
        mask:  2-D boolean/uint8 array (H, W), non-zero = object pixels.
        image: 2-D float or int array (H, W), the source channel image.

    Returns:
        Dict mapping feature name -> float value.
    """
    mask_bool = mask.astype(bool)

    # --- Intensity features ---
    pixels = image[mask_bool].astype(np.float64)
    if pixels.size == 0:
        mean_i = max_i = std_i = 0.0
    else:
        mean_i = float(np.mean(pixels))
        max_i  = float(np.max(pixels))
        std_i  = float(np.std(pixels))

    # --- Geometry features ---
    mask_u8 = mask_bool.astype(np.uint8) * 255
    area = float(np.sum(mask_bool))

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "mean_intensity": mean_i,
            "max_intensity":  max_i,
            "std_intensity":  std_i,
            "area":           area,
            "perimeter":      0.0,
            "circularity":    0.0,
            "elongation":     1.0,
            "eccentricity":   0.0,
            "solidity":       1.0,
        }

    contour = max(contours, key=cv2.contourArea)

    perimeter = float(cv2.arcLength(contour, True))
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Elongation = major_axis / minor_axis from fitted ellipse
    elongation = 1.0
    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            (_, _), (minor, major), _ = cv2.fitEllipse(contour)
            if minor > 0:
                elongation = float(major / minor)
            # Eccentricity = sqrt(1 - (b/a)^2)
            a = max(major, minor) / 2.0
            b = min(major, minor) / 2.0
            if a > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2)))
        except cv2.error:
            pass

    # Solidity = area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 1.0

    return {
        "mean_intensity": mean_i,
        "max_intensity":  max_i,
        "std_intensity":  std_i,
        "area":           area,
        "perimeter":      perimeter,
        "circularity":    min(1.0, circularity),
        "elongation":     elongation,
        "eccentricity":   eccentricity,
        "solidity":       min(1.0, solidity),
    }


def build_feature_vector(
    features: Dict[str, float],
    selected: Dict[str, bool],
) -> np.ndarray:
    """Return a 1-D feature vector containing only the selected features."""
    return np.array(
        [features[k] for k, enabled in selected.items() if enabled],
        dtype=np.float64,
    )


def train_object_classifier(
    labeled_objects: List[Dict[str, Any]],
    selected_features: Dict[str, bool],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> RandomForestClassifier:
    """
    Train an object-level RF classifier.

    Args:
        labeled_objects: List of dicts, each with:
            - 'features': Dict[str, float]  (from compute_object_features)
            - 'label': int  (1 = True Exosome, 0 = Noise)
        selected_features: Which features to use (name -> bool).
        n_estimators: Number of trees.
        max_depth: Max tree depth (None = unlimited).

    Returns:
        Trained RandomForestClassifier.
    """
    if not labeled_objects:
        raise ValueError("No labeled objects provided.")

    X = np.array([
        build_feature_vector(obj["features"], selected_features)
        for obj in labeled_objects
    ])
    y = np.array([obj["label"] for obj in labeled_objects], dtype=int)

    classes = np.unique(y)
    if len(classes) < 2:
        present = "Exosome" if 1 in classes else "Noise"
        missing = "Noise" if 1 in classes else "Exosome"
        raise ValueError(
            f"Need both Exosome and Noise labels to train. "
            f"Only '{present}' provided. Please label at least one '{missing}' object."
        )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)
    return rf


def predict_objects(
    rf: RandomForestClassifier,
    object_features_list: List[Dict[str, float]],
    selected_features: Dict[str, bool],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Classify a list of objects.

    Returns list of dicts with keys:
        - 'is_exosome': bool
        - 'confidence': float  (probability of class=1)
        - 'feature_importances': Dict[str, float]
    """
    if not object_features_list:
        return []

    X = np.array([
        build_feature_vector(f, selected_features)
        for f in object_features_list
    ])

    proba = rf.predict_proba(X)
    exosome_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else -1

    # Feature importances keyed by selected feature names
    selected_names = [k for k, v in selected_features.items() if v]
    importances = dict(zip(selected_names, rf.feature_importances_.tolist()))

    results = []
    for i in range(len(object_features_list)):
        conf = float(proba[i, exosome_idx]) if exosome_idx >= 0 else 0.0
        results.append({
            "is_exosome": conf >= threshold,
            "confidence": conf,
            "feature_importances": importances,
        })
    return results


def save_model(
    rf: RandomForestClassifier,
    selected_features: Dict[str, bool],
    save_dir: Path,
    name: str = "object_classifier",
) -> Tuple[Path, Path]:
    """
    Save RF model (.pkl) and feature config (.json).

    Returns: (model_path, config_path)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path  = save_dir / f"{name}.pkl"
    config_path = save_dir / f"{name}_features.json"

    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    with open(config_path, "w") as f:
        json.dump({"selected_features": selected_features}, f, indent=2)

    return model_path, config_path


def load_model(
    save_dir: Path,
    name: str = "object_classifier",
) -> Tuple[RandomForestClassifier, Dict[str, bool]]:
    """
    Load RF model + feature config.

    Returns: (rf_classifier, selected_features)
    """
    model_path  = save_dir / f"{name}.pkl"
    config_path = save_dir / f"{name}_features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config not found: {config_path}")

    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    return rf, config.get("selected_features", DEFAULT_FEATURES)
