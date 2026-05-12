"""
Random Forest Interactive Pixel-Level Classifier for Exosome Detection
Uses scikit-learn RandomForestClassifier with multi-scale features
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
from skimage import filters, feature
import cv2
import json
import pickle
import logging
import datetime
from pathlib import Path
import sklearn
import os


LOGGER = logging.getLogger(__name__)
MAX_RF_MODEL_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB safety ceiling
DEFAULT_FEATURE_SIGMAS: Tuple[float, ...] = (1.0, 2.0, 4.0)
FEATURE_NAMES: Tuple[str, ...] = (
    "intensity_raw",
    "gaussian_sigma_1.0",
    "gaussian_sigma_2.0",
    "gaussian_sigma_4.0",
    "laplacian",
    "gradient_magnitude",
    "local_variance_5x5",
)


def get_feature_params() -> Dict[str, Any]:
    """Return the canonical feature-extraction schema used for RF models."""
    return {
        "feature_names": list(FEATURE_NAMES),
        "gaussian_sigmas": list(DEFAULT_FEATURE_SIGMAS),
        "laplacian": {"enabled": True},
        "gradient_magnitude": {"enabled": True},
        "local_variance": {"window_size": 5},
    }


def _feature_schema_matches(saved_feature_params: Any) -> bool:
    """Check whether saved feature schema matches current extraction schema."""
    if not isinstance(saved_feature_params, dict):
        return False
    current = get_feature_params()
    return (
        saved_feature_params.get("feature_names") == current.get("feature_names")
        and saved_feature_params.get("gaussian_sigmas") == current.get("gaussian_sigmas")
        and saved_feature_params.get("local_variance", {}).get("window_size")
        == current.get("local_variance", {}).get("window_size")
    )


def validate_feature_params(saved_feature_params: Any) -> bool:
    """Public wrapper for feature-schema compatibility checks."""
    return _feature_schema_matches(saved_feature_params)


def save_rf_model(model: RandomForestClassifier, feature_params: Dict[str, Any], save_path: str) -> Dict[str, Any]:
    """
    Save a trained pixel-level Random Forest model and companion metadata.

    Args:
        model: Trained RandomForestClassifier.
        feature_params: Metadata dictionary including at least feature schema and
            optionally trained_on context.
        save_path: Target path for the model pickle file (must end with .pkl).

    Returns:
        The metadata dictionary written to disk.

    Failure modes:
        Raises ValueError for invalid save path or missing model.
        Raises OSError / IOError on filesystem write failures.
        Raises TypeError for non-serializable metadata.
    """
    if model is None:
        raise ValueError("model is required")

    model_path = Path(save_path)
    if model_path.suffix.lower() != ".pkl":
        raise ValueError(f"save_path must end with .pkl, got: {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path.with_name(model_path.name.replace("_rf_model.pkl", "_metadata.json"))
    if metadata_path == model_path:
        metadata_path = model_path.with_suffix(".metadata.json")

    metadata: Dict[str, Any] = {
        "sklearn_version": sklearn.__version__,
        "feature_params": feature_params.get("feature_params", feature_params),
        "trained_on": feature_params.get("trained_on", {}),
        "saved_at": datetime.datetime.now().isoformat(),
    }

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata


def load_rf_model(load_path: str) -> Tuple[RandomForestClassifier, Dict[str, Any], Optional[str]]:
    """
    Load a previously saved pixel-level Random Forest model and metadata.

    Args:
        load_path: Path to model pickle file.

    Returns:
        (model, metadata, warning) where warning is a non-blocking message when
        sklearn version differs, otherwise None.

    Failure modes:
        Raises FileNotFoundError if model or metadata files are missing.
        Raises ValueError if model object is invalid.
        Raises pickle.UnpicklingError / JSONDecodeError for malformed files.
    """
    model_path = Path(load_path)
    if '..' in model_path.parts:
        raise ValueError(f"Refusing to load model path containing '..': {load_path}")

    data_root = Path(os.getenv('SEA_DATA_ROOT', 'data/input')).resolve()
    resolved_model_path = model_path.resolve()
    try:
        resolved_model_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to load RF model outside data root. "
            f"model={resolved_model_path}, data_root={data_root}"
        ) from exc

    parts = resolved_model_path.parts
    if 'rf_models' not in parts:
        raise ValueError(
            f"Refusing to load RF model outside rf_models directory: {resolved_model_path}"
        )
    rf_idx = parts.index('rf_models')
    # Allow legacy: <sample>/rf_models/<channel_key>/<position>_rf_model.pkl
    # Or current: <sample>/rf_models/<stem>.pkl  (e.g. ch0.pkl)
    remainder = parts[rf_idx + 1 :]
    if rf_idx < 1 or len(remainder) < 1:
        raise ValueError(
            f"Invalid RF model path under rf_models: {resolved_model_path}"
        )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {resolved_model_path}")
    if resolved_model_path.stat().st_size > MAX_RF_MODEL_SIZE_BYTES:
        size_mb = resolved_model_path.stat().st_size / (1024 * 1024)
        raise ValueError(
            f"Refusing to load RF model larger than 200 MB "
            f"({size_mb:.2f} MB): {resolved_model_path}"
        )

    model_path = resolved_model_path
    metadata_path = model_path.with_name(model_path.name.replace("_rf_model.pkl", "_metadata.json"))
    if metadata_path == model_path:
        metadata_path = model_path.with_suffix(".metadata.json")

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded object is not a valid RandomForestClassifier-like model")

    saved_version = str(metadata.get("sklearn_version", "")).strip()
    current_version = sklearn.__version__
    warning: Optional[str] = None
    if saved_version and saved_version != current_version:
        warning = (
            f"Model saved with sklearn {saved_version}, "
            f"current environment is sklearn {current_version}."
        )
        LOGGER.warning("[Random Forest] %s", warning)

    return model, metadata, warning


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale features for each pixel.
    
    Features:
    - Raw intensity
    - Gaussian blur (sigma 1, 2, 4)
    - Laplacian of Gaussian
    - Gradient magnitude
    - Local variance (texture window)
    
    Args:
        image: Input image (H, W) numpy array
        
    Returns:
        Feature array (H, W, N_features) where N_features = 7
    """
    # Normalize image to [0, 1] if needed
    if image.dtype != np.float32 and image.dtype != np.float64:
        if image.max() > 1.0:
            image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-10)).astype(np.float32)
        else:
            image_norm = image.astype(np.float32)
    else:
        image_norm = image.astype(np.float32)
        if image_norm.max() > 1.0:
            image_norm = ((image_norm - image_norm.min()) / (image_norm.max() - image_norm.min() + 1e-10))
    
    h, w = image_norm.shape
    features = []
    
    # 1. Raw intensity
    features.append(image_norm)
    
    # 2. Gaussian blur (sigma 1, 2, 4)
    for sigma in DEFAULT_FEATURE_SIGMAS:
        blurred = filters.gaussian(image_norm, sigma=sigma)
        features.append(blurred)
    
    # 3. Laplacian of Gaussian
    log = filters.laplace(image_norm)
    features.append(log)
    
    # 4. Gradient magnitude
    grad_y, grad_x = np.gradient(image_norm)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    features.append(grad_mag)
    
    # 5. Local variance (texture window 5x5)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(image_norm, -1, kernel)
    local_var = cv2.filter2D((image_norm - local_mean)**2, -1, kernel)
    features.append(local_var)
    
    # Stack features: (H, W, N_features)
    feature_array = np.stack(features, axis=-1)
    
    return feature_array


def train_random_forest(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced"
) -> RandomForestClassifier:
    """
    Train Random Forest classifier on user annotations.
    
    Args:
        image: Input image (H, W)
        annotations: List of annotation dicts with:
            - 'points': List of [x, y] coordinates
            - 'label': 1 for exosome, 0 for background
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        class_weight: Class weight strategy
        
    Returns:
        Trained RandomForestClassifier
    """
    # Extract features
    print(f"[Random Forest] Extracting features from image shape {image.shape}...")
    features = extract_features(image)
    h, w, n_features = features.shape
    
    # Collect exosome pixel coords first so they take priority over background
    exosome_pixels = set()
    for ann in annotations:
        if ann['label'] == 1:
            for x, y in ann['points']:
                exosome_pixels.add((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))

    # Collect training samples (exosome pixels override background at same location)
    X_train = []
    y_train = []

    for ann in annotations:
        label = ann['label']  # 1 = exosome, 0 = background
        points = ann['points']  # List of [x, y]

        for x, y in points:
            # Ensure coordinates are within bounds
            y = int(np.clip(y, 0, h - 1))
            x = int(np.clip(x, 0, w - 1))

            # Skip background points that overlap with exosome annotations
            if label == 0 and (x, y) in exosome_pixels:
                continue

            # Get feature vector for this pixel
            feature_vec = features[y, x, :]
            X_train.append(feature_vec)
            y_train.append(label)
    
    if len(X_train) == 0:
        raise ValueError("No training samples collected from annotations")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    unique_classes = np.unique(y_train)
    n_exosome = int(np.sum(y_train == 1))
    n_background = int(np.sum(y_train == 0))
    print(f"[Random Forest] Training on {len(X_train)} samples ({n_exosome} exosome, {n_background} background)...")
    
    if len(unique_classes) < 2:
        present = "Exosome" if 1 in unique_classes else "Background"
        missing = "Background" if 1 in unique_classes else "Exosome"
        raise ValueError(
            f"Need both Exosome and Background annotations to train, "
            f"but only {present} annotations were provided ({len(X_train)} points). "
            f"Please add at least one {missing} annotation."
        )
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    print(f"[Random Forest] Training completed. Feature importance: {rf.feature_importances_}")
    
    return rf


def predict_segmentation(
    image: np.ndarray,
    classifier: RandomForestClassifier,
    confidence_threshold: float = 0.5,
    min_area: int = 0,
    apply_morphology: bool = False
) -> Dict[str, Any]:
    """
    Apply trained classifier to full image and generate segmentation.
    
    Args:
        image: Input image (H, W)
        classifier: Trained RandomForestClassifier
        confidence_threshold: Threshold for binary mask (0-1)
        min_area: Minimum area for connected components (px)
        apply_morphology: Whether to apply morphological closing
        
    Returns:
        Dictionary with:
            - 'probability_map': (H, W) float array [0, 1]
            - 'binary_mask': (H, W) bool array
            - 'overlay': (H, W, 3) uint8 RGB overlay image
    """
    print(f"[Random Forest] Extracting features for prediction...")
    features = extract_features(image)
    h, w, n_features = features.shape
    
    # Reshape for prediction: (H*W, N_features)
    features_flat = features.reshape(-1, n_features)
    
    print(f"[Random Forest] Predicting probabilities for {len(features_flat)} pixels...")
    # Get probability of class 1 (exosome)
    proba_all = classifier.predict_proba(features_flat)
    if proba_all.shape[1] >= 2:
        exosome_idx = list(classifier.classes_).index(1) if 1 in classifier.classes_ else 1
        proba = proba_all[:, exosome_idx]
    else:
        # Single class — fill with 1.0 if that class is exosome, 0.0 if background
        proba = np.ones(len(features_flat)) if classifier.classes_[0] == 1 else np.zeros(len(features_flat))
    
    # Reshape back to image dimensions
    probability_map = proba.reshape(h, w)
    
    print(f"[Random Forest] Probability map: min={probability_map.min():.3f}, max={probability_map.max():.3f}, mean={probability_map.mean():.3f}")
    
    # Threshold to create binary mask
    binary_mask = (probability_map >= confidence_threshold).astype(np.uint8) * 255
    
    # Post-processing: remove small objects
    if min_area > 0:
        print(f"[Random Forest] Removing small objects (min_area={min_area})...")
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Filter by area
        filtered_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 255
        
        binary_mask = filtered_mask
    
    # Optional morphological closing
    if apply_morphology:
        print(f"[Random Forest] Applying morphological closing...")
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create overlay: green semi-transparent overlay on original image
    # Convert image to RGB for overlay
    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()
    
    # Normalize image to [0, 255] for display
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() > 1.0:
            image_rgb = ((image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-10) * 255).astype(np.uint8)
        else:
            image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Create green overlay where mask is active
    overlay = image_rgb.copy()
    mask_bool = (binary_mask > 0)
    overlay[mask_bool] = (overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
    
    return {
        'probability_map': probability_map.tolist(),  # Convert to list for JSON
        'binary_mask': binary_mask.tolist(),
        'overlay': overlay.tolist(),
    }


def segment_with_random_forest(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
    min_area: int = 0,
    apply_morphology: bool = False,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced"
) -> Dict[str, Any]:
    """
    Complete Random Forest segmentation pipeline.
    
    Args:
        image: Input image (H, W)
        annotations: List of annotation dicts with 'points' and 'label'
        confidence_threshold: Threshold for binary mask
        min_area: Minimum area for connected components
        apply_morphology: Whether to apply morphological closing
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        class_weight: Class weight strategy
        
    Returns:
        Dictionary with probability_map, binary_mask, overlay, and detections
    """
    # Train classifier
    classifier = train_random_forest(
        image=image,
        annotations=annotations,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight
    )
    
    # Predict segmentation
    result = predict_segmentation(
        image=image,
        classifier=classifier,
        confidence_threshold=confidence_threshold,
        min_area=min_area,
        apply_morphology=apply_morphology
    )
    
    # Extract detections from binary mask
    binary_mask = np.array(result['binary_mask'])
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8) * 255,
        connectivity=8
    )
    
    detections = []
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Compute perimeter and circularity from the component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(contours[0], True))
            circularity = round(min(1.0, (4.0 * np.pi * float(area)) / (perimeter ** 2)), 4) if perimeter > 0 else 0.0
        else:
            perimeter = 0.0
            circularity = 0.0

        detections.append({
            'area':        float(area),
            'centroid':    [float(centroids[i][0]), float(centroids[i][1])],
            'bbox':        [int(x), int(y), int(x + w), int(y + h)],
            'perimeter':   round(perimeter, 2),
            'circularity': circularity,
        })
    
    result['detections'] = detections
    result['scores'] = [1.0] * len(detections)  # All have same confidence for now
    
    # Convert binary_mask to boolean array for consistency with other methods
    # Random Forest returns a single mask (not a list of masks)
    result['masks'] = [(binary_mask > 0).tolist()]  # Single list: one mask entry containing 2D array
    
    return result


def segment_with_loaded_random_forest(
    image: np.ndarray,
    classifier: RandomForestClassifier,
    confidence_threshold: float = 0.5,
    min_area: int = 0,
    apply_morphology: bool = False
) -> Dict[str, Any]:
    """
    Run segmentation using a pre-trained Random Forest classifier.

    Args:
        image: Input image (H, W).
        classifier: Pre-trained RandomForestClassifier.
        confidence_threshold: Threshold for binary mask.
        min_area: Minimum area for connected components.
        apply_morphology: Whether to apply morphological closing.

    Returns:
        Dictionary with probability_map, binary_mask, overlay, detections, scores, and masks.
    """
    result = predict_segmentation(
        image=image,
        classifier=classifier,
        confidence_threshold=confidence_threshold,
        min_area=min_area,
        apply_morphology=apply_morphology
    )

    binary_mask = np.array(result['binary_mask'])
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8) * 255,
        connectivity=8
    )

    detections = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(contours[0], True))
            circularity = round(min(1.0, (4.0 * np.pi * float(area)) / (perimeter ** 2)), 4) if perimeter > 0 else 0.0
        else:
            perimeter = 0.0
            circularity = 0.0

        detections.append({
            'area': float(area),
            'centroid': [float(centroids[i][0]), float(centroids[i][1])],
            'bbox': [int(x), int(y), int(x + w), int(y + h)],
            'perimeter': round(perimeter, 2),
            'circularity': circularity,
        })

    result['detections'] = detections
    result['scores'] = [1.0] * len(detections)
    result['masks'] = [(binary_mask > 0).tolist()]
    return result

