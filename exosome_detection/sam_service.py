"""
SAM (Segment Anything Model) Service for Exosome Detection
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
import cv2
from skimage import measure, morphology

# Try to import segment_anything
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("WARNING: segment_anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


# Global model cache
_model_cache: Dict[str, Any] = {}


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device for PyTorch."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def load_sam_model(
    checkpoint_path: str,
    model_type: str = "sam_vit_h",
    device: str = "auto"
) -> Any:
    """
    Load SAM model with caching.
    
    Args:
        checkpoint_path: Path to SAM checkpoint file
        model_type: Model type (sam_vit_h, sam_vit_l, sam_vit_b)
        device: Device string ('auto', 'cuda', 'cpu')
    
    Returns:
        SAM predictor object
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "segment_anything is not installed. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    
    cache_key = f"{checkpoint_path}_{model_type}_{device}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    
    device_obj = get_device(device)
    
    print(f"[SAM] Loading model {model_type} from {checkpoint_path} on {device_obj}")
    
    # Map our model type names to SAM registry keys
    model_type_map = {
        'sam_vit_h': 'vit_h',
        'sam_vit_l': 'vit_l',
        'sam_vit_b': 'vit_b',
        'vit_h': 'vit_h',
        'vit_l': 'vit_l',
        'vit_b': 'vit_b',
    }
    
    sam_registry_key = model_type_map.get(model_type, model_type)
    
    if sam_registry_key not in sam_model_registry:
        available = list(sam_model_registry.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {available}. "
            f"Mapped to: {sam_registry_key}"
        )
    
    try:
        sam = sam_model_registry[sam_registry_key](checkpoint=str(checkpoint_path))
        sam.to(device=device_obj)
        predictor = SamPredictor(sam)
        
        _model_cache[cache_key] = predictor
        print(f"[SAM] Model loaded and cached")
        
        return predictor
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM model: {e}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for SAM (convert to RGB uint8 if needed).
    
    Args:
        image: Input image (any dtype, any channels)
    
    Returns:
        Preprocessed RGB uint8 image
    """
    # Handle different input formats
    if len(image.shape) == 2:
        # Grayscale -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] != 3:
        # Multi-channel -> use first 3 channels
        image = image[:, :, :3]
    
    # Normalize to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() > 1.0:
            # Assume 16-bit or float
            image = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        else:
            # Assume float [0, 1]
            image = (image * 255).astype(np.uint8)
    
    return image


def postprocess_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int = 10,
    max_area: int = 10000,
    remove_small_objects: bool = True,
    fill_holes: bool = False,
    score_thresh: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Post-process masks: filter by area, remove small objects, fill holes.
    
    Args:
        masks: Boolean masks (N, H, W)
        scores: Confidence scores (N,)
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels
        remove_small_objects: Whether to remove small objects
        fill_holes: Whether to fill holes
        score_thresh: Minimum score threshold
    
    Returns:
        Filtered masks and scores
    """
    if masks.shape[0] == 0:
        return masks, scores
    
    # Filter by score
    valid_mask = scores >= score_thresh
    masks = masks[valid_mask]
    scores = scores[valid_mask]
    
    if masks.shape[0] == 0:
        return masks, scores
    
    filtered_masks = []
    filtered_scores = []
    
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        
        # Fill holes if requested
        if fill_holes:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            mask = cv2.fillPoly(mask, [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], 255)
            mask = (mask > 0).astype(bool)
        
        # Remove small objects if requested
        if remove_small_objects:
            mask = morphology.remove_small_objects(mask, min_size=min_area)
        
        # Calculate area
        area = np.sum(mask)
        
        # Filter by area
        if min_area <= area <= max_area:
            filtered_masks.append(mask)
            filtered_scores.append(scores[i])
    
    if len(filtered_masks) == 0:
        return np.zeros((0, masks.shape[1], masks.shape[2]), dtype=bool), np.array([])
    
    return np.array(filtered_masks, dtype=bool), np.array(filtered_scores)


def compute_detection_features(masks: np.ndarray) -> List[Dict[str, Any]]:
    """
    Compute features for each mask: area, centroid, bbox.
    
    Args:
        masks: Boolean masks (N, H, W)
    
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    for i in range(masks.shape[0]):
        mask = masks[i]
        
        # Get connected components (should be single component per mask)
        labeled = measure.label(mask)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            continue
        
        # Use largest region
        region = max(regions, key=lambda r: r.area)
        
        # Extract features
        area = region.area
        centroid = [region.centroid[1], region.centroid[0]]  # (x, y) format
        bbox = [
            region.bbox[1],  # x1
            region.bbox[0],  # y1
            region.bbox[3],  # x2
            region.bbox[2],  # y2
        ]
        
        # Compute perimeter and circularity from the region mask
        region_mask = (region.image).astype(np.uint8) * 255
        contours_sam, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_sam:
            perimeter = float(cv2.arcLength(contours_sam[0], True))
            circularity = round(min(1.0, (4.0 * np.pi * float(area)) / (perimeter ** 2)), 4) if perimeter > 0 else 0.0
        else:
            perimeter = 0.0
            circularity = 0.0

        detections.append({
            'area':        float(area),
            'centroid':    [float(centroid[0]), float(centroid[1])],
            'bbox':        [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            'perimeter':   round(perimeter, 2),
            'circularity': circularity,
        })
    
    return detections


def segment_exosomes(
    image: np.ndarray,
    mode: str,
    prompts: Dict[str, Any],
    ckpt_path: str,
    model_type: str = "sam_vit_h",
    device: str = "auto",
    score_thresh: float = 0.0,
    min_area: int = 10,
    max_area: int = 10000,
    remove_small_objects: bool = True,
    fill_holes: bool = False,
) -> Dict[str, Any]:
    """
    Segment exosomes using SAM.
    
    Args:
        image: Input image (numpy array, any dtype)
        mode: Detection mode ('box', 'point', 'auto')
        prompts: Prompt dictionary
            - For 'box': {"box": [x1, y1, x2, y2]}
            - For 'point': {"points": [[x, y], ...], "labels": [1, 0, ...]}
        ckpt_path: Path to SAM checkpoint
        model_type: Model type (sam_vit_h, sam_vit_l, sam_vit_b)
        device: Device ('auto', 'cuda', 'cpu')
        score_thresh: Minimum score threshold
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels
        remove_small_objects: Whether to remove small objects
        fill_holes: Whether to fill holes
    
    Returns:
        Dictionary with:
            - masks: (N, H, W) boolean array
            - scores: (N,) float array
            - bboxes: (N, 4) int array [x1, y1, x2, y2]
            - centroids: (N, 2) float array [x, y]
            - detections: List of detection dicts
    """
    if not SAM_AVAILABLE:
        raise ImportError(
            "segment_anything is not installed. "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    
    # Preprocess image
    image_rgb = preprocess_image(image)
    original_shape = image_rgb.shape[:2]  # (H, W)
    
    # Load model
    predictor = load_sam_model(ckpt_path, model_type, device)
    
    # Set image
    predictor.set_image(image_rgb)
    
    # Prepare prompts
    if mode == 'box':
        if 'box' not in prompts:
            raise ValueError("Box mode requires 'box' in prompts")
        box = np.array(prompts['box'], dtype=np.float32)
        # SAM expects box in format [x1, y1, x2, y2]
        box_input = box.reshape(1, 4)
        point_coords = None
        point_labels = None
    elif mode == 'point':
        if 'points' not in prompts or 'labels' not in prompts:
            raise ValueError("Point mode requires 'points' and 'labels' in prompts")
        point_coords = np.array(prompts['points'], dtype=np.float32)
        point_labels = np.array(prompts['labels'], dtype=np.int32)
        box_input = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Run prediction
    try:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            multimask_output=True,  # Get multiple masks per prompt
        )
        
        # masks: (num_masks, H, W)
        # scores: (num_masks,)
        
        # Use the best mask (highest score) for each prompt
        # For simplicity, we'll use all masks and filter later
        # Flatten to get all masks
        all_masks = masks.reshape(-1, masks.shape[1], masks.shape[2])
        all_scores = scores.flatten()
        
        # Post-process
        filtered_masks, filtered_scores = postprocess_masks(
            all_masks,
            all_scores,
            min_area=min_area,
            max_area=max_area,
            remove_small_objects=remove_small_objects,
            fill_holes=fill_holes,
            score_thresh=score_thresh,
        )
        
        # Compute features
        detections = compute_detection_features(filtered_masks)
        
        # Convert masks to list of 2D arrays for JSON serialization
        masks_list = [mask.tolist() for mask in filtered_masks]
        
        # Extract bboxes and centroids
        bboxes = np.array([d['bbox'] for d in detections]) if detections else np.zeros((0, 4), dtype=int)
        centroids = np.array([d['centroid'] for d in detections]) if detections else np.zeros((0, 2), dtype=float)
        
        return {
            'masks': masks_list,  # List of 2D boolean arrays
            'scores': filtered_scores.tolist(),
            'bboxes': bboxes.tolist(),
            'centroids': centroids.tolist(),
            'detections': detections,
        }
        
    except Exception as e:
        raise RuntimeError(f"SAM prediction failed: {e}")
    finally:
        # Clear image from predictor to free memory
        predictor.reset_image()


if __name__ == "__main__":
    # Test/demo code
    print("SAM Service for Exosome Detection")
    print(f"SAM Available: {SAM_AVAILABLE}")

