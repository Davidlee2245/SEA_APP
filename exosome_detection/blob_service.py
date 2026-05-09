"""
Blob-Based Detection Service for Exosome Detection
Uses scikit-image and OpenCV for blob detection

CONFIGURATION:
- Detects BRIGHT blobs (white dots on dark background) - correct for exosomes
- Uses cv2.THRESH_BINARY: pixels > threshold become 255 (foreground)
- No image inversion is applied
- Equivalent to: filterByColor=True, blobColor=255 (if using SimpleBlobDetector)
"""

import numpy as np
from typing import Dict, Any, List
from skimage import measure, morphology, filters
import cv2


def detect_blobs(
    image: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 10,
    max_area: int = 10000,
    min_circularity: float = 0.3,
    max_circularity: float = 1.0,
    min_inertia_ratio: float = 0.3,
    remove_small_objects: bool = True,
    fill_holes: bool = False,
) -> Dict[str, Any]:
    """
    Detect blobs (exosomes) using thresholding and morphological operations.
    
    CONFIGURATION FOR BRIGHT BLOBS:
    - Uses cv2.THRESH_BINARY: pixels > threshold become 255 (foreground)
    - Detects BRIGHT blobs (white dots on dark background) - correct for exosomes
    - Equivalent to: filterByColor=True, blobColor=255 (if using SimpleBlobDetector)
    - No image inversion is applied
    
    Args:
        image: Input image (numpy array, any dtype)
        threshold: Threshold value for binarization (0-1, normalized)
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels
        min_circularity: Minimum circularity (0-1)
        max_circularity: Maximum circularity (0-1)
        min_inertia_ratio: Minimum inertia ratio (0-1)
        remove_small_objects: Whether to remove small objects
        fill_holes: Whether to fill holes
    
    Returns:
        Dictionary with:
            - masks: (N, H, W) boolean array
            - scores: (N,) float array (all 1.0 for blob detection)
            - bboxes: (N, 4) int array [x1, y1, x2, y2]
            - centroids: (N, 2) float array [x, y]
            - detections: List of detection dicts
    """
    # Normalize image to [0, 1] if needed
    if image.dtype != np.uint8:
        if image.max() > 1.0:
            # Assume 16-bit or float
            image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-10)).astype(np.float32)
        else:
            # Assume float [0, 1]
            image_norm = image.astype(np.float32)
    else:
        image_norm = (image.astype(np.float32) / 255.0)
    
    # Convert to uint8 for OpenCV operations
    image_uint8 = (image_norm * 255).astype(np.uint8)
    
    # Apply threshold
    # THRESH_BINARY: pixels > threshold become 255 (white/foreground) = DETECTS BRIGHT BLOBS
    # THRESH_BINARY_INV: pixels > threshold become 0 (black/background) = DETECTS DARK BLOBS
    # For exosomes (bright dots on dark background), we use THRESH_BINARY to detect bright blobs
    threshold_value = int(threshold * 255)
    _, binary = cv2.threshold(image_uint8, threshold_value, 255, cv2.THRESH_BINARY)
    print(f"[Blob Detection] Using THRESH_BINARY mode: detecting BRIGHT blobs (pixels > {threshold_value} become foreground)")
    
    # If threshold produces very few foreground pixels, try adaptive thresholding as fallback
    foreground_ratio = np.sum(binary > 0) / binary.size
    if foreground_ratio < 0.01:  # Less than 1% foreground
        print(f"[Blob Detection] Simple threshold produced only {foreground_ratio*100:.2f}% foreground, trying adaptive threshold")
        # Use adaptive thresholding with mean method
        binary_adaptive = cv2.adaptiveThreshold(
            image_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive_foreground_ratio = np.sum(binary_adaptive > 0) / binary_adaptive.size
        print(f"[Blob Detection] Adaptive threshold produced {adaptive_foreground_ratio*100:.2f}% foreground")
        if adaptive_foreground_ratio > foreground_ratio:
            binary = binary_adaptive
            print(f"[Blob Detection] Using adaptive threshold result")
    
    # Debug: Check how many pixels passed threshold
    num_foreground_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    print(f"[Blob Detection] After threshold {threshold_value}: {num_foreground_pixels}/{total_pixels} pixels ({100*num_foreground_pixels/total_pixels:.2f}%) are foreground")
    
    # Fill holes if requested
    if fill_holes:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Fill holes using floodFill
        h, w = binary.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(binary.copy(), mask, (0, 0), 255)
        binary = cv2.bitwise_not(mask[1:-1, 1:-1])
    
    # Remove small objects if requested
    if remove_small_objects:
        binary = morphology.remove_small_objects(binary > 0, min_size=min_area).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"[Blob Detection] Found {num_labels - 1} connected components (excluding background)")
    
    masks = []
    detections = []
    scores = []
    filtered_by_area = 0
    filtered_by_circularity = 0
    filtered_by_inertia = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get mask for this component
        mask = (labels == i).astype(bool)
        area = np.sum(mask)
        
        # Filter by area
        if area < min_area or area > max_area:
            filtered_by_area += 1
            continue
        
        # Calculate circularity and inertia ratio
        # Get contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        
        contour = contours[0]
        
        # Calculate circularity: 4π * area / perimeter^2
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            circularity = 0.0
        
        # Filter by circularity
        if circularity < min_circularity or circularity > max_circularity:
            filtered_by_circularity += 1
            continue
        
        # Calculate inertia ratio (ratio of minor to major axis)
        # Using moments
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            # Calculate eigenvalues of covariance matrix
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            
            # Eigenvalues
            lambda1 = 0.5 * (mu20 + mu02 + np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
            lambda2 = 0.5 * (mu20 + mu02 - np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
            
            if lambda1 > 0:
                inertia_ratio = lambda2 / lambda1
            else:
                inertia_ratio = 0.0
        else:
            inertia_ratio = 0.0
        
        # Filter by inertia ratio
        if inertia_ratio < min_inertia_ratio:
            filtered_by_inertia += 1
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]
        
        # Get centroid
        centroid = [float(centroids[i][0]), float(centroids[i][1])]
        
        masks.append(mask.tolist())
        detections.append({
            'area':        float(area),
            'centroid':    centroid,
            'bbox':        bbox,
            'perimeter':   round(perimeter, 2),
            'circularity': round(min(1.0, circularity), 4),
        })
        scores.append(1.0)  # Blob detection doesn't have confidence scores
    
    # Convert masks to list of 2D arrays for JSON serialization
    masks_list = masks
    
    # Extract bboxes and centroids
    bboxes = np.array([d['bbox'] for d in detections]) if detections else np.zeros((0, 4), dtype=int)
    centroids = np.array([d['centroid'] for d in detections]) if detections else np.zeros((0, 2), dtype=float)
    
    print(f"[Blob Detection] Filtering summary: {filtered_by_area} by area, {filtered_by_circularity} by circularity, {filtered_by_inertia} by inertia")
    print(f"[Blob Detection] Final result: {len(detections)} blobs detected")
    
    return {
        'masks': masks_list,
        'scores': scores,
        'bboxes': bboxes.tolist(),
        'centroids': centroids.tolist(),
        'detections': detections,
    }

