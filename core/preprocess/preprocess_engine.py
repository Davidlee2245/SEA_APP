"""
Core preprocessing engine for applying parameters to images.
Uses OpenCV and scikit-image for image processing operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


def apply_preprocessing(
    image_path: str,
    params: Dict[str, Any],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Apply preprocessing parameters to an image.
    
    Supported operations (applied in order):
    1. denoising_strength - Non-local means denoising
    2. background_subtraction_radius - Rolling ball background subtraction
    3. clahe_clip_limit, clahe_grid_size - CLAHE contrast enhancement
    4. gamma_correction - Gamma adjustment
    5. contrast_alpha, brightness_beta - Linear contrast/brightness
    6. sharpening_strength - Unsharp masking
    7. threshold_value, threshold_method - Thresholding
    
    Args:
        image_path: Path to input image
        params: Dictionary of preprocessing parameters
        output_path: Optional path to save preprocessed image
        
    Returns:
        Preprocessed image as numpy array
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        # Load image
        img_path = Path(image_path)
        if not img_path.exists():
            raise PreprocessingError(f"Image not found: {image_path}")
        
        # Load with PIL first to handle various formats (including TIFF)
        pil_img = Image.open(img_path)
        
        # Convert to numpy array
        img = np.array(pil_img)
        
        # Normalize 16-bit to 8-bit if needed
        if img.dtype == np.uint16 or img.dtype == np.int16:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Ensure 8-bit
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Convert to grayscale if needed for some operations
        is_grayscale = len(img.shape) == 2
        if not is_grayscale and img.shape[2] in (3, 4):
            # RGB/RGBA image
            working_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            working_img = img.copy()
        
        # Apply preprocessing operations in order
        
        # 1. Denoising
        if "denoising_strength" in params and params["denoising_strength"] > 0:
            strength = params["denoising_strength"]
            # Convert strength (0-1) to h parameter (0-30)
            h = int(strength * 30)
            working_img = cv2.fastNlMeansDenoising(working_img, None, h=h, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Background subtraction (morphological opening)
        if "background_subtraction_radius" in params and params["background_subtraction_radius"] > 0:
            radius = int(params["background_subtraction_radius"])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            background = cv2.morphologyEx(working_img, cv2.MORPH_OPEN, kernel)
            # Subtract background safely
            working_img = cv2.subtract(working_img, background)
        
        # 3. CLAHE contrast enhancement
        if "clahe_clip_limit" in params:
            clip_limit = params.get("clahe_clip_limit", 2.0)
            grid_size = params.get("clahe_grid_size", 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            working_img = clahe.apply(working_img)
        
        # 4. Gamma correction
        if "gamma_correction" in params and params["gamma_correction"] != 1.0:
            gamma = params["gamma_correction"]
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            working_img = cv2.LUT(working_img, table)
        
        # 5. Linear contrast and brightness
        if "contrast_alpha" in params or "brightness_beta" in params:
            alpha = params.get("contrast_alpha", 1.0)
            beta = params.get("brightness_beta", 0)
            working_img = cv2.convertScaleAbs(working_img, alpha=alpha, beta=beta)
        
        # 6. Sharpening
        if "sharpening_strength" in params and params["sharpening_strength"] > 0:
            strength = params["sharpening_strength"]
            # Gaussian blur
            blurred = cv2.GaussianBlur(working_img, (0, 0), 3)
            # Unsharp mask
            working_img = cv2.addWeighted(working_img, 1.0 + strength, blurred, -strength, 0)
        
        # 7. Thresholding (applied last)
        if "threshold_method" in params:
            method = params["threshold_method"]
            if method == "otsu":
                _, working_img = cv2.threshold(working_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == "adaptive":
                working_img = cv2.adaptiveThreshold(
                    working_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif method == "binary" and "threshold_value" in params:
                _, working_img = cv2.threshold(working_img, params["threshold_value"], 255, cv2.THRESH_BINARY)
            elif method == "fixed" and "threshold_value" in params:
                _, working_img = cv2.threshold(working_img, params["threshold_value"], 255, cv2.THRESH_BINARY)
        
        # Convert back to RGB if original was color
        if not is_grayscale and len(img.shape) == 3:
            result_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2RGB)
        else:
            result_img = working_img
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with PIL to handle various formats
            pil_output = Image.fromarray(result_img)
            pil_output.save(output_path)
        
        return result_img
        
    except Exception as e:
        raise PreprocessingError(f"Preprocessing failed for {image_path}: {e}")


def preview_preprocessing(
    image: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Apply preprocessing to an in-memory image (for preview).
    
    Args:
        image: Input image as numpy array
        params: Dictionary of preprocessing parameters
        
    Returns:
        Preprocessed image
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    try:
        img = image.copy()
        
        # Ensure 8-bit
        if img.dtype == np.uint16 or img.dtype == np.int16:
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Same logic as apply_preprocessing but on in-memory array
        is_grayscale = len(img.shape) == 2
        if not is_grayscale and img.shape[2] in (3, 4):
            working_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            working_img = img.copy()
        
        # Apply same operations as apply_preprocessing
        # (abbreviated for brevity - same code as above)
        
        if "denoising_strength" in params and params["denoising_strength"] > 0:
            h = int(params["denoising_strength"] * 30)
            working_img = cv2.fastNlMeansDenoising(working_img, None, h=h, templateWindowSize=7, searchWindowSize=21)
        
        if "background_subtraction_radius" in params and params["background_subtraction_radius"] > 0:
            radius = int(params["background_subtraction_radius"])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            background = cv2.morphologyEx(working_img, cv2.MORPH_OPEN, kernel)
            working_img = cv2.subtract(working_img, background)
        
        if "clahe_clip_limit" in params:
            clip_limit = params.get("clahe_clip_limit", 2.0)
            grid_size = params.get("clahe_grid_size", 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            working_img = clahe.apply(working_img)
        
        if "gamma_correction" in params and params["gamma_correction"] != 1.0:
            gamma = params["gamma_correction"]
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            working_img = cv2.LUT(working_img, table)
        
        if "contrast_alpha" in params or "brightness_beta" in params:
            alpha = params.get("contrast_alpha", 1.0)
            beta = params.get("brightness_beta", 0)
            working_img = cv2.convertScaleAbs(working_img, alpha=alpha, beta=beta)
        
        if "sharpening_strength" in params and params["sharpening_strength"] > 0:
            strength = params["sharpening_strength"]
            blurred = cv2.GaussianBlur(working_img, (0, 0), 3)
            working_img = cv2.addWeighted(working_img, 1.0 + strength, blurred, -strength, 0)
        
        if "threshold_method" in params:
            method = params["threshold_method"]
            if method == "otsu":
                _, working_img = cv2.threshold(working_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == "adaptive":
                working_img = cv2.adaptiveThreshold(
                    working_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif method in ["binary", "fixed"] and "threshold_value" in params:
                _, working_img = cv2.threshold(working_img, params["threshold_value"], 255, cv2.THRESH_BINARY)
        
        if not is_grayscale and len(img.shape) == 3:
            result_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2RGB)
        else:
            result_img = working_img
        
        return result_img
        
    except Exception as e:
        raise PreprocessingError(f"Preview preprocessing failed: {e}")


