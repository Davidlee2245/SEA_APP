"""
Image utility functions.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from PIL import Image


def save_image(image: np.ndarray, output_path: Path, 
               bit_depth: int = 8, normalize: bool = True, format: str = 'png') -> bool:
    """
    Save image as PNG (default) or TIFF.
    
    Args:
        image: Image array
        output_path: Output file path
        bit_depth: Target bit depth (8, 16, or 32)
        normalize: Whether to normalize to [0, 1] before conversion
        format: Output format ('png' or 'tif')
        
    Returns:
        bool: True if successful
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize if needed
        if normalize:
            img_norm = image.astype(np.float32)
            if img_norm.max() > 1.0:
                img_norm = img_norm / img_norm.max()
        else:
            img_norm = image.astype(np.float32)
        
        # Convert to target bit depth
        if bit_depth == 8:
            img_out = (img_norm * 255).astype(np.uint8)
        elif bit_depth == 16:
            img_out = (img_norm * 65535).astype(np.uint16)
        elif bit_depth == 32:
            img_out = img_norm.astype(np.float32)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        # Save based on format
        if format.lower() == 'png':
            # PNG: Use PIL for better compatibility
            # Convert to 8-bit for PNG (PNG supports 16-bit but browsers don't)
            if img_out.dtype == np.uint16:
                # Normalize 16-bit to 8-bit for web compatibility
                img_8bit = (img_out / 256).astype(np.uint8)
            elif img_out.dtype == np.float32:
                img_8bit = (np.clip(img_out, 0, 1) * 255).astype(np.uint8)
            else:
                img_8bit = img_out
            
            # Handle grayscale vs RGB
            if len(img_8bit.shape) == 2:
                pil_img = Image.fromarray(img_8bit, mode='L')
            elif len(img_8bit.shape) == 3:
                pil_img = Image.fromarray(img_8bit, mode='RGB')
            else:
                raise ValueError(f"Unsupported image shape: {img_8bit.shape}")
            
            pil_img.save(str(output_path), 'PNG', optimize=True)
            logger.debug(f"Saved PNG image: {output_path}")
        else:
            # TIFF: Use tifffile for preserving bit depth
            tifffile.imwrite(str(output_path), img_out)
            logger.debug(f"Saved TIFF image: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save image {output_path}: {e}")
        return False


def create_overlay(images: dict, output_path: Path, 
                   channel_colors: Optional[dict] = None) -> bool:
    """
    Create RGB overlay from multiple channels.
    
    Args:
        images: Dict mapping channel names to image arrays
        output_path: Output file path
        channel_colors: Dict mapping channel names to RGB tuples (0-1)
        
    Returns:
        bool: True if successful
    """
    try:
        import matplotlib.pyplot as plt
        
        if not images:
            logger.error("No images provided for overlay")
            return False
        
        # Default colors: red, green, blue, cyan, magenta, yellow
        default_colors = {
            'ch1': (1.0, 0.0, 0.0),  # Red
            'ch2': (0.0, 1.0, 0.0),  # Green
            'ch3': (0.0, 0.0, 1.0),  # Blue
            'ch4': (0.0, 1.0, 1.0),  # Cyan
            'ch5': (1.0, 0.0, 1.0),  # Magenta
            'ch6': (1.0, 1.0, 0.0),  # Yellow
        }
        
        if channel_colors is None:
            channel_colors = default_colors
        
        # Get first image shape
        first_img = list(images.values())[0]
        overlay = np.zeros((*first_img.shape, 3), dtype=np.float32)
        
        # Normalize and combine channels
        for i, (channel_name, image) in enumerate(images.items()):
            img_norm = image.astype(np.float32)
            if img_norm.max() > 0:
                img_norm = img_norm / img_norm.max()
            
            color = channel_colors.get(channel_name, default_colors.get(f'ch{i+1}', (1.0, 1.0, 1.0)))
            
            for c in range(3):
                overlay[:, :, c] += img_norm * color[c]
        
        # Normalize overlay
        overlay = np.clip(overlay, 0, 1)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_path, overlay)
        logger.debug(f"Saved overlay: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create overlay {output_path}: {e}")
        return False

