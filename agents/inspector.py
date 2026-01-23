"""
Phase 1: Inspector Agent
- Diagnoses image quality
- Performs denoising if needed
- Detects anchor channel for registration
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tifffile
from loguru import logger

from .base_agent import BaseAgent


class Inspector(BaseAgent):
    """
    Inspector agent for image quality assessment and preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        self.snr_threshold = config.get('snr_threshold', 10.0)
        self.denoise_threshold = config.get('denoise_threshold', 5.0)
        self.denoise_method = config.get('denoise_method', 'n2v')
        self.anchor_method = config.get('anchor_channel_method', 'sharpness')
        self._denoise_model = None
        
    def validate_input(self, image_paths: List[Path]) -> bool:
        """
        Validate that all image paths exist and are TIFF files.
        
        Args:
            image_paths: List of paths to channel images
            
        Returns:
            bool: True if all valid
        """
        if not image_paths:
            self.logger.error("No image paths provided")
            return False
        
        for path in image_paths:
            if not path.exists():
                self.logger.error(f"Image not found: {path}")
                return False
            if path.suffix.lower() not in ['.tif', '.tiff']:
                self.logger.warning(f"Non-TIFF file: {path}")
        
        return True
    
    def calculate_snr(self, image: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio using background estimation.
        
        Args:
            image: Input image array
            
        Returns:
            float: SNR value
        """
        # Normalize to float32 [0, 1]
        img_norm = self.ensure_float32(image)
        
        # Estimate background as lower percentile
        background = np.percentile(img_norm, 10)
        
        # Estimate signal as upper percentile
        signal = np.percentile(img_norm, 90)
        
        # Estimate noise as std of background region
        noise_mask = img_norm < np.percentile(img_norm, 20)
        noise = np.std(img_norm[noise_mask]) if np.any(noise_mask) else np.std(img_norm)
        
        if noise == 0:
            return float('inf')
        
        snr = (signal - background) / noise
        return float(snr)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image array
            
        Returns:
            float: Sharpness metric (higher = sharper)
        """
        from scipy import ndimage
        
        img_norm = self.ensure_float32(image)
        
        # Apply Laplacian filter (laplace is the Laplacian operator in scipy.ndimage)
        laplacian = ndimage.laplace(img_norm)
        
        # Variance of Laplacian is a measure of sharpness
        sharpness = np.var(laplacian)
        return float(sharpness)
    
    def get_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from TIFF file.
        
        Args:
            image_path: Path to TIFF file
            
        Returns:
            Dict containing metadata
        """
        try:
            with tifffile.TiffFile(str(image_path)) as tif:
                metadata = {
                    'shape': tif.pages[0].shape,
                    'dtype': str(tif.pages[0].dtype),
                    'bits_per_sample': tif.pages[0].bitspersample,
                }
                
                # Try to extract additional metadata
                if hasattr(tif.pages[0], 'tags'):
                    tags = {}
                    for tag in tif.pages[0].tags.values():
                        tags[tag.name] = tag.value
                    metadata['tags'] = tags
                
                return metadata
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from {image_path}: {e}")
            return {}
    
    def denoise_n2v(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise image using Noise2Void.
        
        Args:
            image: Input noisy image
            
        Returns:
            np.ndarray: Denoised image
        """
        try:
            from n2v.models import N2V
            import torch
            
            self.logger.info("Initializing Noise2Void model...")
            
            # Initialize model (in production, load pre-trained weights)
            # For now, use a simple self-supervised approach
            img_norm = self.ensure_float32(image)
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Note: In production, you would load a pre-trained N2V model
            # For now, return original image with a warning
            self.logger.warning("N2V model not fully implemented. Returning original image.")
            self.logger.info("To use N2V, train a model first or load pre-trained weights.")
            
            return image
            
        except ImportError:
            self.logger.error("Noise2Void not available. Install with: pip install n2v")
            return image
        except Exception as e:
            self.logger.error(f"N2V denoising failed: {e}")
            return image
    
    def denoise_care(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise image using CARE.
        
        Args:
            image: Input noisy image
            
        Returns:
            np.ndarray: Denoised image
        """
        try:
            from csbdeep.models import CARE
            import torch
            
            self.logger.info("Initializing CARE model...")
            
            # Note: CARE requires a pre-trained model
            # For now, return original image with a warning
            self.logger.warning("CARE model not fully implemented. Returning original image.")
            self.logger.info("To use CARE, train a model first or load pre-trained weights.")
            
            return image
            
        except ImportError:
            self.logger.error("CARE not available. Install with: pip install csbdeep")
            return image
        except Exception as e:
            self.logger.error(f"CARE denoising failed: {e}")
            return image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising based on configured method.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Denoised image
        """
        if self.denoise_method == 'n2v':
            return self.denoise_n2v(image)
        elif self.denoise_method == 'care':
            return self.denoise_care(image)
        else:
            self.logger.warning(f"Unknown denoise method: {self.denoise_method}")
            return image
    
    def detect_anchor_channel(self, images: Dict[str, np.ndarray]) -> str:
        """
        Detect the sharpest channel to use as anchor for registration.
        
        Args:
            images: Dict mapping channel names to image arrays
            
        Returns:
            str: Name of the anchor channel
        """
        if self.anchor_method == 'sharpness':
            sharpness_scores = {}
            for channel_name, image in images.items():
                sharpness = self.calculate_sharpness(image)
                sharpness_scores[channel_name] = sharpness
                self.logger.debug(f"Channel {channel_name} sharpness: {sharpness:.4f}")
            
            anchor = max(sharpness_scores, key=sharpness_scores.get)
            self.logger.info(f"Selected anchor channel: {anchor} (sharpness: {sharpness_scores[anchor]:.4f})")
            return anchor
        else:
            # Default: use first channel
            anchor = list(images.keys())[0]
            self.logger.info(f"Using first channel as anchor: {anchor}")
            return anchor
    
    def query_llm_qa(self, metadata_summary: str) -> Dict[str, Any]:
        """
        Query LLM for quality assessment decision.
        
        Args:
            metadata_summary: Text summary of image metadata and statistics
            
        Returns:
            Dict with LLM decision and reasoning
        """
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv(self.config.get('llm', {}).get('api_key_env', 'OPENAI_API_KEY'))
            if not api_key:
                self.logger.warning("LLM API key not found. Skipping LLM QA.")
                return {'denoise_needed': False, 'reasoning': 'LLM not available'}
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""You are a bio-image analysis expert. Analyze the following image quality metrics and decide if denoising is needed or if the image is too dark to use.

{metadata_summary}

Respond in JSON format:
{{
    "denoise_needed": true/false,
    "too_dark": true/false,
    "reasoning": "brief explanation"
}}"""
            
            response = client.chat.completions.create(
                model=self.config.get('llm', {}).get('analysis_model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('llm', {}).get('temperature', 0.3),
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            self.logger.info(f"LLM QA Decision: {result}")
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM QA failed: {e}. Using rule-based decision.")
            return {'denoise_needed': False, 'reasoning': f'LLM error: {e}'}
    
    def process(self, sample_dir: Path) -> Dict[str, Any]:
        """
        Process a sample directory: load images, assess quality, denoise if needed.
        
        Args:
            sample_dir: Path to sample directory containing channel TIFF files
            
        Returns:
            Dict containing processed images, metadata, and anchor channel
        """
        self.logger.info(f"Processing sample: {sample_dir.name}")
        
        # Find all TIFF files in sample directory
        image_paths = sorted(sample_dir.glob("*.tif*"))
        if not image_paths:
            self.logger.error(f"No TIFF files found in {sample_dir}")
            return {'error': 'No images found'}
        
        if not self.validate_input(image_paths):
            return {'error': 'Input validation failed'}
        
        # Load images
        images = {}
        metadata = {}
        snr_values = {}
        processed_images = {}
        
        for img_path in image_paths:
            channel_name = img_path.stem
            self.logger.info(f"Loading channel: {channel_name}")
            
            # Load image
            image = self.load_image(img_path)
            images[channel_name] = image
            
            # Extract metadata
            metadata[channel_name] = self.get_image_metadata(img_path)
            
            # Calculate SNR
            snr = self.calculate_snr(image)
            snr_values[channel_name] = snr
            self.logger.info(f"Channel {channel_name} SNR: {snr:.2f}")
        
        # Prepare metadata summary for LLM
        metadata_summary = f"Sample: {sample_dir.name}\n"
        for ch_name, ch_metadata in metadata.items():
            metadata_summary += f"\nChannel {ch_name}:\n"
            metadata_summary += f"  Shape: {ch_metadata.get('shape', 'N/A')}\n"
            metadata_summary += f"  Dtype: {ch_metadata.get('dtype', 'N/A')}\n"
            metadata_summary += f"  SNR: {snr_values[ch_name]:.2f}\n"
        
        # Query LLM for denoising decision
        llm_decision = self.query_llm_qa(metadata_summary)
        
        # Apply denoising if needed
        for channel_name, image in images.items():
            snr = snr_values[channel_name]
            
            # Decision logic: LLM recommendation OR rule-based fallback
            should_denoise = (
                llm_decision.get('denoise_needed', False) or
                snr < self.denoise_threshold
            )
            
            if should_denoise:
                self.logger.info(f"Denoising channel {channel_name} (SNR: {snr:.2f})")
                denoised = self.denoise_image(image)
                processed_images[channel_name] = denoised
            else:
                processed_images[channel_name] = image
        
        # Detect anchor channel
        anchor_channel = self.detect_anchor_channel(processed_images)
        
        # Prepare output
        self._output = {
            'sample_name': sample_dir.name,
            'images': processed_images,
            'metadata': metadata,
            'snr_values': snr_values,
            'anchor_channel': anchor_channel,
            'llm_decision': llm_decision,
            'denoised_channels': [ch for ch in images.keys() 
                                  if processed_images[ch] is not images[ch]]
        }
        
        self.logger.info(f"Inspection complete for {sample_dir.name}")
        return self._output

