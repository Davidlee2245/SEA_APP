"""
Phase 2: Aligner Agent
- Corrects mechanical drift and chromatic aberration
- Uses SuperPoint + SuperGlue for feature detection/matching
- Applies Affine or TPS transformation using Kornia
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

try:
    from scipy.ndimage import fourier_shift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    fourier_shift = None
    logger.warning("scipy not available, Phase Cross Correlation will not work")

# FFT functions - use numpy if scipy not available
try:
    from scipy.fft import fft2, ifft2
except ImportError:
    # Fallback to numpy.fft (always available)
    from numpy.fft import fft2, ifft2

try:
    from skimage.registration import phase_cross_correlation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("skimage.registration not available, using manual PCC implementation")

from .base_agent import BaseAgent


class Aligner(BaseAgent):
    """
    Aligner agent for multi-channel image registration.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        self.feature_detector = config.get('feature_detector', 'superpoint')
        self.matcher = config.get('matcher', 'superglue')
        self.initial_transform = config.get('initial_transform', 'affine')
        self.fallback_transform = config.get('fallback_transform', 'tps')
        self.residual_threshold = config.get('residual_threshold', 2.0)
        self.max_features = config.get('max_features', 1024)
        self.match_threshold = config.get('match_threshold', 0.7)
        self.min_matches = config.get('min_matches', 4)  # Minimum matches needed for transform
        self.llm_qa_enabled = config.get('llm_qa_enabled', True)
        self.interpolation_mode = config.get('interpolation_mode', 'bicubic')  # 'bilinear', 'bicubic', 'nearest'
        
        # Initialize device
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self._superpoint_model = None
        self._superglue_model = None
    
    def validate_input(self, images: Dict[str, np.ndarray], anchor_channel: str) -> bool:
        """
        Validate input images and anchor channel.
        
        Args:
            images: Dict mapping channel names to image arrays
            anchor_channel: Name of the anchor channel
            
        Returns:
            bool: True if valid
        """
        if anchor_channel not in images:
            self.logger.error(f"Anchor channel '{anchor_channel}' not found in images")
            return False
        
        if len(images) < 2:
            self.logger.warning("Only one channel found, no registration needed")
            return False
        
        anchor_shape = images[anchor_channel].shape
        for ch_name, img in images.items():
            if img.shape != anchor_shape:
                self.logger.error(f"Channel {ch_name} shape {img.shape} != anchor shape {anchor_shape}")
                return False
        
        return True
    
    def load_superpoint(self):
        """Load feature detector (using LoFTR as modern alternative to SuperPoint)."""
        try:
            import kornia.feature as KF
            
            self.logger.info("Loading LoFTR model (SuperPoint alternative)...")
            # LoFTR is a modern transformer-based matcher that works well for fluorescence microscopy
            self._superpoint_model = KF.LoFTR(pretrained='indoor').to(self.device)
            self._superpoint_model.eval()
            return True
        except Exception as e:
            self.logger.error(f"Failed to load LoFTR: {e}")
            # Fallback to KeyNet + HardNet
            try:
                self.logger.info("Trying KeyNet + HardNet as fallback...")
                self._keynet_detector = KF.KeyNetDetector(pretrained=True, num_features=self.max_features).to(self.device)
                self._hardnet_descriptor = KF.HardNet(pretrained=True).to(self.device)
                self._keynet_detector.eval()
                self._hardnet_descriptor.eval()
                self._superpoint_model = 'keynet+hardnet'  # Flag to use alternative path
                return True
            except Exception as e2:
                self.logger.error(f"Failed to load KeyNet+HardNet fallback: {e2}")
                return False
    
    def load_superglue(self):
        """Load matcher (using DescriptorMatcher)."""
        try:
            import kornia.feature as KF
            
            self.logger.info("Loading DescriptorMatcher...")
            # Use DescriptorMatcher directly with mutual nearest neighbor strategy
            self._superglue_model = KF.DescriptorMatcher('smnn', self.match_threshold)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load matcher: {e}")
            return False
    
    def detect_features_superpoint(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and descriptors using LoFTR or KeyNet+HardNet.
        
        Args:
            image: Input image array (float32, [0, 1])
            
        Returns:
            Tuple of (keypoints, descriptors) tensors
        """
        import kornia.feature as KF
        
        if self._superpoint_model is None:
            if not self.load_superpoint():
                raise RuntimeError("Feature detector not available")
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            img_tensor = image
        
        # Ensure float32 and [0, 1] range
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / img_tensor.max()
        
        # Check if using KeyNet+HardNet fallback
        if self._superpoint_model == 'keynet+hardnet':
            with torch.no_grad():
                # Detect keypoints with KeyNet
                lafs, scores = self._keynet_detector(img_tensor)  # LAFs: Local Affine Frames
                
                # Extract keypoint coordinates from LAFs
                keypoints = lafs[:, :, :2, 2]  # Extract translation part [B, N, 2]
                
                # Compute descriptors with HardNet
                patches = KF.extract_patches_from_pyramid(img_tensor, lafs, 32)
                descriptors = self._hardnet_descriptor(patches)  # [B*N, 128]
                descriptors = descriptors.view(img_tensor.size(0), -1, 128).transpose(1, 2)  # [B, 128, N]
            
            return keypoints[0], descriptors[0]
        else:
            # Using LoFTR (detection will be done in matching phase)
            # For compatibility, return dummy values - LoFTR does joint detection+matching
            self.logger.warning("LoFTR requires image pairs for matching. Use detect_and_match_loftr() instead.")
            # Return empty tensors as placeholders
            return torch.empty(0, 2, device=self.device), torch.empty(256, 0, device=self.device)
    
    def detect_and_match_loftr(self, image1: np.ndarray, image2: np.ndarray) -> Optional[torch.Tensor]:
        """
        Detect and match features using LoFTR (joint detection+matching).
        
        Args:
            image1: First image (anchor)
            image2: Second image (to register)
            
        Returns:
            torch.Tensor: Matched keypoints [M, 2, 2] or None
        """
        if self._superpoint_model is None or self._superpoint_model == 'keynet+hardnet':
            return None
        
        import kornia.feature as KF
        import torch.nn.functional as F
        
        try:
            # Clear CUDA cache before processing
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Convert to tensors
            img1_tensor = torch.from_numpy(self.ensure_float32(image1)).unsqueeze(0).unsqueeze(0)
            img2_tensor = torch.from_numpy(self.ensure_float32(image2)).unsqueeze(0).unsqueeze(0)
            
            # Downsample large images to save memory (LoFTR works better on smaller images anyway)
            max_size = 1024
            h1, w1 = img1_tensor.shape[2:]
            if max(h1, w1) > max_size:
                scale = max_size / max(h1, w1)
                new_h, new_w = int(h1 * scale), int(w1 * scale)
                img1_tensor = F.interpolate(img1_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                img2_tensor = F.interpolate(img2_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                self.logger.info(f"Downsampled images from {h1}x{w1} to {new_h}x{new_w} for LoFTR")
            else:
                scale = 1.0
            
            # Move to device only when needed
            img1_tensor = img1_tensor.to(self.device)
            img2_tensor = img2_tensor.to(self.device)
            
            # Normalize to [0, 1]
            if img1_tensor.max() > 1.0:
                img1_tensor = img1_tensor / img1_tensor.max()
            if img2_tensor.max() > 1.0:
                img2_tensor = img2_tensor / img2_tensor.max()
            
            # Prepare input dict for LoFTR
            input_dict = {
                'image0': img1_tensor,
                'image1': img2_tensor
            }
            
            # Monitor memory usage
            if self.device == "cuda":
                mem_before = torch.cuda.memory_allocated() / 1e9
                self.logger.debug(f"LoFTR memory before: {mem_before:.2f} GB")
            
            with torch.no_grad():
                correspondences = self._superpoint_model(input_dict)
            
            # Check memory after inference
            if self.device == "cuda":
                mem_after = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                self.logger.debug(f"LoFTR memory after: {mem_after:.2f} GB (peak: {mem_peak:.2f} GB)")
                torch.cuda.reset_peak_memory_stats()  # Reset for next measurement
            
            # Extract matched keypoints
            mkpts0 = correspondences['keypoints0']  # [B, N, 2]
            mkpts1 = correspondences['keypoints1']  # [B, N, 2]
            
            # Clear intermediate tensors
            del img1_tensor, img2_tensor, input_dict
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            num_matches = mkpts0.shape[1]
            if num_matches < self.min_matches:
                self.logger.warning(
                    f"LoFTR found only {num_matches} matches (need {self.min_matches}). "
                    f"This may indicate: (1) images already well-aligned, "
                    f"(2) low contrast/poor SNR, (3) different structures, or "
                    f"(4) LoFTR model mismatch for fluorescence microscopy."
                )
                return None
            
            # Scale keypoints back to original resolution
            if scale != 1.0:
                mkpts0 = mkpts0 / scale
                mkpts1 = mkpts1 / scale
            
            # Stack into [M, 2, 2] format: [match_idx, (src/dst), (x/y)]
            matches = torch.stack([mkpts1[0], mkpts0[0]], dim=1)  # Note: mkpts1 is moving, mkpts0 is anchor
            
            return matches
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.error(f"CUDA OOM in LoFTR. Falling back to KeyNet+HardNet")
                # Clear cache and try fallback
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                # Load fallback
                self._superpoint_model = 'keynet+hardnet'
                self.load_superpoint()
                return None
            else:
                raise
    
    def match_features_superglue(self, desc1: torch.Tensor, desc2: torch.Tensor,
                                 kpts1: torch.Tensor, kpts2: torch.Tensor) -> torch.Tensor:
        """
        Match features using SuperGlue.
        
        Args:
            desc1, desc2: Descriptor tensors [256, N]
            kpts1, kpts2: Keypoint tensors [N, 2]
            
        Returns:
            torch.Tensor: Match indices [M, 2] or None if matching fails
        """
        if self._superglue_model is None:
            if not self.load_superglue():
                raise RuntimeError("SuperGlue model not available")
        
        # Prepare input for SuperGlue
        # SuperGlue expects specific format
        try:
            import kornia.feature as KF
            
            # Create image pair data structure
            img1_tensor = torch.zeros(1, 1, int(kpts1[:, 1].max()) + 1, int(kpts1[:, 0].max()) + 1).to(self.device)
            img2_tensor = torch.zeros(1, 1, int(kpts2[:, 1].max()) + 1, int(kpts2[:, 0].max()) + 1).to(self.device)
            
            # Format keypoints and descriptors
            laf1 = KF.laf_from_center_scale_ori(kpts1.unsqueeze(0), torch.ones(1, len(kpts1), 1, 1).to(self.device))
            laf2 = KF.laf_from_center_scale_ori(kpts2.unsqueeze(0), torch.ones(1, len(kpts2), 1, 1).to(self.device))
            
            desc1_norm = desc1.unsqueeze(0) / (desc1.norm(dim=0, keepdim=True) + 1e-8)
            desc2_norm = desc2.unsqueeze(0) / (desc2.norm(dim=0, keepdim=True) + 1e-8)
            
            with torch.no_grad():
                dists, idxs = self._superglue_model(
                    desc1_norm, desc2_norm,
                    laf1, laf2
                )
            
            # Extract matches
            matches = idxs[0]  # [N1, 2] where matches[i] = [idx1, idx2]
            valid_matches = matches[:, 1] >= 0
            
            if valid_matches.sum() == 0:
                return None
            
            matched_kpts1 = kpts1[matches[valid_matches, 0]]
            matched_kpts2 = kpts2[matches[valid_matches, 1]]
            
            return torch.stack([matched_kpts1, matched_kpts2], dim=1)  # [M, 2, 2]
            
        except Exception as e:
            self.logger.warning(f"SuperGlue matching failed: {e}. Using simple nearest neighbor.")
            # Fallback to simple matching
            return self.match_features_simple(desc1, desc2, kpts1, kpts2)
    
    def match_features_simple(self, desc1: torch.Tensor, desc2: torch.Tensor,
                              kpts1: torch.Tensor, kpts2: torch.Tensor) -> torch.Tensor:
        """
        Simple nearest neighbor matching as fallback.
        
        Args:
            desc1, desc2: Descriptor tensors
            kpts1, kpts2: Keypoint tensors
            
        Returns:
            torch.Tensor: Match pairs [M, 2, 2]
        """
        # Normalize descriptors
        desc1_norm = desc1 / (desc1.norm(dim=0, keepdim=True) + 1e-8)
        desc2_norm = desc2 / (desc2.norm(dim=0, keepdim=True) + 1e-8)
        
        # Compute pairwise distances
        dists = torch.cdist(desc1_norm.t(), desc2_norm.t())  # [N1, N2]
        
        # Find mutual nearest neighbors
        matches_1to2 = dists.argmin(dim=1)  # [N1]
        matches_2to1 = dists.argmin(dim=0)  # [N2]
        
        # Keep only mutual matches
        valid = []
        for i, j in enumerate(matches_1to2):
            if matches_2to1[j] == i and dists[i, j] < self.match_threshold:
                valid.append((i, j))
        
        if not valid:
            return None
        
        valid = torch.tensor(valid, device=self.device)
        matched_kpts1 = kpts1[valid[:, 0]]
        matched_kpts2 = kpts2[valid[:, 1]]
        
        return torch.stack([matched_kpts1, matched_kpts2], dim=1)
    
    def estimate_affine_transform(self, src_points: torch.Tensor, dst_points: torch.Tensor) -> torch.Tensor:
        """
        Estimate affine transformation matrix.
        
        Args:
            src_points: Source keypoints [N, 2]
            dst_points: Destination keypoints [N, 2]
            
        Returns:
            torch.Tensor: Affine transformation matrix [3, 3]
        """
        import kornia.geometry.transform as KGT
        
        # Kornia's affine estimation
        try:
            transform = KGT.get_affine_matrix2d(
                src_points.unsqueeze(0),
                dst_points.unsqueeze(0),
                mode='least_squares'
            )
            return transform[0]  # [3, 3]
        except Exception as e:
            self.logger.warning(f"Kornia affine estimation failed: {e}. Using OpenCV fallback.")
            # Fallback to OpenCV
            import cv2
            src_pts = src_points.cpu().numpy().astype(np.float32)
            dst_pts = dst_points.cpu().numpy().astype(np.float32)
            M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            M_homogeneous = np.vstack([M, [0, 0, 1]])
            return torch.from_numpy(M_homogeneous).float().to(self.device)
    
    def estimate_tps_transform(self, src_points: torch.Tensor, dst_points: torch.Tensor,
                              image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Estimate Thin-Plate Spline transformation.
        
        Args:
            src_points: Source keypoints [N, 2]
            dst_points: Destination keypoints [N, 2]
            image_shape: (height, width) of the image
            
        Returns:
            torch.Tensor: TPS transformation parameters
        """
        try:
            import kornia.geometry.transform as KGT
            
            # Kornia TPS estimation
            tps = KGT.get_tps_transform(
                src_points.unsqueeze(0),
                dst_points.unsqueeze(0)
            )
            return tps
        except Exception as e:
            self.logger.warning(f"Kornia TPS estimation failed: {e}")
            # Fallback: return identity
            return None
    
    def apply_transform_kornia(self, image: np.ndarray, transform: torch.Tensor,
                               transform_type: str = 'affine', 
                               interpolation_mode: str = 'bicubic') -> np.ndarray:
        """
        Apply transformation using Kornia (GPU-accelerated) with controllable interpolation.
        
        Args:
            image: Input image array (preserves original dtype range)
            transform: Transformation matrix/parameters
            transform_type: 'affine', 'translation', 'euclidean', or 'tps'
            interpolation_mode: 'bilinear', 'bicubic', or 'nearest' (default: 'bicubic' for sharpness)
            
        Returns:
            np.ndarray: Warped image (normalized to [0,1] range, float32)
        """
        import kornia.geometry.transform as KGT
        import torch.nn.functional as F
        
        # Convert to float32 [0, 1] for processing
        img_float = self.ensure_float32(image)
        img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Map interpolation mode to PyTorch mode
        mode_map = {
            'bilinear': 'bilinear',
            'bicubic': 'bicubic',
            'nearest': 'nearest'
        }
        interp_mode = mode_map.get(interpolation_mode, 'bicubic')
        
        # translation and euclidean use the same 3x3 homogeneous matrix format as affine
        affine_types = ('affine', 'translation', 'euclidean')
        
        with torch.no_grad():
            if transform_type in affine_types:
                theta_2x3 = transform[:2, :]  # [2, 3]
                
                h, w = img_tensor.shape[-2:]
                grid = F.affine_grid(theta_2x3.unsqueeze(0), size=(1, 1, h, w), align_corners=False)
                
                warped = F.grid_sample(
                    img_tensor,
                    grid,
                    mode=interp_mode,
                    padding_mode='zeros',
                    align_corners=False
                )
            elif transform_type == 'tps':
                warped = KGT.warp_image_tps(
                    img_tensor,
                    transform,
                    dsize=img_tensor.shape[-2:]
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Convert back to numpy and ensure valid range
        warped_np = warped[0, 0].cpu().numpy()
        warped_np = np.clip(warped_np, 0.0, 1.0)
        
        return warped_np
    
    def calculate_residual_error(self, src_points: torch.Tensor, dst_points: torch.Tensor,
                                 transform: torch.Tensor, transform_type: str = 'affine') -> float:
        """
        Calculate residual error after transformation.
        
        Args:
            src_points: Source keypoints
            dst_points: Destination keypoints
            transform: Transformation matrix
            transform_type: 'affine', 'translation', 'euclidean', or 'tps'
            
        Returns:
            float: Mean residual error in pixels
        """
        # Apply transform to source points
        # translation and euclidean use the same 3x3 matrix format as affine
        if transform_type in ('affine', 'translation', 'euclidean'):
            src_homogeneous = torch.cat([
                src_points,
                torch.ones(len(src_points), 1, device=self.device)
            ], dim=1)  # [N, 3]
            
            transformed = (transform @ src_homogeneous.t()).t()  # [N, 3]
            transformed_xy = transformed[:, :2] / (transformed[:, 2:3] + 1e-8)
        else:
            # For TPS, use simplified error (would need full TPS evaluation)
            transformed_xy = src_points
        
        # Calculate error
        errors = torch.norm(transformed_xy - dst_points, dim=1)
        mean_error = errors.mean().item()
        
        return mean_error
    
    def query_llm_visual_qa(self, overlay_image_path: Path) -> Dict[str, Any]:
        """
        Query LLM for visual QA of alignment.
        
        Args:
            overlay_image_path: Path to overlay preview image
            
        Returns:
            Dict with LLM assessment
        """
        if not self.llm_qa_enabled:
            return {'alignment_ok': True, 'reasoning': 'LLM QA disabled'}
        
        try:
            import os
            import base64
            from openai import OpenAI
            
            api_key = os.getenv(self.config.get('llm', {}).get('api_key_env', 'OPENAI_API_KEY'))
            if not api_key:
                self.logger.warning("LLM API key not found. Skipping visual QA.")
                return {'alignment_ok': True, 'reasoning': 'LLM not available'}
            
            client = OpenAI(api_key=api_key)
            
            # Encode image
            with open(overlay_image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = """You are a bio-image analysis expert. Examine this overlay of two fluorescence channels after registration. Assess if the alignment is correct.

Look for:
- Colocalized structures (exosomes) should overlap
- No systematic offset between channels
- Minimal distortion artifacts

Respond in JSON:
{
    "alignment_ok": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of any problems"],
    "reasoning": "brief explanation"
}"""
            
            import json
            
            # Vision models don't support response_format, so we'll parse JSON manually
            response = client.chat.completions.create(
                model=self.config.get('llm', {}).get('model', 'gpt-4o'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                temperature=self.config.get('llm', {}).get('temperature', 0.3)
            )
            
            # Parse JSON from response (vision models return text, not structured JSON)
            content = response.choices[0].message.content
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            self.logger.info(f"LLM Visual QA: {result}")
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM visual QA failed: {e}")
            return {'alignment_ok': True, 'reasoning': f'LLM error: {e}'}
    
    def register_channels_phase_cross_correlation(self, images: Dict[str, np.ndarray], anchor_channel: str, transform_type: str = 'translation') -> Dict[str, Any]:
        """
        Register channels using Phase Cross Correlation (PCC).
        PCC is good for rigid translations and can be more robust than feature-based methods.
        
        Args:
            images: Dict mapping channel names to image arrays
            anchor_channel: Name of the anchor channel
            transform_type: 'translation' (default) or 'euclidean' (translation + rotation)
            
        Returns:
            Dict containing registered images and transformation metadata
        """
        if not self.validate_input(images, anchor_channel):
            return {'error': 'Input validation failed'}
        
        anchor_image = self.ensure_float32(images[anchor_channel])
        registered_images = {anchor_channel: anchor_image}
        transformations = {}
        
        if not SCIPY_AVAILABLE:
            self.logger.error("scipy not available, cannot use Phase Cross Correlation")
            return {'error': 'scipy not available for Phase Cross Correlation'}
        
        self.logger.info(f"Using Phase Cross Correlation for registration (anchor: {anchor_channel}, transform: {transform_type})")
        
        for channel_name, image in images.items():
            if channel_name == anchor_channel:
                continue
            
            self.logger.info(f"Registering channel {channel_name} using Phase Cross Correlation")
            img_norm = self.ensure_float32(image)
            
            try:
                # Use skimage's phase_cross_correlation if available (more robust)
                if SKIMAGE_AVAILABLE:
                    shift, error, diffphase = phase_cross_correlation(
                        anchor_image, 
                        img_norm,
                        upsample_factor=10  # Sub-pixel accuracy
                    )
                    # shift is (row, col) = (y, x), convert to (x, y)
                    dx = float(shift[1])  # column shift
                    dy = float(shift[0])  # row shift
                    self.logger.info(f"PCC shift for {channel_name}: dx={dx:.3f}, dy={dy:.3f}, error={error:.6f}, phase_diff={diffphase:.6f}")
                else:
                    # Manual implementation using FFT
                    # Compute cross-correlation in frequency domain
                    fft_anchor = fft2(anchor_image)
                    fft_image = fft2(img_norm)
                    
                    # Cross-power spectrum
                    cross_power = fft_anchor * np.conj(fft_image)
                    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)
                    
                    # Inverse FFT to get correlation
                    correlation = np.real(ifft2(cross_power_norm))
                    
                    # Find peak (shift)
                    h, w = correlation.shape
                    center = (h // 2, w // 2)
                    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
                    
                    # Calculate shift (accounting for FFT wrapping)
                    dy = (peak[0] - center[0]) % h
                    if dy > h // 2:
                        dy -= h
                    dx = (peak[1] - center[1]) % w
                    if dx > w // 2:
                        dx -= w
                    
                    error = 1.0 - correlation[peak]  # Normalized error
                    diffphase = 0.0
                    self.logger.info(f"PCC shift for {channel_name}: dx={dx:.3f}, dy={dy:.3f}, error={error:.6f}")
                
                # Check if shift is significant
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude < 0.1:  # Very small shift, likely already aligned
                    self.logger.info(f"Shift for {channel_name} is very small ({magnitude:.3f} px), using identity transform")
                    registered_images[channel_name] = img_norm
                    transformations[channel_name] = {
                        'type': 'identity',
                        'error': float(error) if 'error' in locals() else 0.0,
                        'dx': 0.0,
                        'dy': 0.0,
                        'magnitude': 0.0
                    }
                    continue
                
                # Build transformation matrix based on transform_type
                if transform_type.lower() in ['translation', 'euclidean']:
                    # NOTE (PCC limitation):
                    # Phase Cross Correlation computes a sub-pixel SHIFT (dx, dy) from the
                    # Fourier phase spectrum.  It does NOT estimate a rotation angle.
                    # Therefore, even when transform_type='euclidean' is requested, only a
                    # pure translation matrix is built here.  The rotation component of an
                    # Euclidean transform is left as identity (0°).
                    #
                    # To add rotation support for PCC:
                    #   1. Estimate rotation via log-polar / scale-invariant PCC first.
                    #   2. Build M = T(cx,cy) · R(theta) · T(-cx,-cy) · T(dx,dy).
                    #   3. Apply with warpAffine / Kornia instead of fourier_shift.
                    #
                    # Until that is implemented, 'euclidean' is silently treated as
                    # 'translation' with this PCC method.
                    if transform_type.lower() == 'euclidean':
                        self.logger.warning(
                            f"[PCC] transform_type='euclidean' requested for {channel_name} "
                            "but PCC currently supports translation-only. "
                            "Rotation will be 0°. Use manual_diagonal or feature-based methods for rotation."
                        )
                    transform_matrix = np.array([
                        [1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                    transform_type_used = 'translation' if transform_type.lower() == 'translation' else 'euclidean'
                    
                    # Apply translation using scipy
                    if SCIPY_AVAILABLE:
                        # Use fourier_shift for sub-pixel accuracy
                        shift_array = np.array([dy, dx])  # (row, col) = (y, x)
                        shifted = fourier_shift(fft2(img_norm), shift_array)
                        registered_img = np.real(ifft2(shifted))
                        # Normalize back to [0, 1]
                        registered_img = np.clip(registered_img, 0, 1)
                    else:
                        # Fallback: use kornia for translation
                        transform_tensor = torch.tensor(transform_matrix, dtype=torch.float32, device=self.device)
                        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
                        registered_tensor = self.apply_transform_kornia(
                            img_norm,
                            transform_tensor,
                            'affine',
                            self.interpolation_mode
                        )
                        registered_img = registered_tensor
                    
                    registered_images[channel_name] = registered_img
                    transformations[channel_name] = {
                        'type': transform_type_used,
                        'transform': transform_matrix.tolist(),
                        'error': float(error) if 'error' in locals() else 0.0,
                        'dx': float(dx),
                        'dy': float(dy),
                        'magnitude': float(magnitude),
                        'num_matches': 1,  # PCC doesn't use matches, but we have 1 "match" (the shift)
                        'note': f'Phase Cross Correlation: error={error:.6f}, phase_diff={diffphase:.6f}' if 'diffphase' in locals() else f'Phase Cross Correlation: error={error:.6f}'
                    }
                    
                    self.logger.info(f"✓ Registered {channel_name} using PCC: shift=({dx:.3f}, {dy:.3f}), magnitude={magnitude:.3f} px")
                else:
                    self.logger.warning(f"Transform type '{transform_type}' not supported for PCC, using translation")
                    # Fall back to translation
                    transform_matrix = np.array([
                        [1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                    if SCIPY_AVAILABLE:
                        shift_array = np.array([dy, dx])
                        shifted = fourier_shift(fft2(img_norm), shift_array)
                        registered_img = np.real(ifft2(shifted))
                        registered_img = np.clip(registered_img, 0, 1)
                    else:
                        transform_tensor = torch.tensor(transform_matrix, dtype=torch.float32, device=self.device)
                        registered_tensor = self.apply_transform_kornia(
                            img_norm,
                            transform_tensor,
                            'affine',
                            self.interpolation_mode
                        )
                        registered_img = registered_tensor
                    
                    registered_images[channel_name] = registered_img
                    transformations[channel_name] = {
                        'type': 'translation',
                        'transform': transform_matrix.tolist(),
                        'error': float(error) if 'error' in locals() else 0.0,
                        'dx': float(dx),
                        'dy': float(dy),
                        'magnitude': float(magnitude),
                        'num_matches': 1
                    }
                    
            except Exception as e:
                self.logger.error(f"PCC failed for {channel_name}: {e}")
                registered_images[channel_name] = img_norm
                transformations[channel_name] = {
                    'type': 'identity',
                    'error': float('inf'),
                    'dx': 0.0,
                    'dy': 0.0,
                    'magnitude': 0.0,
                    'note': f'PCC failed: {str(e)}'
                }
        
        return {
            'registered_images': registered_images,
            'transformations': transformations
        }
    
    def register_channels(self, images: Dict[str, np.ndarray], anchor_channel: str, method: str = 'feature_based', transform_type: str = 'affine') -> Dict[str, Any]:
        """
        Register all channels to the anchor channel.
        
        Args:
            images: Dict mapping channel names to image arrays
            anchor_channel: Name of the anchor channel
            method: 'feature_based' (default) or 'phase_cross_correlation'
            transform_type: 'affine', 'translation', 'euclidean', 'tps' (for feature-based) or 'translation'/'euclidean' (for PCC)
            
        Returns:
            Dict containing registered images and transformation metadata
        """
        # Route to appropriate method
        if method.lower() in ['phase_cross_correlation', 'pcc', 'phase cross correlation']:
            return self.register_channels_phase_cross_correlation(images, anchor_channel, transform_type)
        
        # Feature-based method (original implementation)
        if not self.validate_input(images, anchor_channel):
            return {'error': 'Input validation failed'}
        
        anchor_image = self.ensure_float32(images[anchor_channel])
        registered_images = {anchor_channel: anchor_image}
        transformations = {}
        
        # Initialize feature detector if not already loaded
        if self._superpoint_model is None:
            self.load_superpoint()
        
        # Check if using LoFTR (joint detection+matching)
        use_loftr = (self._superpoint_model is not None and 
                     self._superpoint_model != 'keynet+hardnet')
        
        # For KeyNet+HardNet: detect features in anchor
        anchor_kpts, anchor_desc = None, None
        if not use_loftr:
            self.logger.info(f"Detecting features in anchor channel: {anchor_channel}")
            anchor_kpts, anchor_desc = self.detect_features_superpoint(anchor_image)
        else:
            self.logger.info(f"Using LoFTR for registration (anchor: {anchor_channel})")
        
        for channel_name, image in images.items():
            if channel_name == anchor_channel:
                continue
            
            self.logger.info(f"Registering channel: {channel_name}")
            img_norm = self.ensure_float32(image)
            
            # Match features using appropriate method
            if use_loftr:
                # LoFTR: joint detection and matching
                self.logger.info(f"Using LoFTR for {channel_name}")
                matches = self.detect_and_match_loftr(anchor_image, img_norm)
            else:
                # KeyNet+HardNet: separate detection and matching
                kpts, desc = self.detect_features_superpoint(img_norm)
                matches = self.match_features_superglue(anchor_desc, desc, anchor_kpts, kpts)
            
            if matches is None or len(matches) < self.min_matches:
                num_found = len(matches) if matches is not None else 0
                self.logger.warning(
                    f"Insufficient matches for {channel_name}: found {num_found}, need {self.min_matches}. "
                    f"Using identity transform (no alignment). "
                    f"This may be OK if channels are already aligned or have different structures."
                )
                registered_images[channel_name] = img_norm
                transformations[channel_name] = {'type': 'identity', 'error': float('inf')}
                continue
            
            # Extract matched points
            src_points = matches[:, 0, :]  # Points in moving image
            dst_points = matches[:, 1, :]  # Points in anchor image
            
            # Try affine first
            transform_type = self.initial_transform
            transform = self.estimate_affine_transform(src_points, dst_points)
            
            # Calculate residual
            residual = self.calculate_residual_error(src_points, dst_points, transform, 'affine')
            self.logger.info(f"Affine residual for {channel_name}: {residual:.2f} pixels")
            
            # Switch to TPS if residual is high
            if residual > self.residual_threshold and self.fallback_transform == 'tps':
                self.logger.info(f"Switching to TPS for {channel_name} (residual: {residual:.2f})")
                tps_params = self.estimate_tps_transform(src_points, dst_points, img_norm.shape)
                if tps_params is not None:
                    transform = tps_params
                    transform_type = 'tps'
                    residual = self.calculate_residual_error(src_points, dst_points, transform, 'tps')
            
            # Apply transformation with specified interpolation mode
            warped = self.apply_transform_kornia(
                img_norm, 
                transform, 
                transform_type,
                interpolation_mode=self.interpolation_mode
            )
            registered_images[channel_name] = warped
            transformations[channel_name] = {
                'type': transform_type,
                'transform': transform.cpu().numpy().tolist(),
                'residual_error': residual,
                'num_matches': len(matches)
            }
        
        return {
            'registered_images': registered_images,
            'transformations': transformations,
            'anchor_channel': anchor_channel
        }
    
    def process(self, images: Dict[str, np.ndarray], anchor_channel: str,
                sample_name: str = "unknown") -> Dict[str, Any]:
        """
        Main processing function for alignment.
        
        Args:
            images: Dict mapping channel names to image arrays
            anchor_channel: Name of the anchor channel
            sample_name: Name of the sample (for logging)
            
        Returns:
            Dict containing registered images and metadata
        """
        self.logger.info(f"Aligning channels for sample: {sample_name}")
        
        result = self.register_channels(images, anchor_channel)
        
        if 'error' in result:
            self._output = result
            return result
        
        # Optional: Create overlay for visual QA
        if self.llm_qa_enabled and len(result['registered_images']) >= 2:
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path
                
                # Create overlay preview
                overlay_path = Path(self.config.get('paths', {}).get('output_root', 'data/output')) / \
                              f"{sample_name}_alignment_overlay.png"
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create RGB overlay (first two channels)
                channels = list(result['registered_images'].keys())
                if len(channels) >= 2:
                    ch1 = result['registered_images'][channels[0]]
                    ch2 = result['registered_images'][channels[1]]
                    
                    overlay = np.zeros((*ch1.shape, 3))
                    overlay[:, :, 0] = ch1 / ch1.max() if ch1.max() > 0 else ch1
                    overlay[:, :, 1] = ch2 / ch2.max() if ch2.max() > 0 else ch2
                    
                    plt.imsave(overlay_path, np.clip(overlay, 0, 1))
                    
                    # Query LLM
                    llm_qa = self.query_llm_visual_qa(overlay_path)
                    result['llm_qa'] = llm_qa
            except Exception as e:
                self.logger.warning(f"Visual QA failed: {e}")
        
        self._output = result
        return result

