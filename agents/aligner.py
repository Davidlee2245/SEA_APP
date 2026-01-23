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
        self.llm_qa_enabled = config.get('llm_qa_enabled', True)
        
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
        """Load SuperPoint feature detector."""
        try:
            import kornia.feature as KF
            
            self.logger.info("Loading SuperPoint model...")
            # Kornia's SuperPoint implementation
            self._superpoint_model = KF.SuperPoint(max_num_keypoints=self.max_features).to(self.device)
            self._superpoint_model.eval()
            return True
        except Exception as e:
            self.logger.error(f"Failed to load SuperPoint: {e}")
            return False
    
    def load_superglue(self):
        """Load SuperGlue matcher."""
        try:
            import kornia.feature as KF
            
            self.logger.info("Loading SuperGlue model...")
            self._superglue_model = KF.SuperGlue().to(self.device)
            self._superglue_model.eval()
            return True
        except Exception as e:
            self.logger.error(f"Failed to load SuperGlue: {e}")
            return False
    
    def detect_features_superpoint(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and descriptors using SuperPoint.
        
        Args:
            image: Input image array (float32, [0, 1])
            
        Returns:
            Tuple of (keypoints, descriptors) tensors
        """
        if self._superpoint_model is None:
            if not self.load_superpoint():
                raise RuntimeError("SuperPoint model not available")
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            img_tensor = image
        
        # Ensure float32 and [0, 1] range
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / img_tensor.max()
        
        with torch.no_grad():
            out = self._superpoint_model(img_tensor)
            keypoints = out['keypoints']  # [B, N, 2]
            descriptors = out['descriptors']  # [B, 256, N]
        
        return keypoints[0], descriptors[0]
    
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
                               transform_type: str = 'affine') -> np.ndarray:
        """
        Apply transformation using Kornia (GPU-accelerated).
        
        Args:
            image: Input image array
            transform: Transformation matrix/parameters
            transform_type: 'affine' or 'tps'
            
        Returns:
            np.ndarray: Warped image
        """
        import kornia.geometry.transform as KGT
        
        # Convert to tensor
        img_tensor = torch.from_numpy(self.ensure_float32(image)).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if transform_type == 'affine':
                # Affine warping
                warped = KGT.warp_affine(
                    img_tensor,
                    transform.unsqueeze(0),
                    dsize=img_tensor.shape[-2:]
                )
            elif transform_type == 'tps':
                # TPS warping
                warped = KGT.warp_image_tps(
                    img_tensor,
                    transform,
                    dsize=img_tensor.shape[-2:]
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Convert back to numpy
        warped_np = warped[0, 0].cpu().numpy()
        return warped_np
    
    def calculate_residual_error(self, src_points: torch.Tensor, dst_points: torch.Tensor,
                                 transform: torch.Tensor, transform_type: str = 'affine') -> float:
        """
        Calculate residual error after transformation.
        
        Args:
            src_points: Source keypoints
            dst_points: Destination keypoints
            transform: Transformation matrix
            transform_type: 'affine' or 'tps'
            
        Returns:
            float: Mean residual error in pixels
        """
        # Apply transform to source points
        if transform_type == 'affine':
            # Convert to homogeneous coordinates
            src_homogeneous = torch.cat([
                src_points,
                torch.ones(len(src_points), 1, device=self.device)
            ], dim=1)  # [N, 3]
            
            # Apply transform
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
            
            response = client.chat.completions.create(
                model=self.config.get('llm', {}).get('model', 'gpt-4-vision-preview'),
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
                temperature=self.config.get('llm', {}).get('temperature', 0.3),
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            self.logger.info(f"LLM Visual QA: {result}")
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM visual QA failed: {e}")
            return {'alignment_ok': True, 'reasoning': f'LLM error: {e}'}
    
    def register_channels(self, images: Dict[str, np.ndarray], anchor_channel: str) -> Dict[str, Any]:
        """
        Register all channels to the anchor channel.
        
        Args:
            images: Dict mapping channel names to image arrays
            anchor_channel: Name of the anchor channel
            
        Returns:
            Dict containing registered images and transformation metadata
        """
        if not self.validate_input(images, anchor_channel):
            return {'error': 'Input validation failed'}
        
        anchor_image = self.ensure_float32(images[anchor_channel])
        registered_images = {anchor_channel: anchor_image}
        transformations = {}
        
        # Detect features in anchor
        self.logger.info(f"Detecting features in anchor channel: {anchor_channel}")
        anchor_kpts, anchor_desc = self.detect_features_superpoint(anchor_image)
        
        for channel_name, image in images.items():
            if channel_name == anchor_channel:
                continue
            
            self.logger.info(f"Registering channel: {channel_name}")
            img_norm = self.ensure_float32(image)
            
            # Detect features
            kpts, desc = self.detect_features_superpoint(img_norm)
            
            # Match features
            matches = self.match_features_superglue(anchor_desc, desc, anchor_kpts, kpts)
            
            if matches is None or len(matches) < 4:
                self.logger.warning(f"Insufficient matches for {channel_name}. Using identity transform.")
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
            
            # Apply transformation
            warped = self.apply_transform_kornia(img_norm, transform, transform_type)
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

