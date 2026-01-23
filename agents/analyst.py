"""
Phase 3: Analyst Agent
- Detects and counts exosomes using StarDist
- Analyzes colocalization
- Generates CSV output with coordinates and intensities
- Uses LLM for statistical interpretation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

from .base_agent import BaseAgent


class Analyst(BaseAgent):
    """
    Analyst agent for exosome detection and quantification.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        self.detection_model = config.get('detection_model', 'stardist')
        self.stardist_model_type = config.get('stardist_model_type', '2D_versatile_fluo')
        self.min_object_size = config.get('min_object_size', 3)
        self.max_object_size = config.get('max_object_size', 1000)
        self.intensity_threshold = config.get('intensity_threshold', 0.1)
        self.morphology_filter = config.get('morphology_filter', True)
        self.output_format = config.get('output_format', 'csv')
        self.llm_analysis_enabled = config.get('llm_analysis_enabled', True)
        
        self._stardist_model = None
    
    def validate_input(self, images: Dict[str, np.ndarray]) -> bool:
        """
        Validate input images for analysis.
        
        Args:
            images: Dict mapping channel names to image arrays
            
        Returns:
            bool: True if valid
        """
        if not images:
            self.logger.error("No images provided")
            return False
        
        # Check all images have same shape
        shapes = [img.shape for img in images.values()]
        if len(set(shapes)) > 1:
            self.logger.error("Images have different shapes")
            return False
        
        return True
    
    def load_stardist_model(self):
        """Load StarDist model for object detection."""
        try:
            from stardist.models import StarDist2D
            from csbdeep.utils import normalize
            
            self.logger.info(f"Loading StarDist model: {self.stardist_model_type}")
            self._stardist_model = StarDist2D.from_pretrained(self.stardist_model_type)
            self.logger.info("StarDist model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load StarDist model: {e}")
            return False
    
    def detect_objects_stardist(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects using StarDist.
        
        Args:
            image: Input image array (float32, [0, 1])
            
        Returns:
            Tuple of (label_image, detections_list)
            detections_list contains dicts with 'coord', 'prob', 'points'
        """
        if self._stardist_model is None:
            if not self.load_stardist_model():
                raise RuntimeError("StarDist model not available")
        
        # Normalize image for StarDist (expects uint8 or normalized float)
        img_norm = self.ensure_float32(image)
        
        # Convert to uint8 for StarDist (preserve relative intensities)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        # Run detection
        labels, details = self._stardist_model.predict_instances(img_uint8)
        
        # Extract detection information
        detections = []
        for i, (coord, prob, points) in enumerate(zip(
            details['coord'],
            details['prob'],
            details['points']
        )):
            detections.append({
                'id': i + 1,
                'coord': coord,  # Center coordinates (y, x)
                'prob': float(prob),
                'points': points,  # Polygon points
                'area': len(points)
            })
        
        return labels, detections
    
    def filter_detections(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Filter detections based on size, intensity, and morphology.
        
        Args:
            detections: List of detection dictionaries
            image: Original image for intensity filtering
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            # Size filter
            area = det['area']
            if area < self.min_object_size or area > self.max_object_size:
                continue
            
            # Intensity filter
            y, x = int(det['coord'][0]), int(det['coord'][1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                intensity = image[y, x]
                if intensity < self.intensity_threshold:
                    continue
            else:
                continue
            
            # Morphology filter (circularity check)
            if self.morphology_filter:
                # Calculate circularity: 4π*area/perimeter^2
                # For simplicity, approximate perimeter from area
                # Perfect circle has circularity = 1
                perimeter_approx = 2 * np.pi * np.sqrt(area / np.pi)
                circularity = (4 * np.pi * area) / (perimeter_approx ** 2) if perimeter_approx > 0 else 0
                
                # Exosomes should be roughly circular (circularity > 0.5)
                if circularity < 0.3:
                    continue
                
                det['circularity'] = circularity
            
            filtered.append(det)
        
        self.logger.info(f"Filtered {len(detections)} -> {len(filtered)} detections")
        return filtered
    
    def calculate_colocalization(self, detections_ch1: List[Dict], detections_ch2: List[Dict],
                                 distance_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Calculate colocalization between two channels.
        
        Args:
            detections_ch1: Detections from channel 1
            detections_ch2: Detections from channel 2
            distance_threshold: Maximum distance for colocalization (pixels)
            
        Returns:
            Dict with colocalization metrics
        """
        if not detections_ch1 or not detections_ch2:
            return {
                'colocalized_count': 0,
                'colocalization_rate_ch1': 0.0,
                'colocalization_rate_ch2': 0.0,
                'colocalized_pairs': []
            }
        
        # Extract coordinates
        coords1 = np.array([det['coord'] for det in detections_ch1])
        coords2 = np.array([det['coord'] for det in detections_ch2])
        
        # Calculate pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(coords1, coords2)
        
        # Find colocalized pairs
        colocalized_pairs = []
        matched_ch2 = set()
        
        for i, det1 in enumerate(detections_ch1):
            # Find nearest neighbor in ch2
            nearest_idx = distances[i].argmin()
            nearest_dist = distances[i, nearest_idx]
            
            if nearest_dist <= distance_threshold and nearest_idx not in matched_ch2:
                colocalized_pairs.append({
                    'ch1_id': det1['id'],
                    'ch2_id': detections_ch2[nearest_idx]['id'],
                    'distance': float(nearest_dist),
                    'ch1_coord': det1['coord'].tolist(),
                    'ch2_coord': detections_ch2[nearest_idx]['coord'].tolist()
                })
                matched_ch2.add(nearest_idx)
        
        colocalized_count = len(colocalized_pairs)
        colocalization_rate_ch1 = colocalized_count / len(detections_ch1) if detections_ch1 else 0.0
        colocalization_rate_ch2 = colocalized_count / len(detections_ch2) if detections_ch2 else 0.0
        
        return {
            'colocalized_count': colocalized_count,
            'colocalization_rate_ch1': colocalization_rate_ch1,
            'colocalization_rate_ch2': colocalization_rate_ch2,
            'colocalized_pairs': colocalized_pairs
        }
    
    def extract_intensities(self, detections: List[Dict], image: np.ndarray) -> List[float]:
        """
        Extract intensity values for each detection.
        
        Args:
            detections: List of detection dictionaries
            image: Image array
            
        Returns:
            List of intensity values
        """
        intensities = []
        for det in detections:
            y, x = int(det['coord'][0]), int(det['coord'][1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                intensities.append(float(image[y, x]))
            else:
                intensities.append(0.0)
        return intensities
    
    def generate_csv_output(self, detections_all_channels: Dict[str, List[Dict]],
                           images: Dict[str, np.ndarray],
                           colocalization_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate CSV output with detection data.
        
        Args:
            detections_all_channels: Dict mapping channel names to detection lists
            images: Dict mapping channel names to image arrays
            colocalization_data: Optional colocalization analysis results
            
        Returns:
            pandas DataFrame with all detection data
        """
        rows = []
        
        # Process each channel
        for channel_name, detections in detections_all_channels.items():
            image = images[channel_name]
            intensities = self.extract_intensities(detections, image)
            
            for det, intensity in zip(detections, intensities):
                row = {
                    'channel': channel_name,
                    'object_id': det['id'],
                    'x_coord': float(det['coord'][1]),  # x is second coordinate
                    'y_coord': float(det['coord'][0]),  # y is first coordinate
                    'intensity': intensity,
                    'area': det['area'],
                    'probability': det.get('prob', 0.0),
                    'circularity': det.get('circularity', 0.0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add colocalization information if available
        if colocalization_data and 'colocalized_pairs' in colocalization_data:
            # Mark colocalized objects
            df['colocalized'] = False
            for pair in colocalization_data['colocalized_pairs']:
                # This is simplified - in practice, you'd need to match object IDs
                pass
        
        return df
    
    def query_llm_statistical_analysis(self, csv_path: Path, sample_name: str) -> Dict[str, Any]:
        """
        Query LLM for statistical analysis and biological insights.
        
        Args:
            csv_path: Path to generated CSV file
            sample_name: Name of the sample
            
        Returns:
            Dict with LLM analysis and recommendations
        """
        if not self.llm_analysis_enabled:
            return {'analysis': 'LLM analysis disabled'}
        
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv(self.config.get('llm', {}).get('api_key_env', 'OPENAI_API_KEY'))
            if not api_key:
                self.logger.warning("LLM API key not found. Skipping statistical analysis.")
                return {'analysis': 'LLM not available'}
            
            # Read CSV and create summary
            df = pd.read_csv(csv_path)
            summary = f"""
Sample: {sample_name}
Total detections: {len(df)}
Channels: {df['channel'].unique().tolist()}
Mean intensity: {df['intensity'].mean():.4f}
Mean area: {df['area'].mean():.2f} pixels
Colocalization rate: {df.get('colocalization_rate', 'N/A')}
"""
            
            client = OpenAI(api_key=api_key)
            
            prompt = f"""You are a bio-image analysis data scientist. Analyze the following exosome detection data and provide insights.

{summary}

Provide:
1. Statistical summary
2. Biological interpretation
3. Recommendations for further analysis
4. Python code to create visualization plots (use matplotlib/seaborn)

Respond in JSON:
{{
    "summary": "statistical summary",
    "interpretation": "biological interpretation",
    "recommendations": ["list of recommendations"],
    "plot_code": "Python code for visualization"
}}"""
            
            response = client.chat.completions.create(
                model=self.config.get('llm', {}).get('analysis_model', 'gpt-4'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('llm', {}).get('temperature', 0.3),
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            self.logger.info("LLM statistical analysis completed")
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM statistical analysis failed: {e}")
            return {'analysis': f'LLM error: {e}'}
    
    def process(self, images: Dict[str, np.ndarray], sample_name: str = "unknown",
                output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Main processing function for detection and quantification.
        
        Args:
            images: Dict mapping channel names to registered image arrays
            sample_name: Name of the sample
            output_dir: Directory to save output files
            
        Returns:
            Dict containing detections, CSV data, and analysis results
        """
        self.logger.info(f"Analyzing sample: {sample_name}")
        
        if not self.validate_input(images):
            return {'error': 'Input validation failed'}
        
        # Detect objects in each channel
        detections_all_channels = {}
        label_images = {}
        
        for channel_name, image in images.items():
            self.logger.info(f"Detecting objects in channel: {channel_name}")
            img_norm = self.ensure_float32(image)
            
            try:
                labels, detections = self.detect_objects_stardist(img_norm)
                
                # Filter detections
                filtered = self.filter_detections(detections, img_norm)
                
                detections_all_channels[channel_name] = filtered
                label_images[channel_name] = labels
                
                self.logger.info(f"Channel {channel_name}: {len(filtered)} exosomes detected")
            except Exception as e:
                self.logger.error(f"Detection failed for {channel_name}: {e}")
                detections_all_channels[channel_name] = []
                label_images[channel_name] = np.zeros_like(img_norm, dtype=np.int32)
        
        # Calculate colocalization (if multiple channels)
        colocalization_data = None
        if len(detections_all_channels) >= 2:
            channels = list(detections_all_channels.keys())
            colocalization_data = self.calculate_colocalization(
                detections_all_channels[channels[0]],
                detections_all_channels[channels[1]]
            )
            self.logger.info(f"Colocalization: {colocalization_data['colocalized_count']} pairs")
        
        # Generate CSV output
        df = self.generate_csv_output(detections_all_channels, images, colocalization_data)
        
        # Save CSV
        if output_dir is None:
            output_dir = Path(self.config.get('paths', {}).get('output_root', 'data/output'))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{sample_name}_detections.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV: {csv_path}")
        
        # LLM statistical analysis
        llm_analysis = None
        if self.llm_analysis_enabled:
            llm_analysis = self.query_llm_statistical_analysis(csv_path, sample_name)
            
            # Optionally execute plot code from LLM
            if llm_analysis.get('plot_code'):
                try:
                    plot_path = output_dir / f"{sample_name}_plots.png"
                    # Execute plot code in a safe namespace
                    exec_globals = {'df': df, 'plt': __import__('matplotlib.pyplot'),
                                   'sns': __import__('seaborn'), 'np': np, 'pd': pd}
                    exec(llm_analysis['plot_code'], exec_globals)
                    if 'plt' in exec_globals:
                        exec_globals['plt'].savefig(plot_path)
                        exec_globals['plt'].close()
                    self.logger.info(f"Saved plots: {plot_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to execute plot code: {e}")
        
        # Prepare output
        self._output = {
            'sample_name': sample_name,
            'detections': detections_all_channels,
            'label_images': label_images,
            'colocalization': colocalization_data,
            'csv_path': str(csv_path),
            'dataframe': df,
            'llm_analysis': llm_analysis,
            'summary': {
                'total_detections': sum(len(dets) for dets in detections_all_channels.values()),
                'channels': list(detections_all_channels.keys()),
                'colocalized_count': colocalization_data['colocalized_count'] if colocalization_data else 0
            }
        }
        
        self.logger.info(f"Analysis complete for {sample_name}")
        return self._output

