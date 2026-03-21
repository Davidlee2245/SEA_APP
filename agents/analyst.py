"""
Phase 3: Analyst Agent
- Detects and counts exosomes using StarDist
- Analyzes colocalization
- Generates CSV output with coordinates and intensities
- Uses LLM for statistical interpretation
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict
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
    
    def detect_objects_stardist(self, image: np.ndarray, prob_thresh: float = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect objects using StarDist.
        
        Args:
            image: Input image array (float32, [0, 1] or any dtype)
            prob_thresh: Probability threshold for detection (default from config or 0.3)
            
        Returns:
            Tuple of (label_image, detections_list)
            detections_list contains dicts with 'coord', 'prob', 'points'
        """
        if self._stardist_model is None:
            if not self.load_stardist_model():
                raise RuntimeError("StarDist model not available")
        
        # Get probability threshold from config or use default
        if prob_thresh is None:
            prob_thresh = self.config.get('stardist_prob_thresh', 0.3)
        
        # Normalize image for StarDist using percentile normalization (best practice)
        # This preserves contrast better than simple scaling, especially for wide dynamic range images
        from csbdeep.utils import normalize
        
        # Ensure image is in a workable format
        if image.dtype == np.uint16:
            # For uint16, normalize directly
            img_normalized = normalize(image, pmin=1, pmax=99, axis=None)
        else:
            # For float32 [0,1], convert to uint8 first, then normalize
            img_norm = self.ensure_float32(image)
            img_uint8 = (img_norm * 255).astype(np.uint8)
            img_normalized = normalize(img_uint8, pmin=1, pmax=99, axis=None)
        
        # Run detection with lower probability threshold to catch more cells
        # sparse=False avoids ClipperLib's polygon NMS (which crashes with
        # "Coordinate outside allowed range" on certain images) by using
        # a dense NMS algorithm instead.
        labels, details = self._stardist_model.predict_instances(
            img_normalized,
            prob_thresh=prob_thresh,
            nms_thresh=0.3,  # Non-maximum suppression threshold
            sparse=False
        )
        
        # Extract detection information
        detections = []
        for i, (coord, prob, points) in enumerate(zip(
            details['coord'],
            details['prob'],
            details['points']
        )):
            # Ensure coord is a simple array/list, not nested
            coord_array = np.asarray(coord).flatten()
            detections.append({
                'id': i + 1,
                'coord': coord_array,  # Center coordinates (y, x) as 1D array
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
        filter_stats = {'size': 0, 'intensity': 0, 'morphology': 0, 'passed': 0}
        
        # Normalize image intensity threshold relative to image range
        img_max = image.max()
        img_min = image.min()
        intensity_threshold_abs = img_min + (img_max - img_min) * self.intensity_threshold
        
        for det in detections:
            # Size filter
            area = det['area']
            if area < self.min_object_size or area > self.max_object_size:
                filter_stats['size'] += 1
                continue
            
            # Intensity filter
            coord = det['coord']
            y, x = int(np.round(coord[0])), int(np.round(coord[1]))
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                intensity = image[y, x]
                if intensity < intensity_threshold_abs:
                    filter_stats['intensity'] += 1
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
                    filter_stats['morphology'] += 1
                    continue
                
                det['circularity'] = circularity
            
            filter_stats['passed'] += 1
            filtered.append(det)
        
        self.logger.info(f"Filtered {len(detections)} -> {len(filtered)} detections (size: {filter_stats['size']}, intensity: {filter_stats['intensity']}, morphology: {filter_stats['morphology']})")
        return filtered
    
    def calculate_colocalization(self, detections_ch1: List[Dict], detections_ch2: List[Dict],
                                 distance_threshold: float = 3.0, channel1_name: str = None,
                                 channel2_name: str = None) -> Dict[str, Any]:
        """
        Calculate colocalization between two channels.
        Uses improved matching algorithm that finds all pairs within threshold.
        
        Args:
            detections_ch1: Detections from channel 1
            detections_ch2: Detections from channel 2
            distance_threshold: Maximum distance for colocalization (pixels)
            channel1_name: Name of channel 1 (for CSV marking)
            channel2_name: Name of channel 2 (for CSV marking)
            
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
        
        # Extract coordinates - coord is stored as (y, x), convert to (x, y) for cdist
        # cdist expects (x, y) format for proper Euclidean distance
        coords1 = np.array([[det['coord'][1], det['coord'][0]] for det in detections_ch1])  # (x, y)
        coords2 = np.array([[det['coord'][1], det['coord'][0]] for det in detections_ch2])  # (x, y)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(coords1, coords2)
        
        # Improved matching: Find all pairs within threshold (many-to-many)
        # Then use greedy matching to avoid double-counting
        colocalized_pairs = []
        matched_ch1 = set()
        matched_ch2 = set()
        
        # Find all candidate pairs within threshold
        candidates = []
        for i, det1 in enumerate(detections_ch1):
            for j, det2 in enumerate(detections_ch2):
                dist = distances[i, j]
                if dist <= distance_threshold:
                    candidates.append((i, j, dist, det1['id'], det2['id']))
        
        # Sort by distance (closest first) for greedy matching
        candidates.sort(key=lambda x: x[2])
        
        # Greedy matching: match closest pairs first, avoiding duplicates
        for i, j, dist, id1, id2 in candidates:
            if i not in matched_ch1 and j not in matched_ch2:
                colocalized_pairs.append({
                    'ch1_id': id1,
                    'ch2_id': id2,
                    'ch1_name': channel1_name,
                    'ch2_name': channel2_name,
                    'distance': float(dist),
                    'ch1_coord': detections_ch1[i]['coord'].tolist(),
                    'ch2_coord': detections_ch2[j]['coord'].tolist()
                })
                matched_ch1.add(i)
                matched_ch2.add(j)
        
        colocalized_count = len(colocalized_pairs)
        colocalization_rate_ch1 = colocalized_count / len(detections_ch1) if detections_ch1 else 0.0
        colocalization_rate_ch2 = colocalized_count / len(detections_ch2) if detections_ch2 else 0.0
        
        return {
            'colocalized_count': colocalized_count,
            'colocalization_rate_ch1': colocalization_rate_ch1,
            'colocalization_rate_ch2': colocalization_rate_ch2,
            'colocalized_pairs': colocalized_pairs
        }
    
    def extract_channel_id(self, channel_name: str) -> str:
        """
        Extract channel identifier (ch1, ch2, etc.) from full channel name.
        
        Examples:
            "r01c01f01p01-ch2sk1fk1fl1" -> "ch2"
            "r01c01f01p01-ch3sk1fk1fl1" -> "ch3"
            "ch2" -> "ch2" (already normalized)
        
        Args:
            channel_name: Full channel name or already-normalized channel ID
            
        Returns:
            Channel ID (e.g., "ch2", "ch3")
        """
        # If already in format "chN", return as-is
        if channel_name.startswith('ch') and len(channel_name) <= 4:
            return channel_name
        
        # Extract channel ID from full name using regex
        # Pattern: -ch followed by digits
        match = re.search(r'-ch(\d+)', channel_name)
        if match:
            return f"ch{match.group(1)}"
        
        # Fallback: try to find "ch" followed by digits anywhere
        match = re.search(r'ch(\d+)', channel_name)
        if match:
            return f"ch{match.group(1)}"
        
        # If no pattern found, return original (shouldn't happen)
        self.logger.warning(f"Could not extract channel ID from: {channel_name}")
        return channel_name
    
    def normalize_channel_combination(self, combo: List[str]) -> List[str]:
        """
        Normalize a channel combination to use only channel IDs, sorted numerically.
        
        Args:
            combo: List of full channel names or channel IDs
            
        Returns:
            Sorted list of normalized channel IDs (e.g., ["ch1", "ch2", "ch3"])
        """
        # Extract channel IDs
        channel_ids = [self.extract_channel_id(ch) for ch in combo]
        
        # Remove duplicates and sort numerically (ch1 < ch2 < ch3 ...)
        unique_ids = list(set(channel_ids))
        
        def channel_sort_key(ch_id: str) -> int:
            # Extract number from "chN"
            match = re.search(r'ch(\d+)', ch_id)
            if match:
                return int(match.group(1))
            return 999  # Put unknown channels at end
        
        unique_ids.sort(key=channel_sort_key)
        return unique_ids
    
    def calculate_multi_channel_combinations(
        self, 
        detections_all_channels: Dict[str, List[Dict]], 
        distance_threshold: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calculate multi-channel colocalization combinations using connected components.
        
        Groups objects across channels into connected components, then counts
        how many objects have each channel combination signature.
        
        Args:
            detections_all_channels: Dict mapping channel names to detection lists
            distance_threshold: Maximum distance for colocalization (pixels)
            
        Returns:
            Dict with combo_table (normalized to channel IDs only) and statistics
        """
        import re
        if len(detections_all_channels) < 2:
            channels = list(detections_all_channels.keys())
            # Normalize channel names to channel IDs
            normalized_channels = [self.extract_channel_id(ch) for ch in channels]
            normalized_channels = sorted(set(normalized_channels), key=lambda x: int(re.search(r'ch(\d+)', x).group(1)) if re.search(r'ch(\d+)', x) else 999)
            
            return {
                'channels_present': normalized_channels,
                'threshold_px': distance_threshold,
                'combo_table': [
                    {
                        'combo': [self.extract_channel_id(ch)],
                        'count': len(detections_all_channels[ch]),
                        'rate': 1.0 if len(detections_all_channels) == 1 else 0.0
                    }
                    for ch in channels
                ]
            }
        
        channels = list(detections_all_channels.keys())
        # Normalize channel names to channel IDs for output
        normalized_channels = [self.extract_channel_id(ch) for ch in channels]
        normalized_channels = sorted(set(normalized_channels), key=lambda x: int(re.search(r'ch(\d+)', x).group(1)) if re.search(r'ch(\d+)', x) else 999)
        
        # Step 1: Get all pairwise matches (reuse existing logic)
        all_pairs = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch1, ch2 = channels[i], channels[j]
                coloc = self.calculate_colocalization(
                    detections_all_channels[ch1],
                    detections_all_channels[ch2],
                    distance_threshold=distance_threshold,
                    channel1_name=ch1,
                    channel2_name=ch2
                )
                all_pairs.extend(coloc['colocalized_pairs'])
        
        # Step 2: Build graph - create node identifiers
        # Node format: (channel_name, object_id)
        node_to_index = {}
        index_to_node = []
        node_channel = {}  # Track which channel each node belongs to
        
        # Add all detections as nodes
        for channel_name, detections in detections_all_channels.items():
            for det in detections:
                node_id = (channel_name, det['id'])
                if node_id not in node_to_index:
                    idx = len(index_to_node)
                    node_to_index[node_id] = idx
                    index_to_node.append(node_id)
                    node_channel[idx] = channel_name
        
        # Step 3: Union-Find data structure for connected components
        parent = list(range(len(index_to_node)))
        rank = [0] * len(index_to_node)
        
        def find(x: int) -> int:
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int):
            """Union by rank."""
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # Step 4: Build edges from pairwise matches
        for pair in all_pairs:
            ch1_name = pair['ch1_name']
            ch1_id = pair['ch1_id']
            ch2_name = pair['ch2_name']
            ch2_id = pair['ch2_id']
            
            node1 = (ch1_name, ch1_id)
            node2 = (ch2_name, ch2_id)
            
            if node1 in node_to_index and node2 in node_to_index:
                idx1 = node_to_index[node1]
                idx2 = node_to_index[node2]
                union(idx1, idx2)
        
        # Step 5: Group nodes by connected component
        components = defaultdict(list)
        for idx in range(len(index_to_node)):
            root = find(idx)
            components[root].append(idx)
        
        # Step 6: For each component, compute channel combination signature
        combo_counts = defaultdict(int)
        
        for component_nodes in components.values():
            # Get unique channels in this component
            channels_in_component = set()
            for node_idx in component_nodes:
                channels_in_component.add(node_channel[node_idx])
            
            # Create sorted tuple for consistent key
            combo_signature = tuple(sorted(channels_in_component))
            combo_counts[combo_signature] += 1
        
        # Step 7: Build combo_table with normalized channel IDs
        total_objects = len(components)
        combo_table = []
        
        for combo_sig, count in sorted(combo_counts.items(), key=lambda x: (-len(x[0]), x[0])):
            # Normalize channel combination to use only channel IDs, sorted
            normalized_combo = self.normalize_channel_combination(list(combo_sig))
            combo_table.append({
                'combo': normalized_combo,
                'count': count,
                'rate': count / total_objects if total_objects > 0 else 0.0
            })
        
        return {
            'channels_present': normalized_channels,
            'threshold_px': distance_threshold,
            'combo_table': combo_table
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
            coord = det['coord']
            y, x = int(np.round(coord[0])), int(np.round(coord[1]))
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
                    'x_coord': float(np.round(det['coord'][1])),  # x is second coordinate
                    'y_coord': float(np.round(det['coord'][0])),  # y is first coordinate
                    'intensity': intensity,
                    'area': det['area'],
                    'probability': det.get('prob', 0.0),
                    'circularity': det.get('circularity', 0.0)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add colocalization information if available
        df['colocalized'] = False  # Initialize all as False
        
        if colocalization_data:
            # Get all colocalized pairs (from all_pairs if available, otherwise from colocalized_pairs)
            all_pairs = colocalization_data.get('all_pairs', colocalization_data.get('colocalized_pairs', []))
            
            if all_pairs:
                # Create sets of colocalized object IDs per channel for fast lookup
                coloc_by_channel = {}
                for channel_name in df['channel'].unique():
                    coloc_by_channel[channel_name] = set()
                
                # Collect all colocalized object IDs
                for pair in all_pairs:
                    ch1_name = pair.get('ch1_name') or pair.get('ch1')
                    ch2_name = pair.get('ch2_name') or pair.get('ch2')
                    ch1_id = pair.get('ch1_id')
                    ch2_id = pair.get('ch2_id')
                    
                    if ch1_name and ch1_id is not None:
                        if ch1_name not in coloc_by_channel:
                            coloc_by_channel[ch1_name] = set()
                        coloc_by_channel[ch1_name].add(ch1_id)
                    
                    if ch2_name and ch2_id is not None:
                        if ch2_name not in coloc_by_channel:
                            coloc_by_channel[ch2_name] = set()
                        coloc_by_channel[ch2_name].add(ch2_id)
                
                # Mark colocalized objects in DataFrame
                for channel_name, coloc_ids in coloc_by_channel.items():
                    if coloc_ids:
                        mask = (df['channel'] == channel_name) & (df['object_id'].isin(coloc_ids))
                        df.loc[mask, 'colocalized'] = True
        
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
            
            import json
            
            # Try with response_format first, fallback without it if not supported
            try:
                response = client.chat.completions.create(
                    model=self.config.get('llm', {}).get('analysis_model', 'gpt-4'),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.get('llm', {}).get('temperature', 0.3),
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                if "response_format" in str(e):
                    self.logger.info("Model doesn't support json_object format, retrying without it")
                    response = client.chat.completions.create(
                        model=self.config.get('llm', {}).get('analysis_model', 'gpt-4'),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.get('llm', {}).get('temperature', 0.3)
                    )
                else:
                    raise
            
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
        
        # Calculate multi-channel colocalization combinations
        colocalization_data = None
        combo_analysis = None
        colocalization_distance = self.config.get('colocalization_distance', 10.0)
        
        if len(detections_all_channels) >= 2:
            channels = list(detections_all_channels.keys())
            
            # NEW: Calculate multi-channel combinations using connected components
            combo_analysis = self.calculate_multi_channel_combinations(
                detections_all_channels,
                distance_threshold=colocalization_distance
            )
            
            self.logger.info(f"Multi-channel combination analysis:")
            self.logger.info(f"  Unique combinations: {len(combo_analysis['combo_table'])}")
            for combo_entry in combo_analysis['combo_table']:
                combo_str = ' + '.join(combo_entry['combo'])
                self.logger.info(f"    {combo_str}: {combo_entry['count']} objects ({combo_entry['rate']*100:.1f}%)")
            
            # Also calculate pairwise matches for CSV marking (backward compatibility)
            all_coloc_pairs = []
            total_coloc_count = 0
            
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    ch1, ch2 = channels[i], channels[j]
                    coloc = self.calculate_colocalization(
                        detections_all_channels[ch1],
                        detections_all_channels[ch2],
                        distance_threshold=colocalization_distance,
                        channel1_name=ch1,
                        channel2_name=ch2
                    )
                    total_coloc_count += coloc['colocalized_count']
                    all_coloc_pairs.extend(coloc['colocalized_pairs'])
                    self.logger.info(f"Pairwise {ch1} <-> {ch2}: {coloc['colocalized_count']} pairs")
            
            # Create comprehensive colocalization data structure (for CSV marking)
            colocalization_data = {
                'colocalized_count': total_coloc_count,
                'total_pairs_all_channels': total_coloc_count,
                'all_pairs': all_coloc_pairs,  # All pairs from all channel combinations
                'colocalized_pairs': all_coloc_pairs,  # For backward compatibility
                'colocalization_rate_ch1': 0.0,  # Will be calculated per channel in CSV
                'colocalization_rate_ch2': 0.0,
                'combo_analysis': combo_analysis  # NEW: Include combo analysis
            }
            
            self.logger.info(f"Total pairwise matches: {total_coloc_count} pairs")
        elif len(detections_all_channels) == 1:
            # Single channel case
            channel = list(detections_all_channels.keys())[0]
            # Normalize single channel to channel ID
            normalized_ch = self.extract_channel_id(channel)
            combo_analysis = {
                'channels_present': [normalized_ch],
                'threshold_px': colocalization_distance,
                'combo_table': [
                    {
                        'combo': [normalized_ch],
                        'count': len(detections_all_channels[channel]),
                        'rate': 1.0
                    }
                ]
            }
        
        # Generate CSV output
        df = self.generate_csv_output(detections_all_channels, images, colocalization_data)
        
        # Save CSV
        if output_dir is None:
            output_dir = Path(self.config.get('paths', {}).get('output_root', 'data/output'))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{sample_name}_detections.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV: {csv_path}")
        
        # Save combo analysis to JSON
        if combo_analysis:
            import json
            combo_json_path = output_dir / f"{sample_name}_combinations.json"
            with open(combo_json_path, 'w') as f:
                json.dump(combo_analysis, f, indent=2)
            self.logger.info(f"Saved combo analysis: {combo_json_path}")
        
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
            'combo_analysis': combo_analysis,  # NEW: Multi-channel combination analysis
            'csv_path': str(csv_path),
            'dataframe': df,
            'llm_analysis': llm_analysis,
            'summary': {
                'total_detections': sum(len(dets) for dets in detections_all_channels.values()),
                'channels': list(detections_all_channels.keys()),
                'colocalized_count': colocalization_data['colocalized_count'] if colocalization_data else 0,
                'total_objects_grouped': combo_analysis['total_objects_grouped'] if combo_analysis else 0
            }
        }
        
        self.logger.info(f"Analysis complete for {sample_name}")
        return self._output

