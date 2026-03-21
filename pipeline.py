"""
Main Pipeline Controller for SEA Exosome Analysis
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import torch
import numpy as np
import tifffile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import Inspector, Aligner, Analyst
from utils.config_loader import load_config
from utils.image_utils import save_image, create_overlay


class ExosomeAnalysisPipeline:
    """
    Main pipeline controller that orchestrates the three agent phases.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup paths
        self.input_root = Path(self.config.get('paths', {}).get('input_root', 'data/input'))
        self.output_root = Path(self.config.get('paths', {}).get('output_root', 'data/output'))
        self.models_dir = Path(self.config.get('paths', {}).get('models_dir', 'models'))
        self.logs_dir = Path(self.config.get('paths', {}).get('logs_dir', 'logs'))
        
        # Create directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize agents
        self.inspector = Inspector(
            self.config.get('inspector', {}),
            device=self.device
        )
        self.aligner = Aligner(
            self.config.get('aligner', {}),
            device=self.device
        )
        self.analyst = Analyst(
            self.config.get('analyst', {}),
            device=self.device
        )
        
        logger.info("Pipeline initialized")
        logger.info(f"Input directory: {self.input_root}")
        logger.info(f"Output directory: {self.output_root}")
        logger.info(f"Device: {self.device}")
    
    def _setup_logging(self):
        """Configure logging."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
        log_file = self.logs_dir / 'pipeline.log'
        
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level
        )
        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=log_config.get('rotation', '10 MB'),
            retention=log_config.get('retention', '7 days')
        )
    
    def _setup_device(self) -> str:
        """Setup computing device (CUDA or CPU)."""
        device_config = self.config.get('hardware', {}).get('device', 'cuda')
        
        if device_config == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.warning("CUDA not available, falling back to CPU")
        else:
            device = 'cpu'
            logger.info("Using CPU device")
        
        return device
    
    def find_samples(self) -> list:
        """
        Find all sample directories in INPUT folder.
        
        Returns:
            List of Path objects for sample directories
        """
        if not self.input_root.exists():
            logger.error(f"Input directory does not exist: {self.input_root}")
            return []
        
        samples = [d for d in self.input_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(samples)} samples in {self.input_root}")
        return samples
    
    def process_sample(self, sample_dir: Path) -> Dict[str, Any]:
        """
        Process a single sample through all three pipeline phases.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Dict containing results from all phases
        """
        sample_name = sample_dir.name
        logger.info(f"=" * 60)
        logger.info(f"Processing sample: {sample_name}")
        logger.info(f"=" * 60)
        
        results = {
            'sample_name': sample_name,
            'sample_dir': str(sample_dir),
            'phases': {}
        }
        
        try:
            # Phase 1: Inspector
            logger.info("Phase 1: Inspection and Preprocessing")
            inspection_result = self.inspector.process(sample_dir)
            
            if 'error' in inspection_result:
                logger.error(f"Inspection failed: {inspection_result['error']}")
                results['error'] = inspection_result['error']
                return results
            
            results['phases']['inspection'] = inspection_result
            images = inspection_result['images']
            anchor_channel = inspection_result['anchor_channel']
            
            # Save preprocessed images
            preprocessed_dir = self.output_root / sample_name / "preprocessed"
            for channel_name, image in images.items():
                output_path = preprocessed_dir / f"{channel_name}_preprocessed.png"
                save_image(image, output_path, bit_depth=8, format='png')
            
            # Phase 2: Aligner
            logger.info("Phase 2: Registration and Alignment")
            alignment_result = self.aligner.process(
                images=images,
                anchor_channel=anchor_channel,
                sample_name=sample_name
            )
            
            if 'error' in alignment_result:
                logger.error(f"Alignment failed: {alignment_result['error']}")
                results['error'] = alignment_result['error']
                return results
            
            results['phases']['alignment'] = alignment_result
            registered_images = alignment_result['registered_images']
            
            # Save registered images
            registered_dir = self.output_root / sample_name / "registered"
            for channel_name, image in registered_images.items():
                output_path = registered_dir / f"{channel_name}_registered.png"
                save_image(image, output_path, bit_depth=8, format='png')
            
            # Create overlay
            overlay_path = self.output_root / sample_name / f"{sample_name}_overlay.png"
            create_overlay(registered_images, overlay_path)
            
            # Phase 3: Analyst
            logger.info("Phase 3: Detection and Quantification")
            analysis_result = self.analyst.process(
                images=registered_images,
                sample_name=sample_name,
                output_dir=self.output_root / sample_name
            )
            
            if 'error' in analysis_result:
                logger.error(f"Analysis failed: {analysis_result['error']}")
                results['error'] = analysis_result['error']
                return results
            
            results['phases']['analysis'] = analysis_result
            
            # Save label images
            label_dir = self.output_root / sample_name / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            for channel_name, label_image in analysis_result['label_images'].items():
                output_path = label_dir / f"{channel_name}_labels.png"
                # Convert label image to 8-bit for PNG (labels are typically small integers)
                label_8bit = (label_image.astype(np.float32) / label_image.max() * 255).astype(np.uint8) if label_image.max() > 0 else label_image.astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(label_8bit, mode='L')
                pil_img.save(str(output_path), 'PNG', optimize=True)
            
            logger.info(f"✓ Sample {sample_name} processed successfully")
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_name}: {e}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def run(self, sample_filter: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline on all samples.
        
        Args:
            sample_filter: Optional list of sample names to process. If None, processes all.
            
        Returns:
            Dict containing results for all samples
        """
        logger.info("Starting SEA Exosome Analysis Pipeline")
        logger.info(f"Input: {self.input_root}")
        logger.info(f"Output: {self.output_root}")
        
        # Find samples
        all_samples = self.find_samples()
        
        if not all_samples:
            logger.error("No samples found. Check input directory structure.")
            return {'error': 'No samples found'}
        
        # Filter samples if specified
        if sample_filter:
            samples = [s for s in all_samples if s.name in sample_filter]
            logger.info(f"Filtered to {len(samples)} samples")
        else:
            samples = all_samples
        
        # Process each sample
        all_results = {}
        for sample_dir in samples:
            result = self.process_sample(sample_dir)
            all_results[sample_dir.name] = result
        
        # Summary
        successful = sum(1 for r in all_results.values() if r.get('status') == 'success')
        failed = len(all_results) - successful
        
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info(f"Total samples: {len(all_results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("=" * 60)
        
        return {
            'summary': {
                'total': len(all_results),
                'successful': successful,
                'failed': failed
            },
            'results': all_results
        }


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SEA: Exosome Analysis Pipeline'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--samples',
        nargs='+',
        default=None,
        help='Optional list of sample names to process (default: all)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Override input directory path'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ExosomeAnalysisPipeline(config_path=args.config)
    
    # Override input directory if specified
    if args.input:
        pipeline.input_root = Path(args.input)
        logger.info(f"Using custom input directory: {pipeline.input_root}")
    
    # Run pipeline
    results = pipeline.run(sample_filter=args.samples)
    
    # Exit with appropriate code
    if results.get('summary', {}).get('failed', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

