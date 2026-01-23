"""
Abstract base class for all pipeline agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class defining the interface for all pipeline agents.
    
    Each agent (Inspector, Aligner, Analyst) must implement:
    - process(): Main processing logic
    - validate_input(): Input validation
    - get_output(): Return processed results
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Initialize the agent with configuration.
        
        Args:
            config: Configuration dictionary for this agent
            device: Computing device ("cuda" or "cpu")
        """
        self.config = config
        self.device = device
        self.logger = logger.bind(agent=self.__class__.__name__)
        self._output = None
        
    @abstractmethod
    def validate_input(self, *args, **kwargs) -> bool:
        """
        Validate input data before processing.
        
        Returns:
            bool: True if input is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main processing logic for the agent.
        
        Returns:
            Dict containing processed results and metadata
        """
        pass
    
    def get_output(self) -> Optional[Dict[str, Any]]:
        """
        Get the last processed output.
        
        Returns:
            Dict containing results, or None if not processed yet
        """
        return self._output
    
    def save_output(self, output_path: Path, *args, **kwargs) -> bool:
        """
        Save agent output to disk.
        
        Args:
            output_path: Path to save the output
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self._output is None:
            self.logger.warning("No output to save. Run process() first.")
            return False
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Implementation depends on agent type
            return True
        except Exception as e:
            self.logger.error(f"Failed to save output: {e}")
            return False
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load a TIFF image preserving bit depth.
        
        Args:
            image_path: Path to the TIFF file
            
        Returns:
            np.ndarray: Image array (16-bit or 32-bit)
        """
        try:
            import tifffile
            image = tifffile.imread(str(image_path))
            self.logger.debug(f"Loaded {image_path.name}: shape={image.shape}, dtype={image.dtype}")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def ensure_float32(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to float32 while preserving dynamic range.
        
        Args:
            image: Input image array
            
        Returns:
            np.ndarray: Float32 image normalized to [0, 1]
        """
        if image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint32:
            return image.astype(np.float32) / 4294967295.0
        elif image.dtype == np.float32:
            # Already float32, but ensure it's in [0, 1]
            return np.clip(image, 0, 1)
        elif image.dtype == np.float64:
            return image.astype(np.float32)
        else:
            # Assume uint8
            return image.astype(np.float32) / 255.0

