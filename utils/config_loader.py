"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Dict containing configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")

        # Allow runtime device override (set by api_server_extended.py when
        # CUDA is not available, so agents fall back to CPU automatically).
        import os
        device_override = os.environ.get('SEA_DEVICE_OVERRIDE')
        if device_override:
            config.setdefault('hardware', {})['device'] = device_override
            logger.warning(f"Device overridden to '{device_override}' via SEA_DEVICE_OVERRIDE")

        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

