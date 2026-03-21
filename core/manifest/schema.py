"""
JSON Schema and validation for preprocessing manifest.
Defines the strict data contract for agent-recommended parameters.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


# Preprocessing parameter schema with valid ranges
PREPROCESSING_PARAM_SCHEMA = {
    "denoising_strength": {"type": "float", "min": 0.0, "max": 1.0},
    "background_subtraction_radius": {"type": "int", "min": 1, "max": 500},
    "clahe_clip_limit": {"type": "float", "min": 0.1, "max": 10.0},
    "clahe_grid_size": {"type": "int", "min": 2, "max": 32},
    "threshold_value": {"type": "int", "min": 0, "max": 255},
    "threshold_method": {"type": "str", "allowed": ["otsu", "adaptive", "binary", "fixed"]},
    "gamma_correction": {"type": "float", "min": 0.1, "max": 5.0},
    "sharpening_strength": {"type": "float", "min": 0.0, "max": 2.0},
    "contrast_alpha": {"type": "float", "min": 0.5, "max": 3.0},
    "brightness_beta": {"type": "int", "min": -100, "max": 100},
}


# Status enum
VALID_STATUSES = ["recommended", "approved", "edited", "applied", "failed", "cancelled"]


def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate preprocessing parameters against schema.
    
    Args:
        params: Dictionary of parameter name -> value
        
    Returns:
        Validated and clamped parameters
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(params, dict):
        raise ValidationError(f"params must be a dict, got {type(params)}")
    
    validated = {}
    
    for key, value in params.items():
        if key not in PREPROCESSING_PARAM_SCHEMA:
            # Unknown parameter - store as-is but warn
            validated[key] = value
            continue
        
        schema = PREPROCESSING_PARAM_SCHEMA[key]
        param_type = schema["type"]
        
        # Type validation and coercion
        try:
            if param_type == "float":
                value = float(value)
                # Clamp to valid range
                if "min" in schema:
                    value = max(value, schema["min"])
                if "max" in schema:
                    value = min(value, schema["max"])
            elif param_type == "int":
                value = int(value)
                # Clamp to valid range
                if "min" in schema:
                    value = max(value, schema["min"])
                if "max" in schema:
                    value = min(value, schema["max"])
            elif param_type == "str":
                value = str(value)
                if "allowed" in schema and value not in schema["allowed"]:
                    raise ValidationError(
                        f"{key}={value} not in allowed values: {schema['allowed']}"
                    )
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid type for {key}: {e}")
        
        validated[key] = value
    
    return validated


def validate_agent_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate agent output against strict schema.
    
    Required fields:
    - recommended_params: dict of numeric parameters
    - confidence: float 0-1
    - reasoning: string
    
    Optional fields:
    - flags: list of strings
    - metadata: any additional data
    
    Args:
        output: Raw agent output dictionary
        
    Returns:
        Validated output with all required fields
        
    Raises:
        ValidationError: If required fields missing or invalid
    """
    if not isinstance(output, dict):
        raise ValidationError(f"Agent output must be dict, got {type(output)}")
    
    # Check required fields
    if "recommended_params" not in output:
        raise ValidationError("Missing required field: recommended_params")
    
    # Validate recommended_params
    try:
        validated_params = validate_params(output["recommended_params"])
    except ValidationError as e:
        raise ValidationError(f"Invalid recommended_params: {e}")
    
    # Validate confidence (optional, default 0.5)
    confidence = output.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5
    
    # Validate reasoning (optional)
    reasoning = output.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    
    # Validate flags (optional)
    flags = output.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(f) for f in flags]
    
    return {
        "recommended_params": validated_params,
        "confidence": confidence,
        "reasoning": reasoning,
        "flags": flags,
        "metadata": output.get("metadata", {}),
    }


def validate_image_record(record: Dict[str, Any]) -> None:
    """
    Validate an image record in the manifest.
    
    Args:
        record: Image record dictionary
        
    Raises:
        ValidationError: If record is invalid
    """
    required_fields = ["image_id", "image_path", "status"]
    
    for field in required_fields:
        if field not in record:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate status
    status = record["status"]
    if status not in VALID_STATUSES:
        raise ValidationError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")
    
    # Validate recommended_params if present
    if "recommended_params" in record and record["recommended_params"] is not None:
        validate_params(record["recommended_params"])
    
    # Validate final_params if present
    if "final_params" in record and record["final_params"] is not None:
        validate_params(record["final_params"])


def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clamp all parameters to their valid ranges without raising errors.
    Use this for user-edited params that might be out of range.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Clamped parameters
    """
    try:
        return validate_params(params)
    except ValidationError:
        # If validation fails completely, return empty dict
        return {}


