"""
Manifest manager for preprocessing parameters.
Provides atomic, crash-safe manifest updates.
"""

import json
import os
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

from .schema import validate_image_record, validate_agent_output, ValidationError


def generate_image_id(image_path: str, frame_index: Optional[int] = None) -> str:
    """
    Generate a stable image ID from file path and optional frame index.
    Uses SHA256 hash of (path + frame_index).
    
    Args:
        image_path: Path to image file
        frame_index: Optional frame index for multi-frame images
        
    Returns:
        Unique image ID (hex string)
    """
    key = image_path
    if frame_index is not None:
        key = f"{image_path}:frame_{frame_index}"
    
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]


def generate_run_id() -> str:
    """
    Generate a unique run ID for this preprocessing session.
    Format: timestamp_uuid
    
    Returns:
        Unique run ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid4())[:8]
    return f"{timestamp}_{unique_id}"


class ManifestManager:
    """
    Manages preprocessing manifest with atomic writes.
    
    The manifest is a JSON file that tracks preprocessing parameters
    for each image in a run.
    """
    
    def __init__(self, run_id: str, base_dir: str = "outputs/runs"):
        """
        Initialize manifest manager.
        
        Args:
            run_id: Unique run identifier
            base_dir: Base directory for runs (default: outputs/runs)
        """
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.manifest_path = self.run_dir / "preprocess_manifest.json"
        
        # Create run directory if it doesn't exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize manifest
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load manifest from disk, or create new one if it doesn't exist.
        
        Returns:
            Manifest dictionary
        """
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                return manifest
            except (json.JSONDecodeError, IOError) as e:
                # Corrupt manifest - back it up and create new one
                backup_path = self.manifest_path.with_suffix('.json.backup')
                if self.manifest_path.exists():
                    self.manifest_path.rename(backup_path)
                print(f"Warning: Corrupt manifest backed up to {backup_path}")
                return self._create_new_manifest()
        else:
            return self._create_new_manifest()
    
    def _create_new_manifest(self) -> Dict[str, Any]:
        """
        Create a new empty manifest.
        
        Returns:
            New manifest dictionary
        """
        return {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0",
            "images": {},
        }
    
    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """
        Atomically write manifest to disk using temp file + rename.
        This prevents corruption if the process crashes mid-write.
        
        Args:
            data: Manifest data to write
        """
        # Update timestamp
        data["updated_at"] = datetime.now().isoformat()
        
        # Write to temporary file in the same directory
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='manifest_',
            dir=self.run_dir,
            text=True
        )
        
        try:
            # Write JSON with pretty formatting
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            
            # Atomic rename (on POSIX systems)
            # On Windows, need to remove target first
            if os.name == 'nt' and self.manifest_path.exists():
                self.manifest_path.unlink()
            
            os.rename(temp_path, self.manifest_path)
            
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise IOError(f"Failed to write manifest: {e}")
    
    def get_image_record(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get record for a specific image.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            Image record or None if not found
        """
        return self.manifest["images"].get(image_id)
    
    def get_all_images(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all image records.
        
        Returns:
            Dictionary of image_id -> record
        """
        return self.manifest["images"]
    
    def update_image_record(
        self,
        image_id: str,
        image_path: str,
        status: str = "recommended",
        recommended_params: Optional[Dict[str, Any]] = None,
        final_params: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        confidence: Optional[float] = None,
        flags: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update or create an image record in the manifest.
        
        Args:
            image_id: Unique image identifier
            image_path: Path to image file
            status: Processing status
            recommended_params: Agent-recommended parameters
            final_params: User-approved/edited parameters
            reasoning: Agent reasoning
            confidence: Confidence score (0-1)
            flags: List of flags
            error_message: Error message if status=failed
            agent_metadata: Additional agent metadata
        """
        # Get existing record or create new one
        if image_id in self.manifest["images"]:
            record = self.manifest["images"][image_id]
            
            # Append to history if recommended_params changed
            if recommended_params is not None:
                if "history" not in record:
                    record["history"] = []
                
                # Only add to history if different from current
                if record.get("recommended_params") != recommended_params:
                    record["history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "recommended_params": record.get("recommended_params"),
                        "reasoning": record.get("reasoning"),
                        "confidence": record.get("confidence"),
                    })
        else:
            record = {
                "image_id": image_id,
                "image_path": image_path,
                "created_at": datetime.now().isoformat(),
                "history": [],
            }
        
        # Update fields
        record["status"] = status
        record["updated_at"] = datetime.now().isoformat()
        
        if recommended_params is not None:
            record["recommended_params"] = recommended_params
        if final_params is not None:
            record["final_params"] = final_params
        if reasoning is not None:
            record["reasoning"] = reasoning
        if confidence is not None:
            record["confidence"] = confidence
        if flags is not None:
            record["flags"] = flags
        if error_message is not None:
            record["error_message"] = error_message
        if agent_metadata is not None:
            record["agent_metadata"] = agent_metadata
        
        # Validate record
        try:
            validate_image_record(record)
        except ValidationError as e:
            raise ValueError(f"Invalid record for {image_id}: {e}")
        
        # Update manifest
        self.manifest["images"][image_id] = record
        
        # Write atomically
        self._atomic_write(self.manifest)
    
    def save_agent_recommendation(
        self,
        image_id: str,
        image_path: str,
        agent_output: Dict[str, Any],
        agent_model: str = "gpt-4o",
        prompt_version: str = "1.0",
    ) -> None:
        """
        Save agent recommendation to manifest.
        Validates agent output and creates/updates image record.
        
        Args:
            image_id: Unique image identifier
            image_path: Path to image file
            agent_output: Raw agent output (will be validated)
            agent_model: Model used for recommendation
            prompt_version: Version of prompt used
            
        Raises:
            ValidationError: If agent output is invalid
        """
        try:
            # Validate agent output
            validated = validate_agent_output(agent_output)
            
            # Update record
            self.update_image_record(
                image_id=image_id,
                image_path=image_path,
                status="recommended",
                recommended_params=validated["recommended_params"],
                reasoning=validated["reasoning"],
                confidence=validated["confidence"],
                flags=validated["flags"],
                agent_metadata={
                    "model": agent_model,
                    "prompt_version": prompt_version,
                    "timestamp": datetime.now().isoformat(),
                    **validated.get("metadata", {}),
                },
            )
            
        except ValidationError as e:
            # Save error record
            self.update_image_record(
                image_id=image_id,
                image_path=image_path,
                status="failed",
                error_message=f"Agent output validation failed: {e}",
            )
            raise
    
    def approve_params(
        self,
        image_id: str,
        final_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Approve parameters for an image.
        If final_params not provided, uses recommended_params.
        
        Args:
            image_id: Unique image identifier
            final_params: Final approved parameters (optional)
        """
        record = self.get_image_record(image_id)
        if not record:
            raise ValueError(f"Image {image_id} not found in manifest")
        
        # Use recommended_params if final_params not provided
        if final_params is None:
            final_params = record.get("recommended_params")
        
        if not final_params:
            raise ValueError(f"No parameters to approve for {image_id}")
        
        # Determine status
        status = "edited" if final_params != record.get("recommended_params") else "approved"
        
        self.update_image_record(
            image_id=image_id,
            image_path=record["image_path"],
            status=status,
            final_params=final_params,
        )
    
    def mark_applied(self, image_id: str) -> None:
        """
        Mark parameters as applied for an image.
        
        Args:
            image_id: Unique image identifier
        """
        record = self.get_image_record(image_id)
        if not record:
            raise ValueError(f"Image {image_id} not found in manifest")
        
        self.update_image_record(
            image_id=image_id,
            image_path=record["image_path"],
            status="applied",
        )
    
    def mark_failed(self, image_id: str, error_message: str) -> None:
        """
        Mark an image as failed with error message.
        
        Args:
            image_id: Unique image identifier
            error_message: Error description
        """
        record = self.get_image_record(image_id)
        if not record:
            raise ValueError(f"Image {image_id} not found in manifest")
        
        self.update_image_record(
            image_id=image_id,
            image_path=record["image_path"],
            status="failed",
            error_message=error_message,
        )
    
    def get_params_for_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get parameters to use for an image.
        Prefers final_params, falls back to recommended_params.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            Parameters dictionary or None
        """
        record = self.get_image_record(image_id)
        if not record:
            return None
        
        # Prefer final_params (user-approved), fallback to recommended
        return record.get("final_params") or record.get("recommended_params")


