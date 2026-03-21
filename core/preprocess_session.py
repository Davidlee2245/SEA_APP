"""
Preprocessing Session Management
Handles persistent storage of preprocessing configurations and pipeline state
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import OrderedDict


class PreprocessingSession:
    """
    Manages preprocessing session state for a (sample, position) pair.
    Stores pipeline configuration, parameters, and generates version hashes.
    """
    
    def __init__(self, sample: str, position: str, cache_root: Path = Path("data/cache")):
        self.sample = sample
        self.position = position
        self.cache_root = cache_root
        self.session_dir = cache_root / "preprocess_sessions" / sample / position
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / "preprocess.json"
        self.log_file = self.session_dir / "operation_log.jsonl"
        
        # Session data structure
        self.session_data: Dict[str, Any] = {
            'sample': sample,
            'position': position,
            'created_at': None,
            'updated_at': None,
            'version_hash': None,
            'pipeline': [],  # Ordered list of steps applied
            'channel_params': {},  # Per-channel parameters
            'global_params': {},  # Global parameters
            'output_stages': {},  # Map of stage -> output file paths
        }
        
        # Load existing session if available
        self.load()
    
    def load(self) -> bool:
        """Load session from disk if it exists."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    self.session_data = json.load(f)
                print(f"[Session] Loaded session for {self.sample}/{self.position}")
                return True
            except Exception as e:
                print(f"[Session] Failed to load session: {e}")
                return False
        return False
    
    def save(self):
        """Save session to disk."""
        self.session_data['updated_at'] = datetime.now().isoformat()
        if not self.session_data.get('created_at'):
            self.session_data['created_at'] = datetime.now().isoformat()
        
        # Compute version hash
        self.session_data['version_hash'] = self.compute_version_hash()
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            print(f"[Session] Saved session for {self.sample}/{self.position}")
        except Exception as e:
            print(f"[Session] Failed to save session: {e}")
    
    def compute_version_hash(self) -> str:
        """
        Compute a hash of the preprocessing pipeline configuration.
        This hash changes whenever steps or parameters change.
        """
        # Create a deterministic representation of the pipeline
        pipeline_repr = {
            'pipeline': self.session_data['pipeline'],
            'channel_params': self.session_data['channel_params'],
            'global_params': self.session_data['global_params'],
        }
        
        # Convert to JSON string (sorted keys for determinism)
        pipeline_str = json.dumps(pipeline_repr, sort_keys=True)
        
        # Compute hash
        return hashlib.sha256(pipeline_str.encode()).hexdigest()[:16]
    
    def add_step(self, step_name: str, step_params: Dict[str, Any], 
                 input_stage: str, output_stage: str, channel_params: Optional[Dict[str, Any]] = None):
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            step_name: Name of the step (e.g., 'contrast_enhance', 'step1')
            step_params: Parameters for this step
            input_stage: Input stage name
            output_stage: Output stage name
            channel_params: Optional per-channel parameters
        """
        step_entry = {
            'step_name': step_name,
            'step_params': step_params,
            'input_stage': input_stage,
            'output_stage': output_stage,
            'timestamp': datetime.now().isoformat(),
        }
        
        if channel_params:
            step_entry['channel_params'] = channel_params
        
        # Add to pipeline (maintain order)
        self.session_data['pipeline'].append(step_entry)
        
        # Update output stages
        self.session_data['output_stages'][output_stage] = {
            'step_name': step_name,
            'timestamp': step_entry['timestamp'],
        }
        
        # Log the operation
        self.log_operation('add_step', step_name, step_params, input_stage, output_stage)
        
        # Save session
        self.save()
    
    def log_operation(self, action: str, step_name: str, params: Dict[str, Any],
                     input_stage: str, output_stage: str):
        """Append an operation log entry."""
        log_entry = {
            'time': datetime.now().isoformat(),
            'user_action': action,
            'step_name': step_name,
            'params': params,
            'input_stage': input_stage,
            'output_stage': output_stage,
            'version_hash': self.compute_version_hash(),
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"[Session] Failed to log operation: {e}")
    
    def get_final_stage(self) -> Optional[str]:
        """Get the final output stage name (last step in pipeline)."""
        if not self.session_data['pipeline']:
            return None
        return self.session_data['pipeline'][-1]['output_stage']
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the preprocessing pipeline."""
        return {
            'sample': self.sample,
            'position': self.position,
            'version_hash': self.session_data.get('version_hash'),
            'num_steps': len(self.session_data['pipeline']),
            'pipeline': self.session_data['pipeline'],
            'final_stage': self.get_final_stage(),
            'channel_params': self.session_data.get('channel_params', {}),
            'global_params': self.session_data.get('global_params', {}),
            'created_at': self.session_data.get('created_at'),
            'updated_at': self.session_data.get('updated_at'),
        }
    
    def clear(self):
        """Clear the session (reset to empty)."""
        self.session_data = {
            'sample': self.sample,
            'position': self.position,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version_hash': None,
            'pipeline': [],
            'channel_params': {},
            'global_params': {},
            'output_stages': {},
        }
        self.save()


def get_session(sample: str, position: str, cache_root: Path = Path("data/cache")) -> PreprocessingSession:
    """Get or create a preprocessing session for a (sample, position) pair."""
    return PreprocessingSession(sample, position, cache_root)

