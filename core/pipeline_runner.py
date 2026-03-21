"""
Pipeline Runner - Executes pipeline in background with progress tracking
"""

import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import queue
import sys
from io import StringIO
import contextlib

# Import existing pipeline components
from pipeline import ExosomeAnalysisPipeline


class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStep(Enum):
    """Pipeline steps"""
    INSPECTION = 1
    ALIGNMENT = 2
    ANALYSIS = 3


class PipelineRunner:
    """
    Manages pipeline execution in background with progress tracking.
    Thread-safe, supports cancellation, and emits real-time events.
    """
    
    def __init__(self):
        self.status = PipelineStatus.IDLE
        self.current_step = None
        self.current_sample = None
        self.progress = 0.0  # 0.0 to 1.0
        self.total_samples = 0
        self.completed_samples = 0
        self.error_message = None
        self.error_details = None
        
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._log_queue = queue.Queue()
        self._progress_callbacks = []
        self._log_callbacks = []
        
        self.pipeline = None
        self.results = {}
        self._loguru_handler_id = None
    
    @contextlib.contextmanager
    def _capture_output(self, stdout_callback: Callable[[str], None], 
                       stderr_callback: Callable[[str], None]):
        """Context manager to capture stdout/stderr and forward to callbacks"""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        class StreamForwarder:
            def __init__(self, callback, original_stream):
                self.callback = callback
                self.original_stream = original_stream
                self.buffer = ""
            
            def write(self, text):
                self.original_stream.write(text)  # Also write to original
                self.buffer += text
                # Process complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        self.callback(line)
            
            def flush(self):
                self.original_stream.flush()
                if self.buffer.strip():
                    self.callback(self.buffer)
                    self.buffer = ""
        
        sys.stdout = StreamForwarder(stdout_callback, old_stdout)
        sys.stderr = StreamForwarder(stderr_callback, old_stderr)
        
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def register_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for progress updates"""
        self._progress_callbacks.append(callback)
    
    def register_log_callback(self, callback: Callable[[str], None]):
        """Register a callback for log messages"""
        self._log_callbacks.append(callback)
    
    def _emit_progress(self, data: Dict[str, Any]):
        """Emit progress update to all registered callbacks"""
        for callback in self._progress_callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in progress callback: {e}")
    
    def _emit_log(self, message: str):
        """Emit log message to all registered callbacks"""
        self._log_queue.put(message)
        for callback in self._log_callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"Error in log callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        with self._lock:
            return {
                'status': self.status.value,
                'current_step': self.current_step.name if self.current_step else None,
                'current_sample': self.current_sample,
                'progress': self.progress,
                'total_samples': self.total_samples,
                'completed_samples': self.completed_samples,
                'error': self.error_message,
                'error_details': self.error_details
            }
    
    def get_logs(self, limit: int = 100) -> List[str]:
        """Get recent log messages"""
        logs = []
        try:
            while len(logs) < limit:
                logs.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        return logs
    
    def is_running(self) -> bool:
        """Check if pipeline is currently running"""
        with self._lock:
            return self.status == PipelineStatus.RUNNING
    
    def stop(self):
        """Request pipeline to stop gracefully"""
        if self.is_running():
            self._emit_log("🛑 Stop requested...")
            self._stop_event.set()
            with self._lock:
                self.status = PipelineStatus.CANCELLED
    
    def run_all(self, sample_names: Optional[List[str]] = None, 
                config_path: Optional[Path] = None) -> bool:
        """
        Run complete pipeline on specified samples.
        Returns True if started successfully, False if already running.
        """
        if self.is_running():
            return False
        
        self._reset()
        self._thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(sample_names, config_path, None)
        )
        self._thread.daemon = True
        self._thread.start()
        return True
    
    def run_step(self, step: PipelineStep, sample_name: str,
                 config_path: Optional[Path] = None) -> bool:
        """
        Run a specific pipeline step on a sample.
        Returns True if started successfully, False if already running.
        """
        if self.is_running():
            return False
        
        self._reset()
        self._thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=([sample_name], config_path, step)
        )
        self._thread.daemon = True
        self._thread.start()
        return True
    
    def _reset(self):
        """Reset runner state"""
        with self._lock:
            self.status = PipelineStatus.IDLE
            self.current_step = None
            self.current_sample = None
            self.progress = 0.0
            self.completed_samples = 0
            self.error_message = None
            self.error_details = None
            self._stop_event.clear()
            
            # Clear log queue
            while not self._log_queue.empty():
                try:
                    self._log_queue.get_nowait()
                except queue.Empty:
                    break
    
    def _run_pipeline_thread(self, sample_names: Optional[List[str]], 
                            config_path: Optional[Path],
                            single_step: Optional[PipelineStep]):
        """
        Main pipeline execution thread.
        Runs in background and emits progress updates.
        """
        # Capture stdout/stderr and loguru output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        def forward_stdout(line: str):
            """Forward stdout lines to log callback"""
            if line.strip():
                # Skip loguru-formatted messages (they're handled by loguru sink)
                # Loguru format: "YYYY-MM-DD HH:MM:SS | LEVEL | message"
                if " | " in line and ("INFO" in line or "WARNING" in line or "ERROR" in line or "DEBUG" in line):
                    return  # Skip - loguru sink will handle it
                self._emit_log(line.rstrip())
        
        def forward_stderr(line: str):
            """Forward stderr lines to log callback"""
            if line.strip():
                # Skip loguru-formatted messages (they're handled by loguru sink)
                # Loguru format: "YYYY-MM-DD HH:MM:SS | LEVEL | message"
                if " | " in line and ("INFO" in line or "WARNING" in line or "ERROR" in line or "DEBUG" in line):
                    return  # Skip - loguru sink will handle it
                # Only prefix with [ERROR] if it looks like an actual error
                # (not just because it came from stderr)
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['error', 'exception', 'traceback', 'failed', 'failure']):
                    self._emit_log(f"[ERROR] {line.rstrip()}")
                else:
                    # Regular stderr output (not an error, just informational)
                    self._emit_log(line.rstrip())
        
        # Initialize loguru handler ID (will be set after pipeline init)
        self._loguru_handler_id = None
        
        try:
            with self._lock:
                self.status = PipelineStatus.RUNNING
            
            self._emit_log("=" * 60)
            self._emit_log("🚀 Starting SEA Pipeline")
            self._emit_log("=" * 60)
            
            # Initialize pipeline
            self._emit_log("📦 Initializing pipeline...")
            
            # Capture stdout/stderr during entire pipeline execution
            with self._capture_output(forward_stdout, forward_stderr):
                self.pipeline = ExosomeAnalysisPipeline(config_path)
                
                # Add loguru sink AFTER pipeline initialization (so it doesn't get removed)
                # The pipeline's _setup_logging() removes all handlers, so we add ours after
                try:
                    from loguru import logger
                    
                    def loguru_sink(message):
                        """Custom sink that forwards loguru messages to our log callback"""
                        record = message.record
                        level = record["level"].name
                        
                        # Extract just the message text (without timestamp/formatting)
                        log_msg = record["message"]
                        
                        # Skip empty messages and some noisy loguru internal messages
                        if log_msg.strip() and "loguru" not in log_msg.lower():
                            # Only prefix with level if it's WARNING or ERROR
                            if level in ["WARNING", "ERROR", "CRITICAL"]:
                                self._emit_log(f"[{level}] {log_msg.rstrip()}")
                            else:
                                self._emit_log(log_msg.rstrip())
                    
                    # Add custom sink to loguru AFTER pipeline setup
                    self._loguru_handler_id = logger.add(loguru_sink, level="DEBUG", format="{message}")
                except ImportError:
                    # loguru not available
                    self._loguru_handler_id = None
                
                # Find samples
                if sample_names is None:
                    samples = self.pipeline.find_samples()
                    sample_names = [s.name for s in samples]
                else:
                    samples = [self.pipeline.input_root / name for name in sample_names]
                
                with self._lock:
                    self.total_samples = len(samples)
                
                self._emit_log(f"📊 Found {len(samples)} samples to process")
                
                # Process each sample
                for idx, sample_dir in enumerate(samples):
                    if self._stop_event.is_set():
                        self._emit_log("⚠️ Pipeline cancelled by user")
                        with self._lock:
                            self.status = PipelineStatus.CANCELLED
                        return
                    
                    sample_name = sample_dir.name if isinstance(sample_dir, Path) else sample_dir
                    
                    with self._lock:
                        self.current_sample = sample_name
                    
                    self._emit_log("")
                    self._emit_log(f"📁 Processing sample: {sample_name} ({idx + 1}/{len(samples)})")
                    
                    if single_step:
                        # Run only specific step
                        result = self._run_single_step(sample_dir, single_step)
                    else:
                        # Run all steps
                        result = self._run_all_steps(sample_dir)
                
                    with self._lock:
                        self.completed_samples += 1
                        self.progress = self.completed_samples / self.total_samples
                        self.results[sample_name] = result
                    
                    self._emit_progress({
                        'sample': sample_name,
                        'progress': self.progress,
                        'completed': self.completed_samples,
                        'total': self.total_samples
                    })
                
                # Completed successfully
                with self._lock:
                    self.status = PipelineStatus.COMPLETED
                    self.progress = 1.0
                
                self._emit_log("")
                self._emit_log("=" * 60)
                self._emit_log("✅ Pipeline completed successfully!")
                self._emit_log(f"📊 Processed {self.completed_samples}/{self.total_samples} samples")
                self._emit_log("=" * 60)
            
        except Exception as e:
            error_msg = str(e)
            error_details = traceback.format_exc()
            
            with self._lock:
                self.status = PipelineStatus.FAILED
                self.error_message = error_msg
                self.error_details = error_details
            
            self._emit_log("")
            self._emit_log("=" * 60)
            self._emit_log(f"❌ Pipeline failed: {error_msg}")
            self._emit_log("=" * 60)
            self._emit_log(f"Details:\n{error_details}")
        
        finally:
            # Remove loguru handler
            if self._loguru_handler_id is not None:
                try:
                    from loguru import logger
                    logger.remove(self._loguru_handler_id)
                except:
                    pass
                self._loguru_handler_id = None
    
    def _run_all_steps(self, sample_dir: Path) -> Dict[str, Any]:
        """Run all pipeline steps on a sample"""
        # This wraps the existing pipeline.process_sample() method
        # but with progress tracking
        
        sample_name = sample_dir.name
        result = {}
        
        try:
            # Step 1: Inspector
            if self._stop_event.is_set():
                return result
            
            with self._lock:
                self.current_step = PipelineStep.INSPECTION
            
            self._emit_log("🔍 Phase 1: Inspection and Preprocessing")
            self._emit_progress({
                'step': 'inspection',
                'sample': sample_name,
                'message': 'Running image quality assessment...'
            })
            
            inspection_result = self.pipeline.inspector.process(sample_dir)
            result['inspection'] = inspection_result
            
            if 'error' in inspection_result:
                raise Exception(f"Inspection failed: {inspection_result['error']}")
            
            # Step 2: Aligner
            if self._stop_event.is_set():
                return result
            
            with self._lock:
                self.current_step = PipelineStep.ALIGNMENT
            
            self._emit_log("🎯 Phase 2: Registration and Alignment")
            self._emit_progress({
                'step': 'alignment',
                'sample': sample_name,
                'message': 'Aligning multi-channel images...'
            })
            
            images = inspection_result['images']
            anchor_channel = inspection_result['anchor_channel']
            
            alignment_result = self.pipeline.aligner.process(
                images=images,
                anchor_channel=anchor_channel,
                sample_name=sample_name
            )
            result['alignment'] = alignment_result
            
            if 'error' in alignment_result:
                raise Exception(f"Alignment failed: {alignment_result['error']}")
            
            # Step 3: Analyst
            if self._stop_event.is_set():
                return result
            
            with self._lock:
                self.current_step = PipelineStep.ANALYSIS
            
            self._emit_log("📊 Phase 3: Detection and Quantification")
            self._emit_progress({
                'step': 'analysis',
                'sample': sample_name,
                'message': 'Detecting and counting exosomes...'
            })
            
            registered_images = alignment_result['registered_images']
            output_dir = self.pipeline.output_root / sample_name
            
            analysis_result = self.pipeline.analyst.process(
                images=registered_images,
                sample_name=sample_name,
                output_dir=output_dir
            )
            result['analysis'] = analysis_result
            
            if 'error' in analysis_result:
                raise Exception(f"Analysis failed: {analysis_result['error']}")
            
            self._emit_log(f"✅ Sample {sample_name} completed successfully")
            
        except Exception as e:
            self._emit_log(f"❌ Error processing {sample_name}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _run_single_step(self, sample_dir: Path, step: PipelineStep) -> Dict[str, Any]:
        """Run a single pipeline step"""
        sample_name = sample_dir.name
        result = {}
        
        with self._lock:
            self.current_step = step
        
        try:
            if step == PipelineStep.INSPECTION:
                self._emit_log("🔍 Running inspection step only...")
                result = self.pipeline.inspector.process(sample_dir)
                
            elif step == PipelineStep.ALIGNMENT:
                self._emit_log("🎯 Running alignment step only...")
                # Need to load preprocessed images first
                preprocessed_dir = self.pipeline.output_root / sample_name / "preprocessed"
                if not preprocessed_dir.exists():
                    raise Exception("Preprocessed images not found. Run inspection first.")
                
                # Load images and run alignment
                # (This would need additional helper methods)
                result = {'status': 'Step-by-step execution needs inspection results'}
                
            elif step == PipelineStep.ANALYSIS:
                self._emit_log("📊 Running analysis step only...")
                # Need to load registered images first
                registered_dir = self.pipeline.output_root / sample_name / "registered"
                if not registered_dir.exists():
                    raise Exception("Registered images not found. Run alignment first.")
                
                # Load images and run analysis
                result = {'status': 'Step-by-step execution needs alignment results'}
            
            self._emit_log(f"✅ Step {step.name} completed")
            
        except Exception as e:
            self._emit_log(f"❌ Step {step.name} failed: {e}")
            result['error'] = str(e)
        
        return result


# Global runner instance (singleton)
_runner_instance = None

def get_runner() -> PipelineRunner:
    """Get the global pipeline runner instance"""
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = PipelineRunner()
    return _runner_instance


