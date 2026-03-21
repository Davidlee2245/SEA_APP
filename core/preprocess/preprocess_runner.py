"""
Batch preprocessing runner with progress tracking and cancellation support.
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from threading import Event
import traceback

from core.manifest import ManifestManager
from .preprocess_engine import apply_preprocessing, PreprocessingError


class PreprocessRunner:
    """
    Runs preprocessing on all images in a manifest with progress tracking.
    """
    
    def __init__(
        self,
        manifest_manager: ManifestManager,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize preprocessing runner.
        
        Args:
            manifest_manager: ManifestManager instance
            progress_callback: Called with (current, total, image_id) after each image
            log_callback: Called with log messages
        """
        self.manifest = manifest_manager
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.cancel_event = Event()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _log(self, message: str, level: str = "info"):
        """Log a message via callback and logger."""
        if self.log_callback:
            self.log_callback(message)
        
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _progress(self, current: int, total: int, image_id: str):
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback(current, total, image_id)
    
    def cancel(self):
        """Request cancellation of the current run."""
        self.cancel_event.set()
        self._log("Cancellation requested", "warning")
    
    def run(
        self,
        status_filter: Optional[list] = None,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Run preprocessing on images in the manifest.
        
        Args:
            status_filter: Only process images with these statuses (default: recommended, approved, edited)
            force_reprocess: If True, reprocess even if status=applied
            
        Returns:
            Summary dictionary with counts of success/failure/cancelled
        """
        # Reset cancel event
        self.cancel_event.clear()
        
        # Default status filter
        if status_filter is None:
            status_filter = ["recommended", "approved", "edited"]
        
        # Get all images
        all_images = self.manifest.get_all_images()
        
        # Filter images
        images_to_process = []
        for image_id, record in all_images.items():
            status = record.get("status")
            
            if force_reprocess:
                # Reprocess all images that have params
                if self.manifest.get_params_for_image(image_id):
                    images_to_process.append((image_id, record))
            else:
                # Only process images in the specified statuses
                if status in status_filter:
                    images_to_process.append((image_id, record))
        
        total = len(images_to_process)
        
        if total == 0:
            self._log("No images to process", "warning")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "cancelled": 0,
                "skipped": 0,
            }
        
        self._log(f"Starting preprocessing for {total} images")
        
        # Process each image
        success_count = 0
        failed_count = 0
        cancelled_count = 0
        
        for idx, (image_id, record) in enumerate(images_to_process, 1):
            # Check for cancellation
            if self.cancel_event.is_set():
                self._log(f"Cancelled at image {idx}/{total}", "warning")
                cancelled_count = total - idx + 1
                
                # Mark remaining as cancelled
                for remaining_id, _ in images_to_process[idx-1:]:
                    self.manifest.update_image_record(
                        image_id=remaining_id,
                        image_path=self.manifest.get_image_record(remaining_id)["image_path"],
                        status="cancelled",
                    )
                break
            
            # Report progress
            self._progress(idx, total, image_id)
            self._log(f"[{idx}/{total}] Processing {image_id}")
            
            # Get parameters
            params = self.manifest.get_params_for_image(image_id)
            
            if not params:
                self._log(f"  ⚠️  No parameters found for {image_id}, skipping", "warning")
                failed_count += 1
                self.manifest.mark_failed(image_id, "No parameters found")
                continue
            
            # Get image path
            image_path = record["image_path"]
            
            # Determine output path
            output_dir = self.manifest.run_dir / "preprocessed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Preserve extension or use PNG
            input_ext = Path(image_path).suffix
            if input_ext.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                output_ext = input_ext
            else:
                output_ext = '.png'
            
            output_path = output_dir / f"{image_id}{output_ext}"
            
            # Apply preprocessing
            try:
                self._log(f"  📝 Applying preprocessing with params: {list(params.keys())}")
                
                apply_preprocessing(
                    image_path=image_path,
                    params=params,
                    output_path=str(output_path),
                )
                
                # Mark as applied
                self.manifest.mark_applied(image_id)
                success_count += 1
                
                self._log(f"  ✅ Success: {output_path}")
                
            except PreprocessingError as e:
                self._log(f"  ❌ Failed: {e}", "error")
                self.manifest.mark_failed(image_id, str(e))
                failed_count += 1
                
            except Exception as e:
                error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
                self._log(f"  ❌ Failed: {error_msg}", "error")
                self.manifest.mark_failed(image_id, error_msg)
                failed_count += 1
        
        # Final summary
        summary = {
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "cancelled": cancelled_count,
            "skipped": 0,
        }
        
        self._log("\n" + "="*50)
        self._log(f"Preprocessing complete!")
        self._log(f"  Total: {total}")
        self._log(f"  ✅ Success: {success_count}")
        self._log(f"  ❌ Failed: {failed_count}")
        self._log(f"  🚫 Cancelled: {cancelled_count}")
        self._log("="*50)
        
        return summary
    
    def run_single(self, image_id: str) -> bool:
        """
        Run preprocessing on a single image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if successful, False otherwise
        """
        record = self.manifest.get_image_record(image_id)
        if not record:
            self._log(f"Image {image_id} not found", "error")
            return False
        
        params = self.manifest.get_params_for_image(image_id)
        if not params:
            self._log(f"No parameters found for {image_id}", "error")
            self.manifest.mark_failed(image_id, "No parameters found")
            return False
        
        image_path = record["image_path"]
        
        output_dir = self.manifest.run_dir / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_ext = Path(image_path).suffix
        output_ext = input_ext if input_ext.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff'] else '.png'
        output_path = output_dir / f"{image_id}{output_ext}"
        
        try:
            self._log(f"Processing {image_id}")
            
            apply_preprocessing(
                image_path=image_path,
                params=params,
                output_path=str(output_path),
            )
            
            self.manifest.mark_applied(image_id)
            self._log(f"✅ Success: {output_path}")
            return True
            
        except PreprocessingError as e:
            self._log(f"❌ Failed: {e}", "error")
            self.manifest.mark_failed(image_id, str(e))
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
            self._log(f"❌ Failed: {error_msg}", "error")
            self.manifest.mark_failed(image_id, error_msg)
            return False


