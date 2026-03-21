"""
Extended Flask API Server with Pipeline Control
Adds WebSocket for real-time logs and pipeline execution endpoints
"""

from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS
from flask_sock import Sock
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import tifffile
from io import BytesIO
from PIL import Image
import threading
import time
import csv
from collections import defaultdict
import hashlib
import re
import cv2
from skimage import exposure
import pandas as pd

# Import openpyxl explicitly - pandas needs this for .xlsx files
try:
    import openpyxl
    _openpyxl_available = True
except ImportError:
    print("=" * 60)
    print("ERROR: openpyxl is not installed!")
    print("Please install it with: pip install openpyxl")
    print("Or: python3 -m pip install openpyxl")
    print("=" * 60)
    _openpyxl_available = False

# Import the existing api_server functions
import sys
import argparse
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline_runner import get_runner, PipelineStatus, PipelineStep
from core.preprocess_session import get_session

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Configuration — paths are placeholders; overwritten in __main__ after
# --data-root is parsed.  Do NOT call .mkdir() here: when running from an
# AppImage the working directory is a read-only squashfs mount.
import os
OUTPUT_ROOT       = Path("data/output")
INPUT_ROOT        = Path("data/input")
DATA_ROOT         = Path(os.getenv('SEA_DATA_ROOT', 'data/input'))
PREVIEW_CACHE     = Path("previews")
PROCESSING_OUTPUT = Path("data/processing")

# Global state for WebSocket clients
ws_clients = set()
ws_lock = threading.Lock()

# Session state: Track preprocessing stages per position
# Structure: {f"{sample}/{position}": {"raw": {...}, "contrast": {...}, "step1": {...}, ...}}
preprocessing_cache: Dict[str, Dict[str, Dict[str, Path]]] = {}


def restore_preprocessing_cache_from_disk(sample_name: str, position_name: str) -> bool:
    """
    Restore preprocessing cache from disk by checking for existing processed files.
    This is called when loading a position to restore cache after server restart.
    
    Returns:
        True if cache was restored, False otherwise
    """
    position_key = f"{sample_name}/{position_name}"
    
    # Check if already in cache
    if position_key in preprocessing_cache and len(preprocessing_cache[position_key]) > 1:
        # Already has more than just raw, assume it's loaded
        return True
    
    try:
        # Get session to find what steps were applied
        session = get_session(sample_name, position_name)
        
        if not session.session_data.get('pipeline'):
            # No preprocessing done
            return False
        
        print(f"[Cache Restore] Restoring cache for {position_key} from disk...")
        
        # Initialize cache entry if needed
        if position_key not in preprocessing_cache:
            preprocessing_cache[position_key] = {}
        
        # Get raw channels first (they should be in cache from load_position)
        if 'raw' not in preprocessing_cache[position_key]:
            print(f"[Cache Restore] WARNING: Raw channels not in cache, cannot restore")
            return False
        
        raw_channels = preprocessing_cache[position_key]['raw']
        
        # Restore each step from disk
        restored_steps = []
        channel_states = {}
        
        for step_entry in session.session_data['pipeline']:
            step = step_entry['step_name']
            output_stage = step_entry.get('output_stage', step)
            
            # Check if processed files exist for this step
            step_dir = PROCESSING_OUTPUT / sample_name / position_name / step
            if not step_dir.exists():
                print(f"[Cache Restore] Step {step} directory not found: {step_dir}")
                continue
            
            # Find all .tif files in the step directory
            output_files = {}
            for tif_file in step_dir.glob("*.tif"):
                # Extract channel name from filename (e.g., "C1_ch1.tif" -> "C1_ch1")
                channel_name = tif_file.stem
                if channel_name in raw_channels:
                    output_files[channel_name] = tif_file
                    print(f"[Cache Restore] Found {step}/{channel_name}: {tif_file}")
            
            if output_files:
                # Restore to cache
                preprocessing_cache[position_key][step] = output_files
                restored_steps.append(step)
                
                # Restore channel states
                channel_params = step_entry.get('channel_params', {})
                for channel_name in output_files.keys():
                    if channel_name not in channel_states:
                        channel_states[channel_name] = {}
                    
                    channel_states[channel_name][step] = {
                        'step': step,
                        'from_stage': step_entry.get('input_stage', 'raw'),
                        'params': channel_params.get(channel_name, step_entry.get('step_params', {})),
                        'output_path': str(output_files[channel_name])
                    }
        
        # Update channel_states in cache
        if channel_states:
            if 'channel_states' not in preprocessing_cache[position_key]:
                preprocessing_cache[position_key]['channel_states'] = {}
            
            for channel_name, states in channel_states.items():
                if channel_name not in preprocessing_cache[position_key]['channel_states']:
                    preprocessing_cache[position_key]['channel_states'][channel_name] = {}
                preprocessing_cache[position_key]['channel_states'][channel_name].update(states)
        
        if restored_steps:
            print(f"[Cache Restore] ✓ Restored {len(restored_steps)} steps: {restored_steps}")
            return True
        else:
            print(f"[Cache Restore] No processed files found on disk")
            return False
            
    except Exception as e:
        import traceback
        print(f"[Cache Restore] Error restoring cache: {e}")
        traceback.print_exc()
        return False

# Marker mapping cache: {cycle_upper: {channel_upper: marker}}
# Example: {"C1": {"CH1": "p62", "CH2": "CD63"}, ...}
marker_map: Dict[str, Dict[str, str]] = {}
MARKER_EXCEL_PATH = Path("data/label/Marker_info.xlsx")

def clear_marker_cache():
    """Clear the marker mapping cache to force reload."""
    global marker_map
    marker_map = {}


def load_marker_mapping() -> Dict[str, Dict[str, str]]:
    """
    Load marker mapping from Excel file.
    Columns: D = Cycle, E = Channel, F = Marker
    
    Returns:
        Dictionary: {cycle_upper: {channel_upper: marker}}
        Example: {"C1": {"CH1": "p62", "CH2": "CD63"}}
    """
    global marker_map
    
    # Return cached if already loaded
    if marker_map:
        return marker_map
    
    marker_map = {}
    
    excel_path = Path(MARKER_EXCEL_PATH)
    if not excel_path.exists():
        print(f"Warning: Marker mapping file not found: {excel_path}")
        return marker_map
    
    # Check if openpyxl is available
    if not _openpyxl_available:
        print(f"ERROR: Cannot load marker mapping - openpyxl is not installed!")
        print(f"Please install it with: pip install openpyxl")
        return marker_map
    
    try:
        # Import openpyxl here to ensure it's loaded before pandas tries to use it
        import openpyxl as _opxl
        
        # Read Excel file (columns D, E, F are index 3, 4, 5)
        # Skip first row (row 0) and header row (row 1), start from row 2
        # Explicitly use openpyxl engine for .xlsx files
        df = pd.read_excel(excel_path, usecols="D:F", header=None, skiprows=2, engine='openpyxl')
        
        # Iterate through data rows
        for idx, row in df.iterrows():
            cycle = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
            channel = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
            marker = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else None
            
            # Skip empty rows or invalid data
            if not cycle or cycle.upper() in ['CYCLE#', 'CYCLE', 'NAN', '']:
                continue
            
            # Skip rows without channel (like C0 with Pan-EV)
            if not channel or str(channel).upper() in ['NAN', '']:
                continue
            
            # Normalize to uppercase for lookup
            cycle_upper = cycle.upper()
            channel_upper = str(channel).upper()
            
            # Ensure channel starts with 'CH' if it's just a number
            if channel_upper.isdigit():
                channel_upper = f"CH{channel_upper}"
            elif not channel_upper.startswith('CH'):
                channel_upper = f"CH{channel_upper}"
            
            # Initialize cycle dict if needed
            if cycle_upper not in marker_map:
                marker_map[cycle_upper] = {}
            
            # Store marker (or empty string if missing)
            # Check for various NaN representations
            if marker and marker.upper() not in ['NAN', 'NONE', '']:
                marker_map[cycle_upper][channel_upper] = marker.strip()
            else:
                marker_map[cycle_upper][channel_upper] = ""
        
        print(f"Loaded marker mapping: {len(marker_map)} cycles")
        for cycle, channels in marker_map.items():
            print(f"  {cycle}: {len(channels)} channels")
            for ch, marker in channels.items():
                print(f"    {ch} -> {marker}")
        
    except Exception as e:
        print(f"Error loading marker mapping: {e}")
        marker_map = {}
    
    return marker_map


def get_marker(cycle: str, channel: str) -> Optional[str]:
    """
    Get marker for a given cycle and channel.
    
    Args:
        cycle: Cycle string (e.g., "C1")
        channel: Channel string (e.g., "Ch1")
    
    Returns:
        Marker string or None if not found
    """
    mapping = load_marker_mapping()
    cycle_upper = cycle.upper()
    channel_upper = channel.upper()
    
    # Normalize channel format
    if channel_upper.isdigit():
        channel_upper = f"CH{channel_upper}"
    elif not channel_upper.startswith('CH'):
        channel_upper = f"CH{channel_upper}"
    
    marker = mapping.get(cycle_upper, {}).get(channel_upper)
    
    # Debug output
    if not marker:
        print(f"[get_marker] Lookup failed: cycle={cycle_upper}, channel={channel_upper}")
        print(f"[get_marker] Available cycles: {list(mapping.keys())}")
        if cycle_upper in mapping:
            print(f"[get_marker] Available channels for {cycle_upper}: {list(mapping[cycle_upper].keys())}")
    
    return marker


def generate_preview_png(
    tiff_path: Path,
    fixed_display_min: Optional[float] = None,
    fixed_display_max: Optional[float] = None
) -> Optional[str]:
    """
    Generate a PNG preview from TIFF file and return URL.
    
    Args:
        tiff_path: Path to TIFF file
        
    Returns:
        URL to preview PNG, or None if generation fails
    """
    try:
        # Generate cache key from file path, mtime, and display scaling params
        file_mtime = tiff_path.stat().st_mtime
        scale_key = f"{fixed_display_min}:{fixed_display_max}"
        cache_key = hashlib.sha256(
            f"{tiff_path}:{file_mtime}:{scale_key}".encode()
        ).hexdigest()
        
        preview_path = PREVIEW_CACHE / f"{cache_key}.png"
        
        # Check if we need to regenerate preview
        if preview_path.exists():
            # Check if cached preview is older than the source file
            preview_mtime = preview_path.stat().st_mtime
            if preview_mtime < file_mtime:
                # Source file is newer, delete old preview and regenerate
                print(f"[Preview] Deleting stale preview (source newer): {preview_path}")
                preview_path.unlink()
            else:
                # Preview is up-to-date, return it with timestamp to force browser refresh
                timestamp = int(file_mtime * 1000)  # Use file mtime as cache-busting parameter
                return f'/api/file?path={preview_path}&t={timestamp}'
        
        # Load TIFF and convert to PNG
        print(f"[Preview] Generating new preview for {tiff_path}")
        img_array = tifffile.imread(str(tiff_path))
        
        # Handle multi-dimensional arrays (squeeze out singleton dimensions)
        while img_array.ndim > 2 and 1 in img_array.shape:
            img_array = np.squeeze(img_array)
        
        # If still 3D, take first slice
        if img_array.ndim == 3:
            img_array = img_array[0]
        
        # Ensure 2D
        if img_array.ndim != 2:
            print(f"Warning: Could not convert {tiff_path} to 2D image (shape: {img_array.shape})")
            return None
        
        # Normalize to 8-bit.
        # If fixed display bounds are provided, use them so different stages
        # (e.g., processed vs aligned) can be rendered on the same intensity scale.
        if img_array.dtype != np.uint8:
            if fixed_display_min is not None and fixed_display_max is not None and fixed_display_max > fixed_display_min:
                img_min = float(fixed_display_min)
                img_max = float(fixed_display_max)
            else:
                img_min = float(img_array.min())
                img_max = float(img_array.max())
            if img_max > img_min:
                img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(img_array, dtype=np.uint8)
        else:
            img_normalized = img_array
        
        # Convert to PIL and save
        img_pil = Image.fromarray(img_normalized)
        img_pil.save(preview_path, format='PNG')
        
        # Return URL with timestamp to force browser refresh
        timestamp = int(file_mtime * 1000)
        return f'/api/file?path={preview_path}&t={timestamp}'
    except Exception as e:
        print(f"Error generating preview for {tiff_path}: {e}")
        return None


def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    with ws_lock:
        dead_clients = set()
        for ws in ws_clients:
            try:
                ws.send(json.dumps(message))
            except Exception as e:
                print(f"Error sending to client: {e}")
                dead_clients.add(ws)
        
        # Remove dead clients
        ws_clients.difference_update(dead_clients)


# ============================================================================
# ORIGINAL API ENDPOINTS (from api_server.py)
# ============================================================================

def calculate_transform_movement(transform_data: Dict[str, Any]) -> tuple:
    """Calculate dx, dy, and magnitude from transformation matrix."""
    transform_type = transform_data.get('type', 'identity')
    
    if transform_type == 'identity':
        return 0.0, 0.0, 0.0
    
    transform = transform_data.get('transform', [])
    
    if not transform or len(transform) < 2:
        return 0.0, 0.0, 0.0
    
    try:
        tx = float(transform[0][2]) if len(transform[0]) > 2 else 0.0
        ty = float(transform[1][2]) if len(transform[1]) > 2 else 0.0
        magnitude = np.sqrt(tx**2 + ty**2)
        return tx, ty, magnitude
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0, 0.0


def parse_colocalization_stats(sample_dir: Path) -> Dict[str, Any]:
    """
    Parse colocalization statistics from detections CSV.
    
    Args:
        sample_dir: Path to sample output directory
        
    Returns:
        Dictionary with colocalization statistics per channel and pairs
    """
    # Find detections CSV
    csv_files = list(sample_dir.glob("*_detections.csv"))
    
    if not csv_files:
        return {}
    
    csv_file = csv_files[0]
    
    try:
        # Read CSV and count detections per channel
        channel_stats = defaultdict(lambda: {'total': 0, 'colocalized': 0})
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                channel = row.get('channel', 'unknown')
                is_coloc = row.get('colocalized', 'False').strip().lower() == 'true'
                
                channel_stats[channel]['total'] += 1
                if is_coloc:
                    channel_stats[channel]['colocalized'] += 1
        
        # Calculate totals
        total_detections = sum(stats['total'] for stats in channel_stats.values())
        total_colocalized = sum(stats['colocalized'] for stats in channel_stats.values())
        
        # Format per-channel statistics
        channel_list = []
        for channel, stats in sorted(channel_stats.items()):
            coloc_percent = (stats['colocalized'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            channel_list.append({
                'channel': channel,
                'totalDetections': stats['total'],
                'colocalizedCount': stats['colocalized'],
                'colocalizedPercent': round(coloc_percent, 1)
            })
        
        return {
            'totalDetections': total_detections,
            'totalColocalized': total_colocalized,
            'channels': channel_list,
            'colocalizationRate': round((total_colocalized / total_detections * 100) if total_detections > 0 else 0.0, 1)
        }
        
    except Exception as e:
        print(f"Error parsing colocalization stats: {e}")
        return {}


def parse_combo_analysis(sample_dir: Path, sample_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse multi-channel combination analysis from JSON file.
    
    Args:
        sample_dir: Path to sample output directory
        sample_name: Name of the sample
        
    Returns:
        Dictionary with combo_table and statistics, or None if not found
    """
    combo_json_path = sample_dir / f"{sample_name}_combinations.json"
    
    if not combo_json_path.exists():
        return None
    
    try:
        import json
        with open(combo_json_path, 'r') as f:
            combo_data = json.load(f)
        
        # Format combo_table for frontend
        combo_table = []
        for entry in combo_data.get('combo_table', []):
            combo_table.append({
                'combo': entry['combo'],
                'count': entry['count'],
                'rate': round(entry['rate'], 4),  # Keep as decimal (0.0-1.0)
                'percentage': round(entry['rate'] * 100, 1)  # Also provide as percentage
            })
        
        return {
            'image_id': sample_name,
            'channels_present': combo_data.get('channels_present', []),
            'threshold_px': combo_data.get('threshold_px', 10.0),
            'total_objects_grouped': combo_data.get('total_objects_grouped', 0),
            'combo_table': combo_table
        }
    except Exception as e:
        print(f"Error parsing combo analysis: {e}")
        return None


def load_alignment_results(sample_name: str) -> Optional[Dict[str, Any]]:
    """Load alignment results from pipeline output."""
    sample_dir = OUTPUT_ROOT / sample_name
    
    if not sample_dir.exists():
        return None
    
    preprocessed_dir = sample_dir / "preprocessed"
    registered_dir = sample_dir / "registered"
    labels_dir = sample_dir / "labels"
    
    if not registered_dir.exists():
        return None
    
    registered_files = list(registered_dir.glob("*.png"))
    
    frames = []
    for reg_file in registered_files:
        channel_name = reg_file.stem.replace('_registered', '')
        preproc_file = preprocessed_dir / f"{channel_name}_preprocessed.png"
        
        if not preproc_file.exists():
            continue
        
        dx, dy = 0.0, 0.0
        magnitude = 0.0
        
        frames.append({
            'frameId': channel_name,
            'channelName': channel_name,
            'beforeImageUrl': f'/api/images/{sample_name}/preprocessed/{channel_name}_preprocessed.png',
            'afterImageUrl': f'/api/images/{sample_name}/registered/{channel_name}_registered.png',
            'movement': {
                'dx': dx,
                'dy': dy,
                'magnitude': magnitude,
                'transformType': 'affine',
                'residualError': 0.0,
                'numMatches': 0
            }
        })
    
    final_results = []
    
    overlay_file = sample_dir / f"{sample_name}_overlay.png"
    if overlay_file.exists():
        final_results.append({
            'type': 'overlay',
            'name': overlay_file.name,
            'url': f'/api/results/{sample_name}/{overlay_file.name}',
            'description': 'RGB overlay of all registered channels'
        })
    
    csv_file = sample_dir / f"{sample_name}_detections.csv"
    if csv_file.exists():
        final_results.append({
            'type': 'csv',
            'name': csv_file.name,
            'url': f'/api/results/{sample_name}/{csv_file.name}',
            'description': 'Quantification data with coordinates and intensities'
        })
    
    for reg_file in registered_files:
        final_results.append({
            'type': 'registered',
            'name': reg_file.name,
            'url': f'/api/results/{sample_name}/registered/{reg_file.name}',
            'description': f'Registered image: {reg_file.stem}'
        })
    
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.png"):
            final_results.append({
                'type': 'label',
                'name': label_file.name,
                'url': f'/api/results/{sample_name}/labels/{label_file.name}',
                'description': f'Detection labels: {label_file.stem}'
            })
    
    # Parse colocalization statistics
    coloc_stats = parse_colocalization_stats(sample_dir)
    
    # Parse multi-channel combination analysis
    combo_analysis = parse_combo_analysis(sample_dir, sample_name)
    
    return {
        'sampleName': sample_name,
        'frames': frames,
        'finalResults': final_results,
        'colocalization': coloc_stats,
        'comboAnalysis': combo_analysis,  # NEW: Multi-channel combination analysis
        'metadata': {
            'anchorChannel': frames[0]['channelName'] if frames else 'unknown',
            'totalChannels': len(frames)
        }
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'SEA API Server is running'
    })


@app.route('/api/alignment/<sample_name>', methods=['GET'])
def get_alignment(sample_name: str):
    """Get alignment data for a sample."""
    result = load_alignment_results(sample_name)
    
    if result is None:
        return jsonify({
            'success': False,
            'error': f'Sample "{sample_name}" not found'
        }), 404
    
    return jsonify({
        'success': True,
        'data': result
    })


@app.route('/api/images/<sample_name>/<subfolder>/<filename>', methods=['GET'])
def get_image(sample_name: str, subfolder: str, filename: str):
    """Serve PNG images (or convert TIFF to PNG for backward compatibility)."""
    file_path = OUTPUT_ROOT / sample_name / subfolder / filename
    
    if not file_path.exists():
        abort(404)
    
    # If already PNG, serve directly
    if file_path.suffix.lower() == '.png':
        return send_file(file_path, mimetype='image/png')
    
    # If TIFF (for backward compatibility with old data), convert to PNG
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            img_array = tifffile.imread(str(file_path))
            img_normalized = ((img_array - img_array.min()) / 
                             (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        except Exception as e:
            app.logger.error(f"Error converting TIFF: {e}")
            abort(500)
    
    # Unknown format
    abort(404)


@app.route('/api/results/<sample_name>/<path:filepath>', methods=['GET'])
def get_result_file(sample_name: str, filepath: str):
    """Serve result files."""
    file_path = OUTPUT_ROOT / sample_name / filepath
    
    if not file_path.exists():
        abort(404)
    
    if file_path.suffix == '.png':
        mimetype = 'image/png'
    elif file_path.suffix == '.csv':
        mimetype = 'text/csv'
    elif file_path.suffix in ['.tif', '.tiff']:
        try:
            img_array = tifffile.imread(str(file_path))
            img_normalized = ((img_array - img_array.min()) / 
                             (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_normalized)
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0)
            return send_file(buffer, mimetype='image/png')
        except Exception as e:
            app.logger.error(f"Error converting TIFF: {e}")
            abort(500)
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(file_path, mimetype=mimetype)


@app.route('/api/samples', methods=['GET'])
def list_samples():
    """List all available samples (output)."""
    if not OUTPUT_ROOT.exists():
        return jsonify({
            'success': False,
            'error': 'Output directory not found'
        }), 404
    
    samples = [d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
    
    return jsonify({
        'success': True,
        'data': samples
    })


@app.route('/api/input/samples', methods=['GET'])
def list_input_samples():
    """List all available input samples (raw data)."""
    if not DATA_ROOT.exists():
        return jsonify({
            'success': False,
            'error': f'Input directory not found: {DATA_ROOT}'
        }), 404
    
    try:
        # List only directories, sorted
        samples = sorted([
            d.name for d in DATA_ROOT.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        return jsonify({
            'success': True,
            'data': samples
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error listing samples: {str(e)}'
        }), 500


@app.route('/api/input/samples/<sample_name>/positions', methods=['GET'])
def list_sample_positions(sample_name: str):
    """List all position folders for a given sample."""
    sample_path = DATA_ROOT / sample_name
    
    if not sample_path.exists():
        return jsonify({
            'success': False,
            'error': f'Sample not found: {sample_name}'
        }), 404
    
    if not sample_path.is_dir():
        return jsonify({
            'success': False,
            'error': f'Sample is not a directory: {sample_name}'
        }), 400
    
    try:
        # List position directories (P1, P2, etc.), sorted naturally
        positions = sorted([
            d.name for d in sample_path.iterdir() 
            if d.is_dir() and d.name.startswith('P') and d.name[1:].isdigit()
        ], key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 0)
        
        return jsonify({
            'success': True,
            'data': positions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error listing positions: {str(e)}'
        }), 500


@app.route('/api/file', methods=['GET'])
def serve_file():
    """
    Serve files from allowed directories (preview cache, output).
    Prevents path traversal attacks.
    """
    file_path_str = request.args.get('path', '')
    
    print(f"[DEBUG] serve_file requested: {file_path_str}")
    
    if not file_path_str:
        print("[ERROR] No path specified")
        return jsonify({'success': False, 'error': 'No path specified'}), 400
    
    try:
        # Strip query parameters (cache-busting timestamp)
        if '?' in file_path_str:
            file_path_str = file_path_str.split('?')[0]
            print(f"[DEBUG] Stripped query params: {file_path_str}")
        
        file_path = Path(file_path_str).resolve()
        print(f"[DEBUG] Resolved path: {file_path}")
        
        # Security: Only allow files from preview cache or output directory
        allowed_roots = [PREVIEW_CACHE.resolve(), OUTPUT_ROOT.resolve()]
        print(f"[DEBUG] Allowed roots: {allowed_roots}")
        
        is_allowed = any(str(file_path).startswith(str(root)) for root in allowed_roots)
        print(f"[DEBUG] Is allowed: {is_allowed}")
        
        if not is_allowed:
            print(f"[ERROR] Access denied for path: {file_path}")
            return jsonify({'success': False, 'error': 'Access denied'}), 403
        
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Determine mimetype
        suffix = file_path.suffix.lower()
        if suffix == '.png':
            mimetype = 'image/png'
        elif suffix == '.jpg' or suffix == '.jpeg':
            mimetype = 'image/jpeg'
        elif suffix == '.tif' or suffix == '.tiff':
            mimetype = 'image/tiff'
        else:
            mimetype = 'application/octet-stream'
        
        print(f"[DEBUG] Serving file: {file_path} with mimetype: {mimetype}")
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        print(f"[ERROR] Exception in serve_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/input/load_position', methods=['POST'])
def load_position():
    """
    Load a position and detect available cycle+channel combinations.
    
    Body:
    {
        "sample": "A2780Cis10",
        "position": "P1"
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "sample": "A2780Cis10",
            "position": "P1",
            "items": [
                {
                    "key": "C1_ch1",
                    "cycle": "C1",
                    "channel": "Ch1",
                    "marker": "p62",
                    "display_label": "C1_ch1(p62)",
                    "tiff_path": "/path/to/file.tif",
                    "preview_url": "/api/file?path=..."
                },
                ...
            ]
        }
    }
    """
    data = request.json or {}
    sample_name = data.get('sample')
    position_name = data.get('position')
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    position_path = DATA_ROOT / sample_name / position_name
    
    if not position_path.exists():
        return jsonify({
            'success': False,
            'error': f'Position not found: {sample_name}/{position_name}'
        }), 404
    
    try:
        # Load marker mapping
        marker_mapping = load_marker_mapping()
        
        # Pattern to match: _C1_Ch1.tif or _C1-Ch1.tif (case insensitive)
        # Also handle variations like C1_Ch1, C1-Ch1
        cycle_channel_pattern = re.compile(
            r'[_-]C(\d+)[_-]Ch(\d+)\.tif$', 
            re.IGNORECASE
        )
        
        items = []
        channel_files_dict = {}  # For preprocessing cache: {key: file_path}
        
        for file_path in position_path.glob('*.tif*'):  # Match both .tif and .tiff
            match = cycle_channel_pattern.search(file_path.name)
            if match:
                cycle_num = int(match.group(1))
                channel_num = int(match.group(2))
                
                cycle = f'C{cycle_num}'
                channel = f'Ch{channel_num}'
                key = f'{cycle}_ch{channel_num}'  # Internal key: C1_ch1
                
                # Get marker from mapping
                marker = get_marker(cycle, channel)
                print(f"[Load Position] Cycle={cycle}, Channel={channel}, Marker={marker}")
                
                # Build display label
                if marker:
                    display_label = f'{cycle}_ch{channel_num}({marker})'
                else:
                    display_label = f'{cycle}_ch{channel_num}'
                    print(f"[Load Position] WARNING: No marker found for {cycle}/{channel}")
                
                # Generate preview
                preview_url = generate_preview_png(file_path)
                
                items.append({
                    'key': key,
                    'cycle': cycle,
                    'channel': channel,
                    'marker': marker,
                    'display_label': display_label,
                    'tiff_path': str(file_path),
                    'preview_url': preview_url
                })
                
                # Store for preprocessing cache
                channel_files_dict[key] = file_path
        
        if not items:
            return jsonify({
                'success': False,
                'error': f'No cycle+channel files found in {sample_name}/{position_name}',
                'warning': 'Expected files matching pattern: *_C1_Ch1.tif, *_C1-Ch1.tif, etc.'
            }), 404
        
        # Sort items by cycle then channel
        items.sort(key=lambda x: (x['cycle'], int(x['channel'][2:])))
        
        # Initialize preprocessing cache with raw files
        position_key = f"{sample_name}/{position_name}"
        preprocessing_cache[position_key] = {
            'raw': channel_files_dict
        }
        
        # Try to restore processed stages from disk
        print(f"[Load Position] Attempting to restore processed stages from disk...")
        restore_preprocessing_cache_from_disk(sample_name, position_name)
        
        if position_key in preprocessing_cache:
            cached_stages = [k for k in preprocessing_cache[position_key].keys() if k != 'raw' and k != 'channel_states']
            if cached_stages:
                print(f"[Load Position] Restored processed stages: {cached_stages}")
        
        return jsonify({
            'success': True,
            'data': {
                'sample': sample_name,
                'position': position_name,
                'items': items,
                'path': str(position_path)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error loading position: {str(e)}'
        }), 500


# ============================================================================
# NEW PIPELINE CONTROL ENDPOINTS
# ============================================================================

def _load_stage_image(position_key: str, input_stage: str, channel_name: str) -> Optional[np.ndarray]:
    """Load a single channel image from the preprocessing cache for the given stage."""
    if position_key not in preprocessing_cache:
        return None
    stage_files = preprocessing_cache[position_key].get(input_stage, {})
    path = stage_files.get(channel_name)
    if path is None:
        return None
    try:
        return tifffile.imread(str(path)).astype(np.float32)
    except Exception:
        return None


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize a float32 image to uint8 for display."""
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def _encode_overlay_to_b64(overlay_bgr: np.ndarray) -> str:
    """Encode a BGR uint8 image to base64 PNG string."""
    _, buf = cv2.imencode('.png', overlay_bgr)
    import base64
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def _preprocess_for_grid(gray_u8: np.ndarray, blur_ksize: int = 21,
                          threshold_method: str = 'otsu', invert: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolate dark PDMS grid lines via: Gaussian blur → invert → threshold → morphological cleanup.
    Returns: (preprocessed_u8, binary_mask_u8)
    The inversion step is critical: dark grid lines become bright so Hough/corner detectors find them.
    """
    # Ensure odd kernel
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_ksize = max(3, blur_ksize)

    # Step 1: Heavy Gaussian blur — suppresses sub-grid fluorescent features
    blurred = cv2.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)

    # Step 2: Invert so dark grid lines → bright features
    preprocessed = cv2.bitwise_not(blurred) if invert else blurred

    # Step 3: Threshold to isolate grid lines
    if threshold_method == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 31, 5)
    elif threshold_method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(preprocessed, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 5)
    else:  # otsu (default)
        _, binary = cv2.threshold(preprocessed, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological opening — removes small noise blobs, keeps grid lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return preprocessed, binary


def _merge_parallel_lines(lines: List[Tuple], merge_dist: float = 20.0,
                           is_vertical: bool = False) -> List[Tuple]:
    """Merge lines that are parallel and closer than merge_dist px into a single representative line."""
    if not lines:
        return []

    def center_coord(line):
        x1, y1, x2, y2 = line
        return (y1 + y2) / 2.0 if not is_vertical else (x1 + x2) / 2.0

    sorted_lines = sorted(lines, key=center_coord)
    merged = []
    group = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        if abs(center_coord(line) - center_coord(group[-1])) <= merge_dist:
            group.append(line)
        else:
            merged.append(_average_line(group, is_vertical))
            group = [line]
    merged.append(_average_line(group, is_vertical))
    return merged


def _average_line(group: List[Tuple], is_vertical: bool) -> Tuple:
    """Collapse a group of similar lines into one representative line."""
    xs1 = [l[0] for l in group]
    ys1 = [l[1] for l in group]
    xs2 = [l[2] for l in group]
    ys2 = [l[3] for l in group]
    if is_vertical:
        avg_x = int(round((sum(xs1) + sum(xs2)) / (2 * len(group))))
        return (avg_x, min(ys1 + ys2), avg_x, max(ys1 + ys2))
    else:
        avg_y = int(round((sum(ys1) + sum(ys2)) / (2 * len(group))))
        return (min(xs1 + xs2), avg_y, max(xs1 + xs2), avg_y)


def _compute_line_intersections(h_lines: List, v_lines: List) -> np.ndarray:
    """Compute pairwise intersection points between horizontal and vertical lines."""
    points = []
    for (hx1, hy1, hx2, hy2) in h_lines:
        for (vx1, vy1, vx2, vy2) in v_lines:
            dxh, dyh = hx2 - hx1, hy2 - hy1
            dxv, dyv = vx2 - vx1, vy2 - vy1
            denom = dxh * dyv - dyh * dxv
            if abs(denom) < 1e-6:
                continue
            t = ((vx1 - hx1) * dyv - (vy1 - hy1) * dxv) / denom
            px = hx1 + t * dxh
            py = hy1 + t * dyh
            points.append([px, py])
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def _detect_grid_lines(gray_u8: np.ndarray,
                       blur_ksize: int = 21,
                       threshold_method: str = 'otsu',
                       invert: bool = True,
                       hough_threshold: int = 50,
                       min_line_length: float = 80.0,
                       max_line_gap: float = 20.0,
                       angle_tolerance: float = 15.0,
                       merge_distance: float = 20.0,
                       ) -> Tuple[List, List, np.ndarray, np.ndarray]:
    """
    Detect dark PDMS grid lines.
    Pipeline: blur → invert → threshold → morphology → Hough → merge.
    Returns (h_lines, v_lines, preprocessed_u8, binary_u8).
    """
    preprocessed, binary = _preprocess_for_grid(gray_u8, blur_ksize, threshold_method, invert)

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )

    h_lines, v_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < angle_tolerance:
                h_lines.append((x1, y1, x2, y2))
            elif angle > (90.0 - angle_tolerance):
                v_lines.append((x1, y1, x2, y2))

    h_lines = _merge_parallel_lines(h_lines, merge_distance, is_vertical=False)
    v_lines = _merge_parallel_lines(v_lines, merge_distance, is_vertical=True)

    return h_lines, v_lines, preprocessed, binary


def _detect_grid_corners(gray_u8: np.ndarray,
                         blur_ksize: int = 21,
                         threshold_method: str = 'otsu',
                         invert: bool = True,
                         max_corners: int = 200,
                         quality_level: float = 0.01,
                         min_distance: float = 10.0,
                         block_size: int = 3,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect grid intersection corners.
    Pipeline: blur → invert → threshold → Shi-Tomasi corner detection.
    Returns (corners Nx2, preprocessed_u8, binary_u8).
    """
    preprocessed, binary = _preprocess_for_grid(gray_u8, blur_ksize, threshold_method, invert)

    corners = cv2.goodFeaturesToTrack(
        binary,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32), preprocessed, binary
    return corners.reshape(-1, 2), preprocessed, binary


def _draw_corners_overlay(gray_u8: np.ndarray, corners: np.ndarray, radius: int = 6) -> np.ndarray:
    """Draw detected corners as green circles on a BGR copy of the image."""
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(bgr, (int(round(x)), int(round(y))), radius, (0, 255, 0), 2)
    return bgr


def _draw_lines_overlay(gray_u8: np.ndarray, h_lines: List, v_lines: List,
                         intersections: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw detected grid lines (yellow H, blue V) and optional intersection points on the image."""
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2) in h_lines:
        cv2.line(bgr, (x1, y1), (x2, y2), (0, 220, 220), 2)   # yellow horizontal
    for (x1, y1, x2, y2) in v_lines:
        cv2.line(bgr, (x1, y1), (x2, y2), (255, 100, 0), 2)   # blue vertical
    if intersections is not None and len(intersections) > 0:
        for px, py in intersections:
            cv2.circle(bgr, (int(round(px)), int(round(py))), 5, (0, 255, 0), -1)
    return bgr


def _draw_intersections_overlay(gray_u8: np.ndarray, intersections: np.ndarray) -> np.ndarray:
    """Draw intersection points as green filled circles."""
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for px, py in intersections:
        cv2.circle(bgr, (int(round(px)), int(round(py))), 5, (0, 255, 0), -1)
    return bgr


# ─── FFT / Frequency-Domain Grid Detection helpers ───────────────────────────

def _make_window2d(h: int, w: int, fn: str = 'hann') -> np.ndarray:
    """Create a 2D separable window of shape (h, w)."""
    if fn == 'hann':
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float64)
    if fn == 'hamming':
        return np.outer(np.hamming(h), np.hamming(w)).astype(np.float64)
    return np.ones((h, w), dtype=np.float64)


def _log_magnitude_spectrum(img_f32: np.ndarray,
                             window_fn: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 1+2: Window → 2D FFT → fftshift → log(1 + |F|).
    Returns (log_mag_f64, complex_shifted_F).
    Phase of the spatial cosine at peak (pr, pc): phi = angle(F_shifted[pr, pc]).
    """
    h, w = img_f32.shape
    win = _make_window2d(h, w, window_fn)
    F = np.fft.fft2(img_f32.astype(np.float64) * win)
    F_shifted = np.fft.fftshift(F)
    log_mag = np.log1p(np.abs(F_shifted))
    return log_mag, F_shifted


def _refine_peak_subpixel_fft(spec: np.ndarray, r: int, c: int) -> Tuple[float, float]:
    """Step 4: Parabolic sub-pixel peak refinement."""
    hh, ww = spec.shape
    r = int(np.clip(r, 1, hh - 2))
    c = int(np.clip(c, 1, ww - 2))
    dr = dc = 0.0
    denom_r = 2 * spec[r, c] - spec[r - 1, c] - spec[r + 1, c]
    if abs(denom_r) > 1e-10:
        dr = 0.5 * (spec[r - 1, c] - spec[r + 1, c]) / denom_r
    denom_c = 2 * spec[r, c] - spec[r, c - 1] - spec[r, c + 1]
    if abs(denom_c) > 1e-10:
        dc = 0.5 * (spec[r, c - 1] - spec[r, c + 1]) / denom_c
    return float(r) + dr, float(c) + dc


def _find_grid_freq_peaks(log_mag: np.ndarray,
                           F_shifted: np.ndarray,
                           min_spacing_px: float,
                           max_spacing_px: float,
                           peak_threshold_rel: float,
                           enable_rotation: bool,
                           angle_tolerance_deg: float = 20.0,
                           ) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Step 3+4: Detect dominant H-grid and V-grid frequency peaks from log-magnitude spectrum.

    Peak at (pr, pc) in shifted spectrum → freq: fy=(pr-cy)/H, fx=(pc-cx)/W (cycles/pixel).
    - H-grid peaks (horizontal lines in image): periodic in Y → fy large, fx≈0 → normal ≈ 90°
    - V-grid peaks (vertical lines in image): periodic in X → fx large, fy≈0 → normal ≈ 0°

    Returns (h_peak_dict, v_peak_dict), each containing:
        row, col, fy, fx, spacing_px, normal_angle_deg, phase_rad, magnitude
    """
    from scipy.ndimage import maximum_filter as mf

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    y_g, x_g = np.ogrid[:h, :w]

    # Suppress DC region
    dc_r = max(5, int(min(h, w) / 30))
    spec = log_mag.copy()
    spec[(y_g - cy) ** 2 + (x_g - cx) ** 2 <= dc_r ** 2] = 0.0

    # Annular band: spacing range → frequency radius range
    dist = np.sqrt((y_g - cy) ** 2 + (x_g - cx) ** 2)
    r_min = min(h, w) / (2.0 * max_spacing_px)   # low freq (large spacing)
    r_max = min(h, w) / (2.0 * min_spacing_px)    # high freq (small spacing)
    annular = (dist >= max(1.0, r_min)) & (dist <= r_max)

    # Upper half-plane only (avoid conjugate duplicates)
    half_plane = (y_g < cy) | ((y_g == cy) & (x_g > cx))
    masked = spec * (annular & half_plane)

    if masked.max() < 1e-10:
        return None, None

    # Local maxima above relative threshold
    nbr = max(3, int(r_min * 0.8))
    local_max_mask = (masked == mf(masked, size=nbr)) & (masked >= masked.max() * peak_threshold_rel)
    positions = np.argwhere(local_max_mask)
    if len(positions) == 0:
        return None, None

    peaks = []
    for pr, pc in positions:
        pr_f, pc_f = _refine_peak_subpixel_fft(spec, pr, pc)
        fy = (pr_f - cy) / float(h)
        fx = (pc_f - cx) / float(w)
        freq_mag = np.sqrt(fy ** 2 + fx ** 2)
        if freq_mag < 1e-12:
            continue
        spacing = 1.0 / freq_mag
        normal_angle = np.degrees(np.arctan2(fy, fx))   # direction of freq vector
        phase_rad = float(np.angle(F_shifted[pr, pc]))   # phase of spatial cosine
        peaks.append(dict(row=pr, col=pc, fy=fy, fx=fx,
                          spacing_px=spacing, normal_angle_deg=normal_angle,
                          phase_rad=phase_rad, magnitude=float(masked[pr, pc])))

    peaks.sort(key=lambda p: p['magnitude'], reverse=True)

    h_peak = v_peak = None
    for p in peaks:
        a = abs(p['normal_angle_deg']) % 180
        tol = angle_tolerance_deg if enable_rotation else 10.0
        is_h = abs(a - 90) < tol               # near 90° → H-grid (periodic in Y)
        is_v = (a < tol) or (a > 180 - tol)    # near 0°/180° → V-grid (periodic in X)
        if h_peak is None and is_h:
            h_peak = p
        elif v_peak is None and is_v:
            v_peak = p
        if h_peak is not None and v_peak is not None:
            break

    return h_peak, v_peak


def _generate_synthetic_grid(h: int, w: int,
                              h_peak: Optional[Dict],
                              v_peak: Optional[Dict],
                              line_width_px: float = 2.0) -> np.ndarray:
    """
    Step 5: Generate ideal grid image from detected frequency peaks.
    Spatial cosine at peak: cos(2π*(fy*y + fx*x) + phase_rad).
    Returns uint8 [0, 255] image.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    grid = np.zeros((h, w), dtype=np.float64)

    def add_lines(peak):
        fy, fx = peak['fy'], peak['fx']
        spacing = peak['spacing_px']
        phi = peak['phase_rad']
        wave = 2.0 * np.pi * (fy * y + fx * x) + phi
        cos_wave = np.cos(wave)
        # threshold: line_width_px wide band around each grid line
        half_w = np.pi * line_width_px / spacing
        return (cos_wave >= np.cos(half_w)).astype(np.float64)

    if h_peak is not None:
        grid = np.maximum(grid, add_lines(h_peak))
    if v_peak is not None:
        grid = np.maximum(grid, add_lines(v_peak))
    return (grid * 255).astype(np.uint8)


def _compute_fft_grid_intersections(h: int, w: int,
                                    h_peak: Dict, v_peak: Dict) -> np.ndarray:
    """
    Step 7: Analytically compute intersection coordinates from frequency peaks.

    H-grid line n: fy_h*y + fx_h*x = n - phi_h/(2π)
    V-grid line m: fy_v*y + fx_v*x = m - phi_v/(2π)
    Solve 2×2 for each (n, m) pair; keep those within image bounds.
    """
    fy_h, fx_h = h_peak['fy'], h_peak['fx']
    fy_v, fx_v = v_peak['fy'], v_peak['fx']
    rhs_h0 = -h_peak['phase_rad'] / (2.0 * np.pi)   # phase of line 0
    rhs_v0 = -v_peak['phase_rad'] / (2.0 * np.pi)

    det = fy_h * fx_v - fx_h * fy_v
    if abs(det) < 1e-12:
        return np.empty((0, 2), dtype=np.float32)

    # RHS range over the image corners
    corners_h = [fy_h * ry + fx_h * rx for ry, rx in [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]]
    corners_v = [fy_v * ry + fx_v * rx for ry, rx in [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]]
    n_range = range(int(np.floor(min(corners_h) + rhs_h0)) - 1,
                    int(np.ceil(max(corners_h) + rhs_h0)) + 2)
    m_range = range(int(np.floor(min(corners_v) + rhs_v0)) - 1,
                    int(np.ceil(max(corners_v) + rhs_v0)) + 2)

    points = []
    for n in n_range:
        rh = n + rhs_h0
        for m in m_range:
            rv = m + rhs_v0
            yp = (rh * fx_v - rv * fx_h) / det
            xp = (fy_h * rv - fy_v * rh) / det
            if 0 <= yp < h and 0 <= xp < w:
                points.append([float(xp), float(yp)])   # (x, y)

    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def _draw_fft_spectrum_overlay(log_mag: np.ndarray,
                                h_peak: Optional[Dict],
                                v_peak: Optional[Dict]) -> np.ndarray:
    """Visualize log-magnitude spectrum with detected H/V peaks highlighted."""
    u8 = _normalize_for_display(log_mag.astype(np.float32))
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    hh, ww = log_mag.shape
    cy, cx = hh // 2, ww // 2

    def draw_peak_pair(peak, color):
        pr, pc = int(peak['row']), int(peak['col'])
        cv2.circle(bgr, (pc, pr), 9, color, 2)
        cv2.drawMarker(bgr, (pc, pr), color, cv2.MARKER_CROSS, 18, 2)
        # conjugate pair (mirror through DC)
        mr, mc = 2 * cy - pr, 2 * cx - pc
        cv2.circle(bgr, (int(mc), int(mr)), 9, color, 2)
        cv2.drawMarker(bgr, (int(mc), int(mr)), color, cv2.MARKER_CROSS, 18, 2)

    if h_peak is not None:
        draw_peak_pair(h_peak, (0, 220, 220))   # yellow → H-grid
    if v_peak is not None:
        draw_peak_pair(v_peak, (255, 100, 0))   # blue   → V-grid
    # DC marker
    cv2.circle(bgr, (cx, cy), 6, (200, 200, 200), 1)
    return bgr


def _simple_pcc_preview(ref_img: np.ndarray, mov_img: np.ndarray) -> np.ndarray:
    """Fallback: phase cross-correlation between two images → shift (dx, dy).
    Returns a JET-colorized correlation map for display."""
    F1 = np.fft.fft2(ref_img.astype(np.float64))
    F2 = np.fft.fft2(mov_img.astype(np.float64))
    cross = F1 * np.conj(F2)
    cc = np.abs(np.fft.ifft2(cross / (np.abs(cross) + 1e-10)))
    cc_shifted = np.fft.fftshift(cc)
    u8 = _normalize_for_display(cc_shifted.astype(np.float32))
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _manual_diagonal_align(
    images_dict: Dict[str, np.ndarray],
    ref_channel: str,
    diagonal_boxes: Dict[str, Any],
    transform_type: str,
    output_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, Any], Dict[str, Any]]:
    """
    Compute per-channel alignment from manually drawn diagonal boxes.

    Each box defines 4 corners via (cx, cy, width, height, angle).
    For each target channel, the homography / affine that maps its box corners
    onto the reference box corners is estimated, then applied to the image.

    Returns: (output_files, shift_vectors, transformations)
    """
    def _box_corners(box: Dict) -> np.ndarray:
        cx, cy = float(box['cx']), float(box['cy'])
        hw, hh = float(box['width']) / 2.0, float(box['height']) / 2.0
        angle  = float(box['angle'])
        cos, sin = np.cos(angle), np.sin(angle)
        locals_ = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return np.float32([
            [cx + lx * cos - ly * sin, cy + lx * sin + ly * cos]
            for lx, ly in locals_
        ])

    ref_box = diagonal_boxes.get(ref_channel)
    if not ref_box:
        raise ValueError(f"No diagonal box provided for reference channel '{ref_channel}'")
    ref_corners = _box_corners(ref_box)

    output_files:    Dict[str, Path]  = {}
    shift_vectors:   Dict[str, Any]   = {}
    transformations: Dict[str, Any]   = {}

    for channel_name, img in images_dict.items():
        h, w = img.shape[:2]
        out_path = output_dir / f"{channel_name}_aligned.tif"

        if channel_name == ref_channel:
            tifffile.imwrite(str(out_path), img)
            output_files[channel_name]    = out_path
            shift_vectors[channel_name]   = {'dx': 0.0, 'dy': 0.0, 'magnitude': 0.0, 'type': 'reference',
                                              'residual_error': 0.0, 'num_matches': 4}
            transformations[channel_name] = {'type': 'identity', 'transform': np.eye(3).tolist()}
            continue

        tgt_box = diagonal_boxes.get(channel_name)
        if not tgt_box:
            # No box placed for this channel — identity (copy)
            tifffile.imwrite(str(out_path), img)
            output_files[channel_name]    = out_path
            shift_vectors[channel_name]   = {'dx': 0.0, 'dy': 0.0, 'magnitude': 0.0, 'type': 'identity',
                                              'residual_error': float('inf'), 'num_matches': 0,
                                              'note': 'No box placed — identity transform used'}
            transformations[channel_name] = {'type': 'identity', 'transform': np.eye(3).tolist()}
            continue

        tgt_corners = _box_corners(tgt_box)

        # Estimate transform: tgt_corners → ref_corners
        img_f32 = img.astype(np.float32)
        if transform_type in ('euclidean', 'similarity'):
            M_partial, _ = cv2.estimateAffinePartial2D(tgt_corners, ref_corners, method=cv2.RANSAC)
            if M_partial is None:
                M_partial = cv2.getAffineTransform(tgt_corners[:3], ref_corners[:3])
            warped = cv2.warpAffine(img_f32, M_partial, (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            M = np.vstack([M_partial, [0.0, 0.0, 1.0]])
        else:
            # Affine / homography — use full homography from 4 point correspondences
            M, _ = cv2.findHomography(tgt_corners, ref_corners, cv2.RANSAC, 1.0)
            if M is None:
                M, _ = cv2.findHomography(tgt_corners, ref_corners, 0)
            if M is None:
                M = np.eye(3, dtype=np.float64)
            warped = cv2.warpPerspective(img_f32, M, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Restore original dtype
        if img.dtype == np.uint16:
            warped = np.clip(warped, 0, 65535).astype(np.uint16)
        elif img.dtype == np.uint8:
            warped = np.clip(warped, 0, 255).astype(np.uint8)
        else:
            warped = warped.astype(img.dtype)

        tifffile.imwrite(str(out_path), warped)
        output_files[channel_name] = out_path

        tx, ty     = float(M[0, 2]), float(M[1, 2])
        magnitude  = float(np.sqrt(tx ** 2 + ty ** 2))
        shift_vectors[channel_name]   = {'dx': tx, 'dy': ty, 'magnitude': magnitude,
                                          'type': 'affine', 'residual_error': 0.0, 'num_matches': 4}
        transformations[channel_name] = {'type': 'affine', 'transform': M.tolist()}

    return output_files, shift_vectors, transformations


@app.route('/api/input/detect_features', methods=['POST'])
def detect_features():
    """
    Stage A — Detect grid features for alignment preview.

    Shared preprocessing params (grid methods):
        blur_ksize: int (default 21) — Gaussian blur kernel; larger = more cell-feature suppression
        threshold_method: "otsu" | "adaptive_mean" | "adaptive_gaussian"
        invert: bool (default true) — invert so dark grid lines become bright targets

    Method-specific params:
        grid_line_hough:
            hough_threshold, min_line_length, max_line_gap,
            angle_tolerance, merge_distance
        grid_intersection_homography:
            max_corners, quality_level, min_distance, block_size

    Returns preview_layers per channel:
        preprocessed — blurred+inverted image (verify grid lines are now bright)
        binary_mask  — thresholded binary (verify clean grid extraction)
        detected     — lines/corners drawn on original gray image
        intersections (Hough only) — computed H×V intersection points
    """
    data = request.json or {}
    sample_name = data.get('sample')
    position_name = data.get('position')
    input_stage = data.get('input_stage', 'contrast_enhance')
    ref_channel = data.get('ref_channel', '')
    method = data.get('method', 'frequency_domain_fft')
    # Accept legacy name
    if method == 'phase_cross_correlation':
        method = 'frequency_domain_fft'
    params = data.get('method_params', {})

    if not sample_name or not position_name:
        return jsonify({'success': False, 'error': 'Missing sample or position'}), 400

    position_key = f"{sample_name}/{position_name}"
    if position_key not in preprocessing_cache:
        return jsonify({'success': False, 'error': 'Position not loaded. Load the position first.'}), 400

    stage_files = preprocessing_cache[position_key].get(input_stage, {})
    if not stage_files:
        return jsonify({'success': False,
                        'error': 'No images found for stage "{}". Process images first.'.format(input_stage)}), 400

    # Shared grid preprocessing params
    blur_ksize = int(params.get('blur_ksize', 21))
    threshold_method = str(params.get('threshold_method', 'otsu'))
    invert = bool(params.get('invert', True))

    overlays: Dict[str, str] = {}
    preview_layers: Dict[str, Dict[str, str]] = {}
    feature_counts: Dict[str, Any] = {}

    # Manual diagonal: no server-side detection — boxes are drawn interactively in the UI
    if method == 'manual_diagonal':
        return jsonify({'success': True, 'data': {
            'preview_layers': {},
            'feature_counts': {},
            'summary': 'Manual diagonal mode — draw a diagonal box on each channel in the canvas.',
        }})

    try:
        if method == 'grid_intersection_homography':
            max_corners = int(params.get('max_corners', 200))
            quality_level = float(params.get('quality_level', 0.01))
            min_distance = float(params.get('min_distance', 10.0))
            block_size = int(params.get('block_size', 3))

            for ch_name, path in stage_files.items():
                img = tifffile.imread(str(path)).astype(np.float32)
                gray = _normalize_for_display(img)

                corners, preprocessed, binary = _detect_grid_corners(
                    gray,
                    blur_ksize=blur_ksize,
                    threshold_method=threshold_method,
                    invert=invert,
                    max_corners=max_corners,
                    quality_level=quality_level,
                    min_distance=min_distance,
                    block_size=block_size,
                )

                detected_bgr = _draw_corners_overlay(gray, corners)
                pre_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                overlays[ch_name] = _encode_overlay_to_b64(detected_bgr)
                preview_layers[ch_name] = {
                    'preprocessed': _encode_overlay_to_b64(pre_bgr),
                    'binary_mask': _encode_overlay_to_b64(bin_bgr),
                    'detected': _encode_overlay_to_b64(detected_bgr),
                }
                feature_counts[ch_name] = {'corners': len(corners)}

            ch_details = ', '.join('{}: {} corners'.format(k, v['corners']) for k, v in feature_counts.items())
            summary = ('Detected grid intersections via blur→invert→threshold→Shi-Tomasi. '
                       'Channels: {}'.format(ch_details))

        elif method == 'grid_line_hough':
            hough_threshold = int(params.get('hough_threshold', 50))
            min_line_length = float(params.get('min_line_length', 80.0))
            max_line_gap = float(params.get('max_line_gap', 20.0))
            angle_tolerance = float(params.get('angle_tolerance', 15.0))
            merge_distance = float(params.get('merge_distance', 20.0))

            for ch_name, path in stage_files.items():
                img = tifffile.imread(str(path)).astype(np.float32)
                gray = _normalize_for_display(img)

                h_lines, v_lines, preprocessed, binary = _detect_grid_lines(
                    gray,
                    blur_ksize=blur_ksize,
                    threshold_method=threshold_method,
                    invert=invert,
                    hough_threshold=hough_threshold,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    angle_tolerance=angle_tolerance,
                    merge_distance=merge_distance,
                )

                intersections = _compute_line_intersections(h_lines, v_lines)
                detected_bgr = _draw_lines_overlay(gray, h_lines, v_lines, intersections if len(intersections) else None)
                inter_bgr = _draw_intersections_overlay(gray, intersections)
                pre_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                bin_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                overlays[ch_name] = _encode_overlay_to_b64(detected_bgr)
                preview_layers[ch_name] = {
                    'preprocessed': _encode_overlay_to_b64(pre_bgr),
                    'binary_mask': _encode_overlay_to_b64(bin_bgr),
                    'detected': _encode_overlay_to_b64(detected_bgr),
                    'intersections': _encode_overlay_to_b64(inter_bgr),
                }
                feature_counts[ch_name] = {
                    'h_lines': len(h_lines),
                    'v_lines': len(v_lines),
                    'intersections': len(intersections),
                }

            line_details = ', '.join(
                '{}:{:d}H+{:d}V({:d}pts)'.format(k, v['h_lines'], v['v_lines'], v['intersections'])
                for k, v in feature_counts.items()
            )
            summary = ('Detected grid lines via blur→invert→threshold→Hough (merged). '
                       'Channels: {}'.format(line_details))

        else:  # frequency_domain_fft (+ legacy 'phase_cross_correlation' alias)
            fft_mode = str(params.get('fft_mode', 'grid_detection'))
            window_fn = str(params.get('window_fn', 'hann'))
            min_spacing_px = float(params.get('min_spacing_px', 20.0))
            max_spacing_px = float(params.get('max_spacing_px', 200.0))
            peak_threshold_rel = float(params.get('peak_threshold', 0.3))
            enable_rotation = bool(params.get('enable_rotation', True))
            line_width_px = float(params.get('grid_line_width', 2.0))
            angle_tolerance_deg = float(params.get('angle_tolerance_deg', 20.0))

            for ch_name, path in stage_files.items():
                img = tifffile.imread(str(path)).astype(np.float32)

                if fft_mode == 'simple_correlation':
                    # Fallback: image-to-image shift detector (original PCC approach)
                    ref_path = stage_files.get(ref_channel)
                    if ref_path is None:
                        ref_path = next(iter(stage_files.values()))
                    ref_img = tifffile.imread(str(ref_path)).astype(np.float32)
                    pcc_bgr = _simple_pcc_preview(ref_img, img)
                    b64 = _encode_overlay_to_b64(pcc_bgr)
                    overlays[ch_name] = b64
                    preview_layers[ch_name] = {'fft_spectrum': b64, 'detected': b64}
                    feature_counts[ch_name] = {'mode': 'simple_correlation'}
                    continue

                # ── Full FFT Grid Detection Pipeline ──────────────────────────────
                # Step 1+2: Window → FFT → log-magnitude spectrum
                log_mag, F_shifted = _log_magnitude_spectrum(img, window_fn=window_fn)

                # Step 3+4: Detect H and V grid frequency peaks (sub-pixel refined)
                h_peak, v_peak = _find_grid_freq_peaks(
                    log_mag, F_shifted,
                    min_spacing_px=min_spacing_px,
                    max_spacing_px=max_spacing_px,
                    peak_threshold_rel=peak_threshold_rel,
                    enable_rotation=enable_rotation,
                    angle_tolerance_deg=angle_tolerance_deg,
                )

                # Spectrum overlay (Step 3 visualization)
                spectrum_bgr = _draw_fft_spectrum_overlay(log_mag, h_peak, v_peak)

                gray = _normalize_for_display(img)

                if h_peak is None and v_peak is None:
                    # No peaks found — show spectrum as only layer with a warning
                    b64_spec = _encode_overlay_to_b64(spectrum_bgr)
                    overlays[ch_name] = b64_spec
                    preview_layers[ch_name] = {'fft_spectrum': b64_spec, 'detected': b64_spec}
                    feature_counts[ch_name] = {
                        'h_spacing_px': None, 'v_spacing_px': None, 'intersections': 0,
                        'warning': 'No peaks found — try adjusting min/max spacing or threshold'
                    }
                    continue

                # Step 5: Synthetic grid template
                synth_u8 = _generate_synthetic_grid(
                    img.shape[0], img.shape[1], h_peak, v_peak, line_width_px)

                # Step 6: Phase already embedded in peaks (from FFT complex value)
                # Synthetic grid overlay on original image
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                synth_mask = synth_u8 > 0
                synth_bgr = gray_bgr.copy()
                synth_bgr[synth_mask] = [0, 220, 100]   # green grid overlay

                # Step 7: Intersection points
                intersections = np.empty((0, 2), dtype=np.float32)
                if h_peak is not None and v_peak is not None:
                    intersections = _compute_fft_grid_intersections(
                        img.shape[0], img.shape[1], h_peak, v_peak)

                inter_bgr = _draw_intersections_overlay(gray, intersections)

                # "All" composite: spectrum + intersection count badge on gray
                all_bgr = gray_bgr.copy()
                all_bgr[synth_mask] = [0, 180, 80]
                for xp, yp in intersections:
                    cv2.circle(all_bgr, (int(round(xp)), int(round(yp))), 4, (0, 255, 0), -1)

                overlays[ch_name] = _encode_overlay_to_b64(inter_bgr)
                preview_layers[ch_name] = {
                    'fft_spectrum':   _encode_overlay_to_b64(spectrum_bgr),
                    'synthetic_grid': _encode_overlay_to_b64(synth_bgr),
                    'intersections':  _encode_overlay_to_b64(inter_bgr),
                    'all':            _encode_overlay_to_b64(all_bgr),
                    'detected':       _encode_overlay_to_b64(inter_bgr),  # compat alias
                }
                feature_counts[ch_name] = {
                    'h_spacing_px': round(h_peak['spacing_px'], 1) if h_peak else None,
                    'v_spacing_px': round(v_peak['spacing_px'], 1) if v_peak else None,
                    'h_angle_deg':  round(h_peak['normal_angle_deg'] - 90, 1) if h_peak else None,
                    'v_angle_deg':  round(v_peak['normal_angle_deg'], 1) if v_peak else None,
                    'intersections': len(intersections),
                }

            if fft_mode == 'simple_correlation':
                summary = ("Simple phase cross-correlation map vs reference channel '{}'. "
                           "Sharp central peak = small shift. Wide peak = ambiguous.").format(ref_channel)
            else:
                fc_parts = []
                for k, v in feature_counts.items():
                    if isinstance(v.get('h_spacing_px'), float):
                        fc_parts.append('{}: {:.0f}×{:.0f}px grid, {:d} pts'.format(
                            k, v['h_spacing_px'] or 0, v['v_spacing_px'] or 0, v['intersections']))
                    else:
                        fc_parts.append('{}: no peaks'.format(k))
                summary = ('FFT grid detection (window={}, spacing {:.0f}–{:.0f}px). '
                           '{}').format(window_fn, min_spacing_px, max_spacing_px, '; '.join(fc_parts))

        return jsonify({
            'success': True,
            'data': {
                'method': method,
                'overlays': overlays,
                'preview_layers': preview_layers,
                'feature_counts': feature_counts,
                'summary': summary,
            }
        })

    except Exception as e:
        import traceback
        print('[DetectFeatures] ERROR: {}\n{}'.format(e, traceback.format_exc()))
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/input/align', methods=['POST'])
def align_position():
    """
    Run alignment on loaded position.
    
    Body:
    {
        "sample": "A2780Cis10",
        "position": "P1",
        "input_stage": "raw" | "contrast_enhance" | "step1" | ...,
        "ref_channel": "ch1",
        "method": "phase_cross_correlation",
        "transform": "EuclideanTransform"
    }
    """
    data = request.json or {}
    sample_name = data.get('sample')
    position_name = data.get('position')
    input_stage = data.get('input_stage', 'raw')
    ref_channel = data.get('ref_channel', 'ch1')
    method = data.get('method', 'phase_cross_correlation')
    transform = data.get('transform', 'EuclideanTransform')
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    position_key = f"{sample_name}/{position_name}"
    position_path = DATA_ROOT / sample_name / position_name
    
    if not position_path.exists():
        return jsonify({
            'success': False,
            'error': f'Position not found: {sample_name}/{position_name}'
        }), 404
    
    try:
        # Determine input files based on input_stage
        print(f"[Alignment] Starting alignment: sample={sample_name}, position={position_name}, input_stage={input_stage}")
        print(f"[Alignment] Preprocessing cache keys: {list(preprocessing_cache.keys())}")
        
        # NEW: Support per-channel preprocessing stages
        # If channel_stages is provided, use channel-specific stages; otherwise use global input_stage
        channel_stages = data.get('channel_stages', {})  # e.g., {"C1_ch1": "contrast_enhance", "C1_ch2": "raw"}
        use_per_channel_stages = bool(channel_stages)
        
        if use_per_channel_stages:
            print(f"[Alignment] Using per-channel preprocessing stages: {channel_stages}")
            # Load files from channel-specific stages
            input_files = {}
            channel_states = preprocessing_cache[position_key].get('channel_states', {})
            
            # Get all available channels from raw
            if position_key not in preprocessing_cache or 'raw' not in preprocessing_cache[position_key]:
                return jsonify({
                    'success': False,
                    'error': 'Raw stage not found. Please load position first.'
                }), 400
            
            raw_channels = preprocessing_cache[position_key]['raw'].keys()
            
            for channel_name in raw_channels:
                # Get stage for this channel (default to input_stage if not specified)
                channel_stage = channel_stages.get(channel_name, input_stage)
                
                # Find the latest preprocessed stage for this channel
                if channel_name in channel_states:
                    # Get the latest step for this channel
                    available_steps = list(channel_states[channel_name].keys())
                    if available_steps:
                        # Use the latest step if channel_stage is not explicitly set
                        if channel_name not in channel_stages:
                            latest_step = available_steps[-1]  # Use last applied step
                            channel_stage = latest_step
                            print(f"[Alignment] Channel {channel_name}: using latest step '{latest_step}'")
                        else:
                            # Use specified stage if it exists
                            if channel_stage in channel_states[channel_name]:
                                state = channel_states[channel_name][channel_stage]
                                input_files[channel_name] = Path(state['output_path'])
                                print(f"[Alignment] Channel {channel_name}: using stage '{channel_stage}'")
                                continue
                
                # Fallback: try to load from global stage cache
                if channel_stage in preprocessing_cache[position_key]:
                    stage_files = preprocessing_cache[position_key][channel_stage]
                    if channel_name in stage_files:
                        input_files[channel_name] = stage_files[channel_name]
                        print(f"[Alignment] Channel {channel_name}: loaded from stage '{channel_stage}'")
                    else:
                        # Final fallback: use raw
                        input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
                        print(f"[Alignment] Channel {channel_name}: fallback to raw")
                else:
                    # Final fallback: use raw
                    input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
                    print(f"[Alignment] Channel {channel_name}: fallback to raw (stage '{channel_stage}' not found)")
        else:
            # Legacy: use global input_stage for all channels
            if input_stage == 'raw':
                # Load from preprocessing cache (should be populated by load_position)
                if position_key not in preprocessing_cache or 'raw' not in preprocessing_cache[position_key]:
                    print(f"[Alignment] ERROR: Raw stage not found in cache for {position_key}")
                    print(f"[Alignment] Please load position first using /api/input/load_position")
                    return jsonify({
                        'success': False,
                        'error': f'Raw stage not found. Please load position first.'
                    }), 400
                
                input_files = preprocessing_cache[position_key]['raw']
                print(f"[Alignment] Loaded {len(input_files)} channels from raw cache: {list(input_files.keys())}")
            else:
                # Load from preprocessing cache
                if position_key not in preprocessing_cache or input_stage not in preprocessing_cache[position_key]:
                    print(f"[Alignment] ERROR: Cache miss for {position_key}/{input_stage}")
                    print(f"[Alignment] Available stages for {position_key}: {list(preprocessing_cache.get(position_key, {}).keys())}")
                    return jsonify({
                        'success': False,
                        'error': f'Input stage "{input_stage}" not found. Run preprocessing first.'
                    }), 400
                
                input_files = preprocessing_cache[position_key][input_stage]
                print(f"[Alignment] Loaded {len(input_files)} channels from cache: {list(input_files.keys())}")
        
        # Create output directory for aligned images
        output_dir = PROCESSING_OUTPUT / sample_name / position_name / f"aligned_from_{input_stage}"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"[Alignment] Processing {len(input_files)} channels for alignment")

        # ── Manual Diagonal: compute alignment from user-drawn box corners ────
        if method == 'manual_diagonal':
            diagonal_boxes = data.get('diagonal_boxes', {})
            if not diagonal_boxes:
                return jsonify({'success': False,
                                'error': 'manual_diagonal requires diagonal_boxes in request body.'}), 400

            # Load images
            images_dict = {}
            for ch, fp in input_files.items():
                img = tifffile.imread(str(fp))
                while img.ndim > 2 and 1 in img.shape:
                    img = np.squeeze(img)
                if img.ndim == 3:
                    img = img[0]
                images_dict[ch] = img

            transform_map = {'EuclideanTransform': 'euclidean', 'AffineTransform': 'affine', 'Translation': 'translation'}
            backend_transform = transform_map.get(transform, 'affine')

            out_files, shift_vectors, transformations = _manual_diagonal_align(
                images_dict, ref_channel, diagonal_boxes, backend_transform, output_dir
            )

            # Generate previews
            previews, stats = {}, {}
            for ch, fp in out_files.items():
                url = generate_preview_png(fp)
                if url:
                    previews[ch] = url
                try:
                    arr = tifffile.imread(str(fp))
                    while arr.ndim > 2 and 1 in arr.shape:
                        arr = np.squeeze(arr)
                    if arr.ndim == 3:
                        arr = arr[0]
                    stats[ch] = {
                        'min': float(arr.min()), 'max': float(arr.max()),
                        'mean': float(arr.mean()), 'median': float(np.median(arr)),
                        'std': float(arr.std()),
                        'p1': float(np.percentile(arr, 1)), 'p99': float(np.percentile(arr, 99)),
                        'shape': list(arr.shape), 'dtype': str(arr.dtype),
                    }
                except Exception as e:
                    print(f"[Alignment] Stats error for {ch}: {e}")

            aligned_stage_key = f"aligned_from_{input_stage}"
            preprocessing_cache.setdefault(position_key, {})[aligned_stage_key] = out_files

            return jsonify({'success': True, 'data': {
                'sample': sample_name, 'position': position_name,
                'input_stage': input_stage, 'ref_channel': ref_channel,
                'method': 'manual_diagonal', 'transform': transform,
                'previews': previews, 'stats': stats,
                'alignment_stats': {'method': 'manual_diagonal', 'error': None},
                'transformations': transformations,
                'shift_vectors': shift_vectors,
            }})
        # ── End manual diagonal ───────────────────────────────────────────────

        # Try to use real alignment, fallback to placeholder if it fails
        use_real_alignment = True
        alignment_error = None
        
        if use_real_alignment:
            try:
                # Import aligner
                from agents import Aligner
                import torch
                
                # Check if ref_channel exists in input_files
                if ref_channel not in input_files:
                    print(f"[Alignment] WARNING: Reference channel '{ref_channel}' not found in input files")
                    print(f"[Alignment] Available channels: {list(input_files.keys())}")
                    # Use first channel as fallback
                    ref_channel = list(input_files.keys())[0]
                    print(f"[Alignment] Using '{ref_channel}' as reference channel instead")
                
                # Check if ref_channel exists in input_files
                if ref_channel not in input_files:
                    print(f"[Alignment] WARNING: Reference channel '{ref_channel}' not found in input files")
                    print(f"[Alignment] Available channels: {list(input_files.keys())}")
                    # Use first channel as fallback
                    ref_channel = list(input_files.keys())[0]
                    print(f"[Alignment] Using '{ref_channel}' as reference channel instead")
                
                # Filter channels based on selected_channels if provided
                selected_channels = data.get('selected_channels', None)
                if selected_channels:
                    print(f"[Alignment] Filtering to selected channels: {selected_channels}")
                    # Only keep channels that are in the selected list
                    input_files = {k: v for k, v in input_files.items() if k in selected_channels}
                    if ref_channel not in input_files:
                        print(f"[Alignment] WARNING: Reference channel '{ref_channel}' not in selected channels")
                        print(f"[Alignment] Available selected channels: {list(input_files.keys())}")
                        if input_files:
                            ref_channel = list(input_files.keys())[0]
                            print(f"[Alignment] Using '{ref_channel}' as reference channel instead")
                        else:
                            raise ValueError("No channels selected for alignment")
                
                # Load all images
                print(f"[Alignment] Loading images for alignment...")
                images_dict = {}
                for channel_name, input_path in input_files.items():
                    print(f"[Alignment]   Loading {channel_name} from {input_path}")
                    img_array = tifffile.imread(str(input_path))
                    
                    # Handle multi-dimensional arrays
                    while img_array.ndim > 2 and 1 in img_array.shape:
                        img_array = np.squeeze(img_array)
                    if img_array.ndim == 3:
                        img_array = img_array[0]
                    
                    # Ensure 2D
                    if img_array.ndim != 2:
                        raise ValueError(f"Image {channel_name} is not 2D: shape={img_array.shape}")
                    
                    images_dict[channel_name] = img_array
                    print(f"[Alignment]     Loaded: shape={img_array.shape}, dtype={img_array.dtype}")
                
                # Use the selected input_stage images as BOTH transform estimation input and
                # final alignment output source. Do not switch back to raw images here.
                print(f"[Alignment] Using '{input_stage}' images as alignment input and output source")
                
                # Initialize aligner with bicubic interpolation for sharpness
                aligner_config = {
                    'feature_detector': 'superpoint',
                    'matcher': 'superglue',
                    'initial_transform': 'affine',
                    'fallback_transform': 'tps',
                    'residual_threshold': 2.0,
                    'max_features': 1024,
                    'match_threshold': 0.7,
                    'min_matches': 4,
                    'llm_qa_enabled': False,  # Disable LLM QA for API
                    'interpolation_mode': 'bicubic'  # Use bicubic for sharper results
                }
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Alignment] Using device: {device}")
                
                aligner = Aligner(aligner_config, device=device)
                
                # Map transform type from frontend to backend
                transform_map = {
                    'EuclideanTransform': 'euclidean',
                    'AffineTransform': 'affine',
                    'Translation': 'translation'
                }
                backend_transform = transform_map.get(transform, 'affine')
                
                # Run alignment with the selected method
                print(f"[Alignment] Running alignment with reference channel: {ref_channel}")
                print(f"[Alignment] Method: {method}, Transform: {backend_transform}")
                # Normalize method name: new UI method names → backend names
                method_alias_map = {
                    'phase_cross_correlation': 'phase_cross_correlation',
                    'pcc': 'phase_cross_correlation',
                    'phase cross correlation': 'phase_cross_correlation',
                    'frequency_domain_fft': 'phase_cross_correlation',   # FFT grid detection → PCC for actual warp
                    'grid_intersection_homography': 'feature_based',
                    'grid_line_hough': 'feature_based',
                    'feature_based': 'feature_based',
                }
                backend_method = method_alias_map.get(method.lower(), method.lower())
                method = backend_method  # overwrite for register_channels call below

                if method == 'phase_cross_correlation':
                    print(f"[Alignment] Using Phase Cross Correlation for alignment")
                else:
                    print(f"[Alignment] Using Feature-Based alignment (estimating on {input_stage} stage)")
                
                alignment_result = aligner.register_channels(
                    images_dict, 
                    anchor_channel=ref_channel,
                    method=method,
                    transform_type=backend_transform
                )
                
                if 'error' in alignment_result:
                    raise Exception(f"Alignment failed: {alignment_result['error']}")
                
                transformations = alignment_result.get('transformations', {})
                
                # Calculate shift vectors (dx, dy) for each channel from transformation matrices
                # This will be used for visualization in the QC overlay
                shift_vectors = {}
                for channel_name, trans_info in transformations.items():
                    transform_type = trans_info.get('type', 'identity')
                    if transform_type == 'identity':
                        shift_vectors[channel_name] = {
                            'dx': 0.0,
                            'dy': 0.0,
                            'magnitude': 0.0,
                            'type': 'identity',
                            'residual_error': float('inf'),
                            'num_matches': 0
                        }
                    else:
                        # Extract translation from affine transformation matrix
                        # Affine matrix format: [[a, b, tx], [c, d, ty], [0, 0, 1]]
                        transform_matrix = trans_info.get('transform', [])
                        if transform_matrix and len(transform_matrix) >= 2:
                            try:
                                tx = float(transform_matrix[0][2]) if len(transform_matrix[0]) > 2 else 0.0
                                ty = float(transform_matrix[1][2]) if len(transform_matrix[1]) > 2 else 0.0
                                magnitude = np.sqrt(tx**2 + ty**2)
                                shift_vectors[channel_name] = {
                                    'dx': tx,
                                    'dy': ty,
                                    'magnitude': magnitude,
                                    'type': transform_type,
                                    'residual_error': trans_info.get('residual_error', 0.0),
                                    'num_matches': trans_info.get('num_matches', 0)
                                }
                            except (IndexError, TypeError, ValueError) as e:
                                print(f"[Alignment] Error extracting shift for {channel_name}: {e}")
                                shift_vectors[channel_name] = {
                                    'dx': 0.0,
                                    'dy': 0.0,
                                    'magnitude': 0.0,
                                    'type': transform_type,
                                    'residual_error': trans_info.get('residual_error', 0.0),
                                    'num_matches': trans_info.get('num_matches', 0)
                                }
                        else:
                            # TPS or other non-affine transform - cannot extract simple dx/dy
                            shift_vectors[channel_name] = {
                                'dx': 0.0,
                                'dy': 0.0,
                                'magnitude': 0.0,
                                'type': transform_type,
                                'residual_error': trans_info.get('residual_error', 0.0),
                                'num_matches': trans_info.get('num_matches', 0),
                                'note': 'Non-affine transform (dx/dy not applicable)'
                            }
                
                # Ensure all channels in input_files have shift vectors
                # Reference channel has no shift
                shift_vectors[ref_channel] = {
                    'dx': 0.0,
                    'dy': 0.0,
                    'magnitude': 0.0,
                    'type': 'reference',
                    'residual_error': 0.0,
                    'num_matches': 0
                }
                
                # Add identity shift vectors for channels without transformations
                for channel_name in input_files.keys():
                    if channel_name not in shift_vectors:
                        shift_vectors[channel_name] = {
                            'dx': 0.0,
                            'dy': 0.0,
                            'magnitude': 0.0,
                            'type': 'identity',
                            'residual_error': float('inf'),
                            'num_matches': 0,
                            'note': 'No transformation (identity)'
                        }
                
                # Use aligned images computed from images_dict (selected input_stage)
                registered_images = alignment_result['registered_images']
                print(f"[Alignment] Using aligned images from input stage '{input_stage}'")
                
                print(f"[Alignment] Alignment complete! Registered {len(registered_images)} channels")
                
                # Save aligned images
                output_files = {}
                for channel_name, aligned_img in registered_images.items():
                    # Ensure proper dtype and verify dimensions
                    original_img = images_dict.get(channel_name)
                    
                    # Pixel-to-pixel consistency check
                    if original_img is not None:
                        if aligned_img.shape != original_img.shape:
                            print(f"[Alignment] WARNING: Shape mismatch for {channel_name}: "
                                  f"original={original_img.shape}, aligned={aligned_img.shape}")
                        original_dtype = original_img.dtype
                    else:
                        original_dtype = images_dict[channel_name].dtype if channel_name in images_dict else np.uint16
                    
                    # Convert to original dtype if needed
                    if aligned_img.dtype == np.float32:
                        if original_dtype == np.uint16:
                            aligned_img = np.clip(aligned_img * 65535.0, 0, 65535).astype(np.uint16)
                        elif original_dtype == np.uint8:
                            aligned_img = np.clip(aligned_img * 255.0, 0, 255).astype(np.uint8)
                        # If original was float32, keep it as float32
                    
                    output_path = output_dir / f"{channel_name}_aligned.tif"
                    tifffile.imwrite(str(output_path), aligned_img)
                    output_files[channel_name] = output_path
                    
                    # Log transformation info
                    if channel_name in transformations:
                        trans_info = transformations[channel_name]
                        print(f"[Alignment]   {channel_name}: {trans_info.get('type', 'unknown')} transform, "
                              f"residual={trans_info.get('residual_error', 0):.2f}px, "
                              f"matches={trans_info.get('num_matches', 0)}")
                    else:
                        print(f"[Alignment]   {channel_name}: identity transform (no alignment needed)")
                
            except ImportError as e:
                print(f"[Alignment] WARNING: Cannot import Aligner: {e}")
                print(f"[Alignment] Falling back to placeholder alignment")
                use_real_alignment = False
                alignment_error = str(e)
            except Exception as e:
                import traceback
                print(f"[Alignment] ERROR: Real alignment failed: {e}")
                traceback.print_exc()
                print(f"[Alignment] Falling back to placeholder alignment")
                use_real_alignment = False
                alignment_error = str(e)
        
        # Fallback to placeholder if real alignment failed or disabled
        if not use_real_alignment:
            print(f"[Alignment] Using placeholder alignment (copying images)")
            output_files = {}
            shift_vectors = {}
            for channel_name, input_path in input_files.items():
                print(f"[Alignment] Processing {channel_name} from {input_path}")
                img_array = tifffile.imread(str(input_path))
                
                # Handle multi-dimensional arrays
                while img_array.ndim > 2 and 1 in img_array.shape:
                    img_array = np.squeeze(img_array)
                if img_array.ndim == 3:
                    img_array = img_array[0]
                
                # Just copy (placeholder)
                aligned = img_array.copy()
                
                # Save aligned TIFF
                output_path = output_dir / f"{channel_name}_aligned.tif"
                tifffile.imwrite(str(output_path), aligned)
                output_files[channel_name] = output_path
                print(f"[Alignment]   Saved to {output_path}")
                
                # Create identity shift vectors for placeholder
                if channel_name == ref_channel:
                    shift_vectors[channel_name] = {
                        'dx': 0.0,
                        'dy': 0.0,
                        'magnitude': 0.0,
                        'type': 'reference',
                        'residual_error': 0.0,
                        'num_matches': 0
                    }
                else:
                    shift_vectors[channel_name] = {
                        'dx': 0.0,
                        'dy': 0.0,
                        'magnitude': 0.0,
                        'type': 'identity',
                        'residual_error': float('inf'),
                        'num_matches': 0,
                        'note': 'Placeholder alignment (no transform applied)'
                    }
        
        # Update cache with aligned results
        aligned_stage_key = f"aligned_from_{input_stage}"
        if position_key not in preprocessing_cache:
            preprocessing_cache[position_key] = {}
        preprocessing_cache[position_key][aligned_stage_key] = output_files
        
        # Generate previews and 16-bit stats
        previews = {}
        stats = {}
        # Build per-channel fixed display windows from input-stage images so that
        # aligned previews use the same visual scale as their source processed images.
        # Use robust percentiles to reduce outlier-driven brightness jumps.
        display_windows: Dict[str, Tuple[float, float]] = {}
        # images_dict is only defined when real alignment ran; skip display window
        # computation on the placeholder / fallback path.
        for channel_name, src_img in (images_dict if use_real_alignment else {}).items():
            try:
                v_low = float(np.percentile(src_img, 1))
                v_high = float(np.percentile(src_img, 99))
                if v_high <= v_low:
                    v_low = float(src_img.min())
                    v_high = float(src_img.max())
                display_windows[channel_name] = (v_low, v_high)
            except Exception as e:
                print(f"[Alignment] Failed to compute display window for {channel_name}: {e}")
        for channel_name, file_path in output_files.items():
            print(f"[Alignment] Generating preview and stats for {channel_name}: {file_path}")
            if channel_name in display_windows:
                w_min, w_max = display_windows[channel_name]
                preview_url = generate_preview_png(
                    file_path,
                    fixed_display_min=w_min,
                    fixed_display_max=w_max
                )
            else:
                preview_url = generate_preview_png(file_path)
            if preview_url:
                previews[channel_name] = preview_url
                
            # Compute accurate 16-bit stats for the aligned image
            try:
                img_array = tifffile.imread(str(file_path))
                while img_array.ndim > 2 and 1 in img_array.shape:
                    img_array = np.squeeze(img_array)
                if img_array.ndim == 3:
                    img_array = img_array[0]
                
                stats[channel_name] = {
                    'min': float(img_array.min()),
                    'max': float(img_array.max()),
                    'mean': float(img_array.mean()),
                    'median': float(np.median(img_array)),
                    'std': float(img_array.std()),
                    'p1': float(np.percentile(img_array, 1)),
                    'p99': float(np.percentile(img_array, 99)),
                    'shape': list(img_array.shape),
                    'dtype': str(img_array.dtype),
                }
            except Exception as e:
                print(f"[Alignment] Failed to compute stats for {channel_name}: {e}")
        
        # Prepare alignment stats
        alignment_stats = {
            'dx': 0.0,
            'dy': 0.0,
            'rotation': 0.0,
            'score': 1.0,
            'method': 'placeholder' if not use_real_alignment else 'feature_based',
            'error': alignment_error if alignment_error else None
        }
        
        return jsonify({
            'success': True,
            'data': {
                'sample': sample_name,
                'position': position_name,
                'input_stage': input_stage,
                'ref_channel': ref_channel,
                'method': method,
                'transform': transform,
                'previews': previews,
                'stats': stats, # NEW: Include accurate 16-bit stats
                'alignment_stats': alignment_stats,
                'transformations': transformations if use_real_alignment else {},
                'shift_vectors': shift_vectors if use_real_alignment or not use_real_alignment else {}
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Alignment error: {str(e)}'
        }), 500


def apply_contrast_enhancement(img_array: np.ndarray, method: str, params: dict) -> np.ndarray:
    """Apply contrast enhancement to image array."""
    if method == 'CLAHE':
        clip_limit = params.get('clip_limit', 2.0)
        tile_grid_size = params.get('tile_grid_size', 8)
        
        # Convert to 8-bit for CLAHE
        if img_array.dtype != np.uint8:
            img_min = float(img_array.min())
            img_max = float(img_array.max())
            if img_max > img_min:
                img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_8bit = np.zeros_like(img_array, dtype=np.uint8)
        else:
            img_8bit = img_array
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        enhanced = clahe.apply(img_8bit)
        
        # If original was 16-bit, convert back
        if img_array.dtype == np.uint16:
            enhanced = (enhanced.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
        
        return enhanced
    
    elif method == 'Linear Stretch':
        p_low = params.get('p_low', 1)
        p_high = params.get('p_high', 99)
        
        # Compute percentiles
        v_low = np.percentile(img_array, p_low)
        v_high = np.percentile(img_array, p_high)
        
        # Stretch
        if v_high > v_low:
            stretched = np.clip((img_array - v_low) / (v_high - v_low), 0, 1)
            
            # Convert back to original dtype range
            if img_array.dtype == np.uint16:
                stretched = (stretched * 65535).astype(np.uint16)
            elif img_array.dtype == np.uint8:
                stretched = (stretched * 255).astype(np.uint8)
            else:
                stretched = stretched.astype(img_array.dtype)
        else:
            stretched = img_array.copy()
        
        return stretched
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


@app.route('/api/input/preprocess', methods=['POST'])
def preprocess_position():
    """
    Apply preprocessing step to position with per-channel parameters.
    
    Body (NEW - per-channel support):
    {
        "sample": "A2780Cis10",
        "position": "P1",
        "from_stage": "raw",
        "step": "contrast_enhance" | "step1" | "step2" | "step3" | "step4",
        "channel_params": {
            // Per-channel parameters (NEW)
            "C1_ch1": {
                "method": "CLAHE",
                "clip_limit": 2.0,
                "tile_grid_size": 8
            },
            "C1_ch2": {
                "method": "Linear Stretch",
                "p_low": 1,
                "p_high": 99
            },
            // ... other channels
        },
        // Legacy support: if channel_params not provided, use global params for all channels
        "params": {
            "method": "CLAHE",
            "clip_limit": 2.0,
            "tile_grid_size": 8
        }
    }
    """
    print("=" * 80)
    print("[Preprocess] ========== NEW PREPROCESS REQUEST ==========")
    print("=" * 80)
    
    data = request.json or {}
    sample_name = data.get('sample')
    position_name = data.get('position')
    from_stage = data.get('from_stage', 'raw')
    step = data.get('step', 'step1')
    channel_params = data.get('channel_params', {})  # NEW: per-channel params
    global_params = data.get('params', {})  # Legacy: global params (fallback)
    
    print(f"[Preprocess] STEP 1: Request received")
    print(f"[Preprocess]   Sample: {sample_name}")
    print(f"[Preprocess]   Position: {position_name}")
    print(f"[Preprocess]   Step: {step}")
    print(f"[Preprocess]   Global from_stage: {from_stage}")
    print(f"[Preprocess]   Channel params keys: {list(channel_params.keys())}")
    print(f"[Preprocess]   Channel params: {channel_params}")
    print(f"[Preprocess]   Global params: {global_params}")
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    position_key = f"{sample_name}/{position_name}"
    position_path = DATA_ROOT / sample_name / position_name
    
    if not position_path.exists():
        return jsonify({
            'success': False,
            'error': f'Position not found: {sample_name}/{position_name}'
        }), 404
    
    try:
        # NEW: Support per-channel from_stage for true independence
        channel_from_stages = data.get('channel_from_stages', {})  # e.g., {"C1_ch1": "contrast_enhance", "C1_ch2": "raw"}
        use_per_channel_from_stage = bool(channel_from_stages)
        
        print(f"[Preprocess] STEP 2: Per-channel from_stage support")
        print(f"[Preprocess]   channel_from_stages: {channel_from_stages}")
        print(f"[Preprocess]   use_per_channel_from_stage: {use_per_channel_from_stage}")
        
        # Determine input files (per-channel or global)
        if use_per_channel_from_stage:
            print(f"[Preprocess] Using per-channel from_stage: {channel_from_stages}")
            input_files = {}
            channel_states = preprocessing_cache[position_key].get('channel_states', {})
            
            # Get all available channels from raw
            if position_key not in preprocessing_cache or 'raw' not in preprocessing_cache[position_key]:
                return jsonify({
                    'success': False,
                    'error': f'Raw stage not found in cache. Please load position first.'
                }), 400
            
            # Only load input files for channels that will be processed (in channel_params)
            # If channel_params is empty, process all channels (backward compatibility)
            channels_to_load = list(channel_params.keys()) if channel_params else list(preprocessing_cache[position_key]['raw'].keys())
            
            print(f"[Preprocess] STEP 3: Load input files (per-channel from_stage mode)")
            print(f"[Preprocess]   Position key: {position_key}")
            print(f"[Preprocess]   Channels to load: {channels_to_load}")
            print(f"[Preprocess]   Available raw channels: {list(preprocessing_cache[position_key]['raw'].keys())}")
            print(f"[Preprocess]   Preprocessing cache keys: {list(preprocessing_cache[position_key].keys())}")
            print(f"[Preprocess]   Channel states in cache: {list(channel_states.keys())}")
            
            for channel_name in channels_to_load:
                # Get channel-specific from_stage (fallback to global)
                channel_from_stage = channel_from_stages.get(channel_name, from_stage)
                
                # Load from appropriate stage for this channel
                if channel_from_stage == 'raw':
                    if channel_name in preprocessing_cache[position_key]['raw']:
                        input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
                        print(f"[Preprocess] Channel {channel_name}: loading from raw")
                    else:
                        print(f"[Preprocess] ERROR: Channel {channel_name} not found in raw cache")
                        continue
                else:
                    # Try to load from channel_states first (most accurate)
                    if channel_name in channel_states and channel_from_stage in channel_states[channel_name]:
                        state = channel_states[channel_name][channel_from_stage]
                        input_files[channel_name] = Path(state['output_path'])
                        print(f"[Preprocess] Channel {channel_name}: loading from channel_states[{channel_from_stage}]")
                    elif channel_from_stage in preprocessing_cache[position_key]:
                        # Fallback to global stage cache
                        stage_files = preprocessing_cache[position_key][channel_from_stage]
                        if channel_name in stage_files:
                            input_files[channel_name] = stage_files[channel_name]
                            print(f"[Preprocess] Channel {channel_name}: loading from global cache[{channel_from_stage}]")
                        else:
                            # Final fallback: use raw
                            if channel_name in preprocessing_cache[position_key]['raw']:
                                input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
                                print(f"[Preprocess] Channel {channel_name}: fallback to raw (not in {channel_from_stage})")
                            else:
                                print(f"[Preprocess] ERROR: Channel {channel_name} not found in raw cache, skipping")
                                continue
                    else:
                        # Final fallback: use raw
                        if channel_name in preprocessing_cache[position_key]['raw']:
                            input_files[channel_name] = preprocessing_cache[position_key]['raw'][channel_name]
                            print(f"[Preprocess] Channel {channel_name}: fallback to raw (stage {channel_from_stage} not found)")
                        else:
                            print(f"[Preprocess] ERROR: Channel {channel_name} not found in raw cache, skipping")
                            continue
            
            print(f"[Preprocess] Loaded {len(input_files)} input file(s): {list(input_files.keys())}")
        else:
            # Legacy: use global from_stage for all channels
            if from_stage == 'raw':
                # Load from preprocessing cache (should be populated by load_position)
                if position_key not in preprocessing_cache or 'raw' not in preprocessing_cache[position_key]:
                    return jsonify({
                        'success': False,
                        'error': f'Raw stage not found in cache. Please load position first.'
                    }), 400
                
                input_files = preprocessing_cache[position_key]['raw']
            else:
                # Load from preprocessing cache
                if position_key not in preprocessing_cache or from_stage not in preprocessing_cache[position_key]:
                    return jsonify({
                        'success': False,
                        'error': f'Stage "{from_stage}" not found in cache. Run previous steps first.'
                    }), 400
                
                input_files = preprocessing_cache[position_key][from_stage]
        
        # Create output directory
        output_dir = PROCESSING_OUTPUT / sample_name / position_name / step
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each channel with per-channel parameters
        output_files = {}
        channel_states = {}  # Track per-channel preprocessing state
        
        # Determine which channels to process
        # If channel_params is provided and non-empty, only process those channels
        # Otherwise, process all channels (backward compatibility)
        channels_to_process = list(channel_params.keys()) if channel_params else list(input_files.keys())
        
        print(f"[Preprocess] STEP 4: Determine channels to process")
        print(f"[Preprocess]   Input files loaded: {list(input_files.keys())}")
        print(f"[Preprocess]   Channel params provided: {list(channel_params.keys())}")
        print(f"[Preprocess]   Channels to process: {channels_to_process}")
        print(f"[Preprocess]   Number of channels to process: {len(channels_to_process)}")
        
        if len(channels_to_process) == 0:
            print(f"[Preprocess] ERROR: No channels to process!")
            print(f"[Preprocess]   This means channel_params is empty AND input_files is empty")
            return jsonify({
                'success': False,
                'error': 'No channels to process. Check channel_params and input_files.'
            }), 400
        
        print(f"[Preprocess] STEP 5: Processing channels")
        for channel_name in channels_to_process:
            print(f"[Preprocess]   --- Processing channel: {channel_name} ---")
            
            if channel_name not in input_files:
                print(f"[Preprocess]   ERROR: Channel {channel_name} not found in input_files!")
                print(f"[Preprocess]   Available input files: {list(input_files.keys())}")
                print(f"[Preprocess]   Skipping channel {channel_name}")
                continue
                
            input_path = input_files[channel_name]
            print(f"[Preprocess]   Input file path: {input_path}")
            print(f"[Preprocess]   Input file exists: {Path(input_path).exists()}")
            
            # Get channel-specific parameters (fallback to global if not provided)
            channel_specific_params = channel_params.get(channel_name, global_params)
            print(f"[Preprocess]   Channel params: {channel_specific_params}")
            
            # Determine actual from_stage used for this channel
            actual_from_stage = channel_from_stages.get(channel_name, from_stage) if use_per_channel_from_stage else from_stage
            
            # Load image
            img_array = tifffile.imread(str(input_path))
            
            # Handle multi-dimensional arrays
            while img_array.ndim > 2 and 1 in img_array.shape:
                img_array = np.squeeze(img_array)
            if img_array.ndim == 3:
                img_array = img_array[0]
            
            # Apply preprocessing based on step with channel-specific params
            if step == 'contrast_enhance':
                method = channel_specific_params.get('method', global_params.get('method', 'CLAHE'))
                processed = apply_contrast_enhancement(img_array, method, channel_specific_params)
            elif step == 'step1':
                # Background Subtraction using rolling ball algorithm
                strength = channel_specific_params.get('strength', global_params.get('strength', 50))
                # Convert strength (0-200) to radius in pixels (roughly 1-100 pixels)
                radius = max(1, int(strength / 2))
                print(f"[Preprocess]   Background subtraction: strength={strength}, radius={radius}")
                
                # Use morphological opening for background estimation
                try:
                    from skimage import morphology
                except ImportError:
                    print("[Preprocess]   WARNING: skimage not available, using cv2 for background subtraction")
                    # Fallback to cv2 morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
                    background = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
                    processed = img_array.astype(np.float32) - background.astype(np.float32)
                    processed = np.clip(processed, 0, None)
                else:
                    # Estimate background using morphological opening (rolling ball approximation)
                    # Create a disk-shaped structuring element
                    selem = morphology.disk(radius)
                    background = morphology.opening(img_array, selem)
                    
                    # Subtract background
                    processed = img_array.astype(np.float32) - background.astype(np.float32)
                    processed = np.clip(processed, 0, None)  # Ensure non-negative
                
                # Preserve original dtype
                if img_array.dtype == np.uint8:
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
                elif img_array.dtype == np.uint16:
                    processed = np.clip(processed, 0, 65535).astype(np.uint16)
                else:
                    processed = processed.astype(img_array.dtype)
                    
            elif step == 'step2':
                # Contrast clipping (placeholder - just copy for now)
                processed = img_array.copy()
            elif step == 'step3':
                # Gaussian Blur
                sigma = channel_specific_params.get('sigma', global_params.get('sigma', 1.0))
                print(f"[Preprocess]   Gaussian blur: sigma={sigma}")
                
                # Use cv2.GaussianBlur (more efficient and always available)
                # Calculate kernel size from sigma (should be odd and ~6*sigma)
                ksize = int(6 * sigma) | 1  # Make odd
                if ksize < 3:
                    ksize = 3
                elif ksize > 31:
                    ksize = 31  # cv2 limit
                
                # Apply Gaussian blur
                processed = cv2.GaussianBlur(img_array, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
                
                # Preserve original dtype (cv2.GaussianBlur may change it)
                if img_array.dtype == np.uint8:
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
                elif img_array.dtype == np.uint16:
                    processed = np.clip(processed, 0, 65535).astype(np.uint16)
                else:
                    processed = processed.astype(img_array.dtype)
            elif step == 'step4':
                # Connect fibrils (placeholder - just copy for now)
                processed = img_array.copy()
            else:
                processed = img_array.copy()
            
            # Save processed TIFF
            output_path = output_dir / f"{channel_name}.tif"
            print(f"[Preprocess]   Saving {channel_name} to {output_path}")
            print(f"[Preprocess]   Processed image shape: {processed.shape}, dtype: {processed.dtype}")
            tifffile.imwrite(str(output_path), processed)
            
            # Verify file was saved
            if not output_path.exists():
                print(f"[Preprocess]   ERROR: File was not saved: {output_path}")
                continue
            
            file_size = output_path.stat().st_size
            print(f"[Preprocess]   ✓ Saved {channel_name}: exists={output_path.exists()}, size={file_size} bytes")
            output_files[channel_name] = output_path
            print(f"[Preprocess]   Added to output_files: {channel_name} -> {output_path}")
            
            # Store per-channel preprocessing state
            # Record the actual from_stage used for this channel
            channel_states[channel_name] = {
                'step': step,
                'from_stage': actual_from_stage,  # Store actual from_stage used for this channel
                'params': channel_specific_params,
                'output_path': str(output_path)
            }
        
        # Update cache with per-channel state tracking
        if position_key not in preprocessing_cache:
            preprocessing_cache[position_key] = {}
        
        # Store per-channel preprocessing state
        if 'channel_states' not in preprocessing_cache[position_key]:
            preprocessing_cache[position_key]['channel_states'] = {}
        
        # Update per-channel state
        for channel_name, state in channel_states.items():
            if channel_name not in preprocessing_cache[position_key]['channel_states']:
                preprocessing_cache[position_key]['channel_states'][channel_name] = {}
            preprocessing_cache[position_key]['channel_states'][channel_name][step] = state
        
        # Also store file paths by step (for backward compatibility)
        preprocessing_cache[position_key][step] = output_files
        
        print(f"[Preprocess] STEP 6: Generate previews")
        print(f"[Preprocess]   Output files to generate previews for: {list(output_files.keys())}")
        previews = {}
        for channel_name, file_path in output_files.items():
            print(f"[Preprocess]   --- Generating preview for {channel_name} ---")
            print(f"[Preprocess]     File path: {file_path}")
            
            # Verify file exists before generating preview
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"[Preprocess]     ERROR: File does not exist: {file_path}")
                continue
            
            file_size = file_path_obj.stat().st_size
            file_mtime = file_path_obj.stat().st_mtime
            print(f"[Preprocess]     File exists: True, size: {file_size} bytes, mtime: {file_mtime}")
            
            # Force preview regeneration by deleting old preview if it exists
            # This ensures we get a fresh preview even if cache key matches
            preview_url = generate_preview_png(file_path_obj)
            if preview_url:
                previews[channel_name] = preview_url
                print(f"[Preprocess]     ✓ Preview URL: {preview_url}")
            else:
                print(f"[Preprocess]     ✗ Failed to generate preview for {channel_name}")
                import traceback
                traceback.print_exc()
        
        print(f"[Preprocess] STEP 7: Summary")
        print(f"[Preprocess]   Channels processed: {len(output_files)}")
        print(f"[Preprocess]   Output files: {list(output_files.keys())}")
        print(f"[Preprocess]   Previews generated: {len(previews)}")
        print(f"[Preprocess]   Preview channels: {list(previews.keys())}")
        print(f"[Preprocess]   Channel states: {list(channel_states.keys())}")
        print(f"[Preprocess] Final previews dict: {previews}")
        
        # Update preprocessing session
        print(f"[Preprocess] STEP 8: Update preprocessing session")
        try:
            session = get_session(sample_name, position_name)
            session.add_step(
                step_name=step,
                step_params=global_params,
                input_stage=from_stage,
                output_stage=step,
                channel_params=channel_params
            )
            version_hash = session.session_data.get('version_hash')
            print(f"[Preprocess]   Session updated, version hash: {version_hash}")
        except Exception as e:
            print(f"[Preprocess]   WARNING: Failed to update session: {e}")
            version_hash = None
        
        print("=" * 80)
        
        return jsonify({
            'success': True,
            'data': {
                'sample': sample_name,
                'position': position_name,
                'step': step,
                'from_stage': from_stage,
                'previews': previews,
                'channel_states': channel_states,  # NEW: return per-channel state
                'version_hash': version_hash  # NEW: return version hash
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Preprocessing failed: {str(e)}'
        }), 500


@app.route('/api/input/preprocess/state', methods=['GET'])
def get_preprocessing_state():
    """
    Get per-channel preprocessing state for a position.
    
    Query params:
    - sample: Sample name
    - position: Position name
    
    Returns:
    {
        "success": true,
        "data": {
            "channel_states": {
                "C1_ch1": {
                    "contrast_enhance": {
                        "step": "contrast_enhance",
                        "from_stage": "raw",
                        "params": {...},
                        "output_path": "..."
                    },
                    // ... other steps
                },
                // ... other channels
            }
        }
    }
    """
    sample_name = request.args.get('sample')
    position_name = request.args.get('position')
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    position_key = f"{sample_name}/{position_name}"
    
    if position_key not in preprocessing_cache:
        return jsonify({
            'success': False,
            'error': 'Position not loaded. Please load position first.'
        }), 404
    
    channel_states = preprocessing_cache[position_key].get('channel_states', {})
    
    return jsonify({
        'success': True,
        'data': {
            'sample': sample_name,
            'position': position_name,
            'channel_states': channel_states
        }
    })


@app.route('/api/input/preprocess/final', methods=['GET'])
def get_processed_final():
    """
    Get the final processed preview and stats for a position.
    This returns the output after ALL preprocessing steps are applied.
    
    Query params:
    - sample: Sample name
    - position: Position name
    - channel: Optional channel name (if not provided, returns all channels)
    
    Returns:
    {
        "success": true,
        "data": {
            "previews": {channel: preview_url},
            "stats": {channel: {min, max, mean, median, p1, p99, ...}},
            "version_hash": "...",
            "final_stage": "..."
        }
    }
    """
    sample_name = request.args.get('sample')
    position_name = request.args.get('position')
    channel_name = request.args.get('channel')  # Optional
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    position_key = f"{sample_name}/{position_name}"
    
    # Try to restore cache from disk if not in memory
    if position_key not in preprocessing_cache:
        print(f"[Preprocess/Final] Position {position_key} not in cache, attempting to restore from disk...")
        restore_preprocessing_cache_from_disk(sample_name, position_name)
    
    # If position not in cache after restore attempt, return empty result (no preprocessing done yet)
    if position_key not in preprocessing_cache:
        print(f"[Preprocess/Final] Position {position_key} not in cache, returning empty result")
        return jsonify({
            'success': True,
            'data': {
                'previews': {},
                'stats': {},
                'version_hash': None,
                'final_stage': 'raw',
            }
        })
    
    try:
        # Get session to find final stage
        session = get_session(sample_name, position_name)
        final_stage = session.get_final_stage()
        
        if not final_stage:
            # No preprocessing done yet, return raw previews
            print(f"[Preprocess/Final] No preprocessing done, returning raw previews")
            raw_channels = preprocessing_cache[position_key].get('raw', {})
            previews = {}
            stats = {}
            
            for ch, file_path in raw_channels.items():
                if channel_name and ch != channel_name:
                    continue
                    
                if file_path.exists():
                    preview_url = generate_preview_png(file_path)
                    if preview_url:
                        previews[ch] = preview_url
                    
                    # Compute stats
                    try:
                        img_array = tifffile.imread(str(file_path))
                        while img_array.ndim > 2 and 1 in img_array.shape:
                            img_array = np.squeeze(img_array)
                        if img_array.ndim == 3:
                            img_array = img_array[0]
                        
                        stats[ch] = {
                            'min': float(img_array.min()),
                            'max': float(img_array.max()),
                            'mean': float(img_array.mean()),
                            'median': float(np.median(img_array)),
                            'std': float(img_array.std()),
                            'p1': float(np.percentile(img_array, 1)),
                            'p99': float(np.percentile(img_array, 99)),
                            'shape': list(img_array.shape),
                            'dtype': str(img_array.dtype),
                        }
                    except Exception as e:
                        print(f"[Preprocess/Final] Failed to compute stats for {ch}: {e}")
            
            return jsonify({
                'success': True,
                'data': {
                    'previews': previews,
                    'stats': stats,
                    'version_hash': None,
                    'final_stage': 'raw',
                }
            })
        
        # Get channel states to find final output files
        channel_states = preprocessing_cache[position_key].get('channel_states', {})
        
        # Determine which channels to process
        if channel_name:
            channels_to_process = [channel_name]
        else:
            # Get all channels from raw
            raw_channels = preprocessing_cache[position_key].get('raw', {})
            channels_to_process = list(raw_channels.keys())
        
        previews = {}
        stats = {}
        
        for ch in channels_to_process:
            # Find the final output file for this channel
            final_file = None
            
            if ch in channel_states and final_stage in channel_states[ch]:
                # Use channel-specific final stage
                state = channel_states[ch][final_stage]
                final_file = Path(state['output_path'])
            elif final_stage in preprocessing_cache[position_key]:
                # Fallback to global stage cache
                stage_files = preprocessing_cache[position_key][final_stage]
                if ch in stage_files:
                    final_file = stage_files[ch]
            else:
                # Fallback to raw
                if ch in preprocessing_cache[position_key].get('raw', {}):
                    final_file = preprocessing_cache[position_key]['raw'][ch]
            
            if final_file and final_file.exists():
                print(f"[Preprocess/Final] Processing channel {ch}, file: {final_file}")
                # Generate preview
                preview_url = generate_preview_png(final_file)
                if preview_url:
                    previews[ch] = preview_url
                    print(f"[Preprocess/Final] Generated preview for {ch}: {preview_url}")
                else:
                    print(f"[Preprocess/Final] Failed to generate preview for {ch}")
                
                # Compute stats from the actual image
                try:
                    img_array = tifffile.imread(str(final_file))
                    # Handle multi-dimensional
                    while img_array.ndim > 2 and 1 in img_array.shape:
                        img_array = np.squeeze(img_array)
                    if img_array.ndim == 3:
                        img_array = img_array[0]
                    
                    # Compute statistics
                    stats[ch] = {
                        'min': float(img_array.min()),
                        'max': float(img_array.max()),
                        'mean': float(img_array.mean()),
                        'median': float(np.median(img_array)),
                        'std': float(img_array.std()),
                        'p1': float(np.percentile(img_array, 1)),
                        'p99': float(np.percentile(img_array, 99)),
                        'shape': list(img_array.shape),
                        'dtype': str(img_array.dtype),
                    }
                    print(f"[Preprocess/Final] Computed stats for {ch}: mean={stats[ch]['mean']:.2f}, shape={stats[ch]['shape']}")
                except Exception as e:
                    print(f"[Preprocess/Final] Failed to compute stats for {ch}: {e}")
        
        return jsonify({
            'success': True,
            'data': {
                'previews': previews,
                'stats': stats,
                'version_hash': session.session_data.get('version_hash'),
                'final_stage': final_stage,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get processed final: {str(e)}'
        }), 500


@app.route('/api/input/preprocess/session', methods=['GET'])
def get_preprocessing_session():
    """
    Get the full preprocessing session data for a position.
    Returns the complete pipeline history, parameters, and state.
    
    Query params:
    - sample: Sample name
    - position: Position name
    
    Returns:
    {
        "success": true,
        "data": {
            "session": {
                "sample": "...",
                "position": "...",
                "version_hash": "...",
                "num_steps": 3,
                "pipeline": [
                    {
                        "step_name": "contrast_enhance",
                        "step_params": {...},
                        "input_stage": "raw",
                        "output_stage": "contrast_enhance",
                        "timestamp": "...",
                        "channel_params": {...}
                    },
                    ...
                ],
                "final_stage": "step3",
                "created_at": "...",
                "updated_at": "..."
            }
        }
    }
    """
    sample_name = request.args.get('sample')
    position_name = request.args.get('position')
    
    if not sample_name or not position_name:
        return jsonify({
            'success': False,
            'error': 'Missing sample or position parameter'
        }), 400
    
    try:
        session = get_session(sample_name, position_name)
        session_summary = session.get_pipeline_summary()
        
        return jsonify({
            'success': True,
            'data': {
                'session': session_summary
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get preprocessing session: {str(e)}'
        }), 500


@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """
    Start pipeline execution.
    
    Body:
    {
        "samples": ["1", "2"],  // Optional, if not provided runs all
        "step": null,  // Optional: "inspection", "alignment", "analysis"
        "config": null  // Optional: path to config file
    }
    """
    runner = get_runner()
    
    if runner.is_running():
        return jsonify({
            'success': False,
            'error': 'Pipeline is already running'
        }), 400
    
    data = request.json or {}
    samples = data.get('samples')
    step = data.get('step')
    config_path = data.get('config')
    
    # Register callbacks for progress and logs
    def progress_callback(progress_data):
        broadcast_to_clients({
            'type': 'progress',
            'data': progress_data
        })
    
    def log_callback(log_message):
        broadcast_to_clients({
            'type': 'log',
            'message': log_message
        })
    
    runner.register_progress_callback(progress_callback)
    runner.register_log_callback(log_callback)
    
    # Start pipeline
    if step:
        # Run single step
        step_enum = {
            'inspection': PipelineStep.INSPECTION,
            'alignment': PipelineStep.ALIGNMENT,
            'analysis': PipelineStep.ANALYSIS
        }.get(step)
        
        if not step_enum or not samples or len(samples) != 1:
            return jsonify({
                'success': False,
                'error': 'Single step execution requires exactly one sample'
            }), 400
        
        success = runner.run_step(step_enum, samples[0], config_path)
    else:
        # Run full pipeline
        success = runner.run_all(samples, config_path)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Pipeline started'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to start pipeline'
        }), 500


@app.route('/api/pipeline/status', methods=['GET'])
def get_pipeline_status():
    """Get current pipeline status"""
    runner = get_runner()
    status = runner.get_status()
    
    return jsonify({
        'success': True,
        'data': status
    })


@app.route('/api/pipeline/stop', methods=['POST'])
def stop_pipeline():
    """Stop running pipeline"""
    runner = get_runner()
    
    if not runner.is_running():
        return jsonify({
            'success': False,
            'error': 'Pipeline is not running'
        }), 400
    
    runner.stop()
    
    return jsonify({
        'success': True,
        'message': 'Pipeline stop requested'
    })


@app.route('/api/pipeline/logs', methods=['GET'])
def get_pipeline_logs():
    """Get recent pipeline logs"""
    runner = get_runner()
    limit = request.args.get('limit', 100, type=int)
    logs = runner.get_logs(limit)
    
    return jsonify({
        'success': True,
        'data': logs
    })


# ============================================================================
# EXOSOME DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/exosome/segment', methods=['POST'])
def exosome_segment():
    """
    Segment exosomes using SAM.
    
    Body:
    {
        "sample": "A2780Cis10",
        "position": "P1",
        "channel": "C1_ch1",
        "mode": "box" | "point",
        "prompts": {
            "box": [x1, y1, x2, y2]  # for box mode
            OR
            "points": [[x, y], ...],  # for point mode
            "labels": [1, 0, ...]  # 1=positive, 0=negative
        },
        "checkpoint_path": "/path/to/sam_checkpoint.pth",
        "model_type": "sam_vit_h" | "sam_vit_l" | "sam_vit_b",
        "device": "auto" | "cuda" | "cpu",
        "score_thresh": 0.0,
        "min_area": 10,
        "max_area": 10000,
        "remove_small_objects": true,
        "fill_holes": false
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "masks": [[[bool, ...], ...], ...],  # List of 2D boolean arrays
            "scores": [float, ...],
            "bboxes": [[x1, y1, x2, y2], ...],
            "centroids": [[x, y], ...],
            "detections": [
                {
                    "area": float,
                    "centroid": [x, y],
                    "bbox": [x1, y1, x2, y2]
                },
                ...
            ]
        }
    }
    """
    try:
        data = request.json or {}
        sample_name = data.get('sample')
        position_name = data.get('position')
        channel_name = data.get('channel')
        method = data.get('method', 'sam')  # 'sam' or 'blob'
        
        min_area = data.get('min_area', 10)
        max_area = data.get('max_area', 10000)
        remove_small_objects = data.get('remove_small_objects', True)
        fill_holes = data.get('fill_holes', False)
        
        if not sample_name or not position_name or not channel_name:
            return jsonify({
                'success': False,
                'error': 'sample, position, and channel are required'
            }), 400
        
        # Load image from preprocessing cache or raw
        position_key = f"{sample_name}/{position_name}"
        
        # For blob detection, prefer raw images; for SAM, prefer processed
        image_path = None
        if position_key in preprocessing_cache:
            if method == 'blob':
                # Blob detection: prefer raw images
                if 'raw' in preprocessing_cache[position_key] and channel_name in preprocessing_cache[position_key]['raw']:
                    image_path = preprocessing_cache[position_key]['raw'][channel_name]
                else:
                    # Fallback to processed if raw not available
                    for stage in ['processed', 'contrast_enhance', 'step1', 'step2', 'step3', 'step4']:
                        if stage in preprocessing_cache[position_key] and channel_name in preprocessing_cache[position_key][stage]:
                            image_path = preprocessing_cache[position_key][stage][channel_name]
                            break
            else:
                # SAM: prefer processed images
                for stage in ['processed', 'contrast_enhance', 'step1', 'step2', 'step3', 'step4']:
                    if stage in preprocessing_cache[position_key] and channel_name in preprocessing_cache[position_key][stage]:
                        image_path = preprocessing_cache[position_key][stage][channel_name]
                        break
                
                # Fallback to raw
                if not image_path and 'raw' in preprocessing_cache[position_key] and channel_name in preprocessing_cache[position_key]['raw']:
                    image_path = preprocessing_cache[position_key]['raw'][channel_name]
        
        if not image_path or not image_path.exists():
            return jsonify({
                'success': False,
                'error': f'Image not found for {sample_name}/{position_name}/{channel_name}'
            }), 404
        
        # Load image
        print(f"[Exosome Detection] Loading image: {image_path}")
        print(f"[Exosome Detection] Method: {method}, Image shape will be checked after loading")
        image_array = tifffile.imread(str(image_path))
        print(f"[Exosome Detection] Image loaded: shape={image_array.shape}, dtype={image_array.dtype}, min={image_array.min()}, max={image_array.max()}")
        
        # Route to appropriate detection method
        if method == 'sam':
            mode = data.get('mode', 'box')
            prompts = data.get('prompts', {})
            checkpoint_path = data.get('checkpoint_path')
            model_type = data.get('model_type', 'sam_vit_h')
            device = data.get('device', 'auto')
            score_thresh = data.get('score_thresh', 0.0)
            
            if not checkpoint_path:
                return jsonify({
                    'success': False,
                    'error': 'checkpoint_path is required for SAM method'
                }), 400
            
            # Import SAM service
            try:
                from exosome_detection import segment_exosomes
            except ImportError as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to import SAM service: {e}. Make sure segment_anything is installed.'
                }), 500
            
            # Run SAM segmentation
            result = segment_exosomes(
                image=image_array,
                mode=mode,
                prompts=prompts,
                ckpt_path=checkpoint_path,
                model_type=model_type,
                device=device,
                score_thresh=score_thresh,
                min_area=min_area,
                max_area=max_area,
                remove_small_objects=remove_small_objects,
                fill_holes=fill_holes,
            )
        elif method == 'blob':
            threshold = data.get('threshold', 0.5)
            min_circularity = data.get('min_circularity', 0.3)
            max_circularity = data.get('max_circularity', 1.0)
            min_inertia_ratio = data.get('min_inertia_ratio', 0.3)
            
            # Import blob detection service
            try:
                from exosome_detection import detect_blobs
            except ImportError as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to import blob detection service: {e}'
                }), 500
            
            # Run blob detection
            print(f"[Exosome Detection] Running blob detection with threshold={threshold}, min_area={min_area}, max_area={max_area}")
            try:
                result = detect_blobs(
                    image=image_array,
                    threshold=threshold,
                    min_area=min_area,
                    max_area=max_area,
                    min_circularity=min_circularity,
                    max_circularity=max_circularity,
                    min_inertia_ratio=min_inertia_ratio,
                    remove_small_objects=remove_small_objects,
                    fill_holes=fill_holes,
                )
                print(f"[Exosome Detection] Blob detection completed: found {len(result.get('detections', []))} blobs")
            except Exception as e:
                print(f"[Exosome Detection] ERROR in blob detection: {e}")
                import traceback
                traceback.print_exc()
                raise
        elif method == 'random_forest':
            annotations = data.get('annotations', [])  # List of {points: [[x,y],...], label: 0 or 1}
            confidence_threshold = data.get('confidence_threshold', 0.5)
            apply_morphology = data.get('apply_morphology', False)
            n_estimators = data.get('n_estimators', 100)
            
            if not annotations or len(annotations) == 0:
                return jsonify({
                    'success': False,
                    'error': 'annotations are required for Random Forest method. Please annotate some pixels first.'
                }), 400
            
            # Import Random Forest service
            try:
                from exosome_detection import segment_with_random_forest
            except ImportError as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to import Random Forest service: {e}'
                }), 500
            
            # Run Random Forest segmentation
            print(f"[Exosome Detection] Running Random Forest with {len(annotations)} annotation groups, confidence_threshold={confidence_threshold}")
            try:
                result = segment_with_random_forest(
                    image=image_array,
                    annotations=annotations,
                    confidence_threshold=confidence_threshold,
                    min_area=min_area,
                    apply_morphology=apply_morphology,
                    n_estimators=n_estimators,
                )
                print(f"[Exosome Detection] Random Forest completed: found {len(result.get('detections', []))} objects")
            except Exception as e:
                print(f"[Exosome Detection] ERROR in Random Forest: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown detection method: {method}. Use "sam", "blob", or "random_forest".'
            }), 400
        
        # For Random Forest, handle probability map separately
        if method == 'random_forest':
            # Random Forest returns probability_map, binary_mask, overlay, detections
            # Don't optimize masks for Random Forest - they're already optimized
            print(f"[Exosome Detection] Serializing Random Forest response...")
            response_data = {
                'success': True,
                'data': result
            }
            return jsonify(response_data)
        
        # Optimize response: masks are too large to send for many detections
        # Calculate approximate size
        num_detections = len(result.get('detections', []))
        masks = result.get('masks', [])
        
        if masks and len(masks) > 0:
            # Estimate size: each mask is HxW boolean array
            try:
                if isinstance(masks[0], list) and len(masks[0]) > 0:
                    mask_height = len(masks[0])
                    mask_width = len(masks[0][0]) if isinstance(masks[0][0], list) else 0
                    mask_size = mask_height * mask_width
                else:
                    mask_size = 0
            except:
                mask_size = 1024 * 1024  # Default estimate
            
            total_size_mb = (num_detections * mask_size) / (1024 * 1024)
            print(f"[Exosome Detection] Response size estimate: {total_size_mb:.2f} MB for {num_detections} masks ({mask_size} pixels each)")
            
            # If response is too large (>50MB), don't send masks - only metadata
            # Frontend can draw simple visualizations from bboxes/centroids
            MAX_RESPONSE_SIZE_MB = 50
            if total_size_mb > MAX_RESPONSE_SIZE_MB:
                print(f"[Exosome Detection] WARNING: Response too large ({total_size_mb:.2f} MB). Omitting masks to prevent timeout.")
                print(f"[Exosome Detection] Sending detection metadata only (bboxes, centroids, areas). Frontend can visualize from bboxes.")
                
                # Create response without masks but with all detection metadata
                optimized_result = {
                    'masks': [],  # Empty - too large to send
                    'scores': result.get('scores', []),
                    'bboxes': result.get('bboxes', []),
                    'centroids': result.get('centroids', []),
                    'detections': result.get('detections', []),
                    '_warning': f'Masks omitted due to large response size ({total_size_mb:.2f} MB). Use bboxes/centroids for visualization.',
                    '_total_detections': num_detections,
                    '_masks_omitted': True,
                }
                
                print(f"[Exosome Detection] Serializing optimized response (metadata only)...")
                response_data = {
                    'success': True,
                    'data': optimized_result
                }
            else:
                print(f"[Exosome Detection] Serializing full response with masks...")
                response_data = {
                    'success': True,
                    'data': result
                }
        else:
            print(f"[Exosome Detection] Serializing response (no masks)...")
            response_data = {
                'success': True,
                'data': result
            }
        
        print(f"[Exosome Detection] Sending response...")
        return jsonify(response_data)
        
    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': f'File not found: {e}'
        }), 404
    except ImportError as e:
        return jsonify({
            'success': False,
            'error': f'SAM not available: {e}. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git'
        }), 500
    except Exception as e:
        app.logger.error(f"Exosome segmentation failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/exosome/export', methods=['POST'])
def exosome_export():
    """
    Export exosome detection results.
    
    Body:
    {
        "sample": "A2780Cis10",
        "position": "P1",
        "channel": "C1_ch1",
        "detections": [...],
        "masks": [[[bool, ...], ...], ...],
        "scores": [float, ...],
        "settings": {...}
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "paths": ["/path/to/overlay.png", "/path/to/masks.npz", "/path/to/results.csv", "/path/to/run.json"]
        }
    }
    """
    try:
        data = request.json or {}
        sample_name = data.get('sample')
        position_name = data.get('position')
        channel_name = data.get('channel')
        detections = data.get('detections', [])
        masks = data.get('masks', [])
        scores = data.get('scores', [])
        settings = data.get('settings', {})
        
        if not sample_name or not position_name or not channel_name:
            return jsonify({
                'success': False,
                'error': 'sample, position, and channel are required'
            }), 400
        
        # Create output directory
        output_dir = OUTPUT_ROOT / 'exosome_detection' / sample_name / position_name / channel_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate timestamp for this run
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save overlay PNG (if we have the original image)
        overlay_path = output_dir / f"{channel_name}_overlay_{timestamp}.png"
        position_key = f"{sample_name}/{position_name}"
        image_path = None
        
        if position_key in preprocessing_cache:
            for stage in ['processed', 'contrast_enhance', 'step1', 'step2', 'step3', 'step4', 'raw']:
                if stage in preprocessing_cache[position_key] and channel_name in preprocessing_cache[position_key][stage]:
                    image_path = preprocessing_cache[position_key][stage][channel_name]
                    break
        
        if image_path and image_path.exists():
            try:
                # Load image and create overlay
                image_array = tifffile.imread(str(image_path))
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB) if len(image_array.shape) == 2 else image_array
                if image_rgb.dtype != np.uint8:
                    image_rgb = ((image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min() + 1e-10) * 255).astype(np.uint8)
                
                # Draw masks on image
                overlay = image_rgb.copy()
                for i, mask_list in enumerate(masks):
                    mask = np.array(mask_list, dtype=bool)
                    if mask.shape != overlay.shape[:2]:
                        continue
                    color = np.array([255, 0, 0], dtype=np.uint8)  # Red overlay
                    overlay[mask] = (overlay[mask] * 0.6 + color * 0.4).astype(np.uint8)
                
                # Draw bounding boxes
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Save overlay
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as e:
                app.logger.warning(f"Failed to create overlay: {e}")
        
        # 2. Save masks as NPZ
        masks_path = output_dir / f"{channel_name}_masks_{timestamp}.npz"
        try:
            masks_array = np.array([np.array(m, dtype=bool) for m in masks])
            np.savez_compressed(str(masks_path), masks=masks_array, scores=np.array(scores))
        except Exception as e:
            app.logger.warning(f"Failed to save masks: {e}")
        
        # 3. Save results CSV
        csv_path = output_dir / f"{channel_name}_results_{timestamp}.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'area', 'centroid_x', 'centroid_y', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'score'])
                writer.writeheader()
                for i, det in enumerate(detections):
                    writer.writerow({
                        'id': i + 1,
                        'area': det.get('area', 0),
                        'centroid_x': det.get('centroid', [0, 0])[0],
                        'centroid_y': det.get('centroid', [0, 0])[1],
                        'bbox_x1': det.get('bbox', [0, 0, 0, 0])[0],
                        'bbox_y1': det.get('bbox', [0, 0, 0, 0])[1],
                        'bbox_x2': det.get('bbox', [0, 0, 0, 0])[2],
                        'bbox_y2': det.get('bbox', [0, 0, 0, 0])[3],
                        'score': scores[i] if i < len(scores) else 0.0,
                    })
        except Exception as e:
            app.logger.warning(f"Failed to save CSV: {e}")
        
        # 4. Save run.json with parameters
        run_json_path = output_dir / f"{channel_name}_run_{timestamp}.json"
        try:
            run_data = {
                'timestamp': timestamp,
                'sample': sample_name,
                'position': position_name,
                'channel': channel_name,
                'num_detections': len(detections),
                'settings': settings,
            }
            with open(run_json_path, 'w') as f:
                json.dump(run_data, f, indent=2)
        except Exception as e:
            app.logger.warning(f"Failed to save run.json: {e}")
        
        # Collect paths
        paths = []
        if overlay_path.exists():
            paths.append(str(overlay_path.relative_to(OUTPUT_ROOT)))
        if masks_path.exists():
            paths.append(str(masks_path.relative_to(OUTPUT_ROOT)))
        if csv_path.exists():
            paths.append(str(csv_path.relative_to(OUTPUT_ROOT)))
        if run_json_path.exists():
            paths.append(str(run_json_path.relative_to(OUTPUT_ROOT)))
        
        return jsonify({
            'success': True,
            'data': {
                'paths': paths
            }
        })
        
    except Exception as e:
        app.logger.error(f"Exosome export failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME UPDATES
# ============================================================================

@sock.route('/ws/pipeline')
def pipeline_websocket(ws):
    """
    WebSocket endpoint for real-time pipeline updates.
    Sends progress and log messages as they occur.
    """
    # Register this client
    with ws_lock:
        ws_clients.add(ws)
    
    try:
        # Send initial status
        runner = get_runner()
        ws.send(json.dumps({
            'type': 'status',
            'data': runner.get_status()
        }))
        
        # Keep connection alive - flask-sock handles this automatically
        # Just wait for messages from client (or disconnection)
        while True:
            try:
                message = ws.receive()
                
                if message:
                    # Handle client commands if needed
                    try:
                        data = json.loads(message)
                        cmd = data.get('command')
                        
                        if cmd == 'get_status':
                            ws.send(json.dumps({
                                'type': 'status',
                                'data': runner.get_status()
                            }))
                    except Exception as e:
                        print(f"Error handling client message: {e}")
            except Exception as e:
                # Connection closed or error - break out of loop
                if "Connection closed" in str(e) or "closed" in str(e).lower():
                    break
                # Re-raise other exceptions
                raise
    
    except Exception as e:
        # Connection closed or error occurred - this is expected
        if "Connection closed" not in str(e) and "closed" not in str(e).lower():
            print(f"WebSocket error: {e}")
    
    finally:
        # Unregister client
        with ws_lock:
            ws_clients.discard(ws)


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # CLI arguments — used by Electron to inject port and data root       #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description='SEA API Server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory for all data (input, output, previews, processing). '
                             'Defaults to the "data" folder next to this script.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml override')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve and apply data root — override module-level path globals    #
    # ------------------------------------------------------------------ #
    if args.data_root:
        _data_root = Path(args.data_root)
    else:
        _data_root = Path(__file__).parent / 'data'

    # Reassign module-level path globals so all endpoint handlers pick them up
    INPUT_ROOT = _data_root / 'input'
    OUTPUT_ROOT = _data_root / 'output'
    DATA_ROOT   = _data_root / 'input'
    PREVIEW_CACHE      = _data_root / 'previews'
    PROCESSING_OUTPUT  = _data_root / 'processing'

    # Ensure directories exist
    for _d in (INPUT_ROOT, OUTPUT_ROOT, PREVIEW_CACHE, PROCESSING_OUTPUT):
        _d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Config path override                                                 #
    # ------------------------------------------------------------------ #
    if args.config:
        os.environ.setdefault('SEA_CONFIG', args.config)

    # ------------------------------------------------------------------ #
    # CUDA availability check                                              #
    # Detect whether a CUDA-capable GPU is present and warn early if not. #
    # Sets SEA_DEVICE_OVERRIDE=cpu so agents fall back gracefully instead  #
    # of crashing mid-pipeline when they first touch a tensor.            #
    # ------------------------------------------------------------------ #
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name} (CUDA {torch.version.cuda})")
        else:
            print("WARNING: No CUDA-capable GPU detected.")
            print("         Pipeline will run on CPU — processing will be slower.")
            print("         To suppress this warning, set  device: cpu  in config.yaml")
            os.environ['SEA_DEVICE_OVERRIDE'] = 'cpu'
    except ImportError:
        print("WARNING: PyTorch not importable — GPU check skipped.")

    # ------------------------------------------------------------------ #
    # Startup banner                                                       #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Starting SEA API Server with Pipeline Control")
    print("=" * 60)
    print(f"Port:         {args.port}")
    print(f"Data root:    {_data_root.absolute()}")
    print(f"Input dir:    {INPUT_ROOT.absolute()}")
    print(f"Output dir:   {OUTPUT_ROOT.absolute()}")
    print("=" * 60)

    # Load marker mapping on startup
    print("\nLoading marker mapping on startup...")
    clear_marker_cache()
    mapping = load_marker_mapping()
    print(f"Marker mapping loaded: {len(mapping)} cycles")

    # ------------------------------------------------------------------ #
    # Start server — use make_server so we can signal readiness AFTER     #
    # the socket is bound (Electron reads "BACKEND_READY" from stdout)    #
    # ------------------------------------------------------------------ #
    from werkzeug.serving import make_server

    server = make_server('0.0.0.0', args.port, app)
    print(f"\nServer listening on http://0.0.0.0:{args.port}")
    print("BACKEND_READY", flush=True)   # ← Electron watches for this line
    server.serve_forever()

