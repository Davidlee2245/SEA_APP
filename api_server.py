"""
Flask API Server for SEA Alignment Visualization
Serves alignment results and images to the React frontend
"""

from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS
from pathlib import Path
import json
import os
import shutil
import tempfile
import traceback
import numpy as np
from typing import Dict, Any, List, Optional
import tifffile
from io import BytesIO
from PIL import Image
import csv
import time
from collections import defaultdict

try:
    import openpyxl  # noqa: F401
    _openpyxl_available = True
except ImportError:
    _openpyxl_available = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React development server

# Configuration
OUTPUT_ROOT = Path("data/output")
INPUT_ROOT = Path("data/input")


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


def calculate_transform_movement(transform_data: Dict[str, Any]) -> tuple:
    """
    Calculate dx, dy, and magnitude from transformation matrix.
    
    Args:
        transform_data: Transformation data from alignment
        
    Returns:
        Tuple of (dx, dy, magnitude)
    """
    transform_type = transform_data.get('type', 'identity')
    
    if transform_type == 'identity':
        return 0.0, 0.0, 0.0
    
    # Extract translation from affine transformation matrix
    # Affine matrix format: [[a, b, tx], [c, d, ty], [0, 0, 1]]
    transform = transform_data.get('transform', [])
    
    if not transform or len(transform) < 2:
        return 0.0, 0.0, 0.0
    
    try:
        # Extract translation components
        tx = float(transform[0][2]) if len(transform[0]) > 2 else 0.0
        ty = float(transform[1][2]) if len(transform[1]) > 2 else 0.0
        
        magnitude = np.sqrt(tx**2 + ty**2)
        
        return tx, ty, magnitude
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0, 0.0


def load_alignment_results(sample_name: str) -> Optional[Dict[str, Any]]:
    """
    Load alignment results from pipeline output.
    
    Args:
        sample_name: Name of the sample
        
    Returns:
        Alignment result dictionary or None if not found
    """
    sample_dir = OUTPUT_ROOT / sample_name
    
    if not sample_dir.exists():
        return None
    
    # Look for alignment metadata (you may need to save this from the pipeline)
    # For now, we'll construct it from the directory structure
    
    preprocessed_dir = sample_dir / "preprocessed"
    registered_dir = sample_dir / "registered"
    labels_dir = sample_dir / "labels"
    
    if not registered_dir.exists():
        return None
    
    # Get list of channels
    registered_files = list(registered_dir.glob("*.tif"))
    
    frames = []
    for reg_file in registered_files:
        channel_name = reg_file.stem.replace('_registered', '')
        
        # Find corresponding preprocessed file
        preproc_file = preprocessed_dir / f"{channel_name}_preprocessed.tif"
        
        if not preproc_file.exists():
            continue
        
        # Try to load transformation data if available
        # In a real scenario, you'd save this during pipeline execution
        # For now, we'll use placeholder values
        dx, dy = 0.0, 0.0  # You can enhance this by saving transform data
        magnitude = 0.0
        
        frames.append({
            'frameId': channel_name,
            'channelName': channel_name,
            'beforeImageUrl': f'/api/images/{sample_name}/preprocessed/{channel_name}_preprocessed.tif',
            'afterImageUrl': f'/api/images/{sample_name}/registered/{channel_name}_registered.tif',
            'movement': {
                'dx': dx,
                'dy': dy,
                'magnitude': magnitude,
                'transformType': 'affine',  # Default
                'residualError': 0.0,
                'numMatches': 0
            }
        })
    
    # Get result files
    final_results = []
    
    # Overlay
    overlay_file = sample_dir / f"{sample_name}_overlay.png"
    if overlay_file.exists():
        final_results.append({
            'type': 'overlay',
            'name': overlay_file.name,
            'url': f'/api/results/{sample_name}/{overlay_file.name}',
            'description': 'RGB overlay of all registered channels'
        })
    
    # CSV
    csv_file = sample_dir / f"{sample_name}_detections.csv"
    if csv_file.exists():
        final_results.append({
            'type': 'csv',
            'name': csv_file.name,
            'url': f'/api/results/{sample_name}/{csv_file.name}',
            'description': 'Quantification data with coordinates and intensities'
        })
    
    # Registered images
    for reg_file in registered_files:
        final_results.append({
            'type': 'registered',
            'name': reg_file.name,
            'url': f'/api/results/{sample_name}/registered/{reg_file.name}',
            'description': f'Registered image: {reg_file.stem}'
        })
    
    # Label images
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.tif"):
            final_results.append({
                'type': 'label',
                'name': label_file.name,
                'url': f'/api/results/{sample_name}/labels/{label_file.name}',
                'description': f'Detection labels: {label_file.stem}'
            })
    
    # Parse colocalization statistics
    coloc_stats = parse_colocalization_stats(sample_dir)
    
    return {
        'sampleName': sample_name,
        'frames': frames,
        'finalResults': final_results,
        'colocalization': coloc_stats,
        'metadata': {
            'anchorChannel': frames[0]['channelName'] if frames else 'unknown',
            'totalChannels': len(frames)
        }
    }


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
    """Serve TIFF images as PNG for browser display."""
    file_path = OUTPUT_ROOT / sample_name / subfolder / filename
    
    if not file_path.exists():
        abort(404)
    
    try:
        # Load TIFF and convert to PNG for web display
        img_array = tifffile.imread(str(file_path))
        
        # Normalize to 8-bit
        img_normalized = ((img_array - img_array.min()) / 
                         (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_normalized)
        
        # Save to BytesIO buffer as PNG
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
    
    except Exception as e:
        app.logger.error(f"Error loading image {file_path}: {e}")
        abort(500)


@app.route('/api/results/<sample_name>/<path:filepath>', methods=['GET'])
def get_result_file(sample_name: str, filepath: str):
    """Serve result files (overlays, CSVs, etc.)."""
    file_path = OUTPUT_ROOT / sample_name / filepath
    
    if not file_path.exists():
        abort(404)
    
    # Determine mimetype
    if file_path.suffix == '.png':
        mimetype = 'image/png'
    elif file_path.suffix == '.csv':
        mimetype = 'text/csv'
    elif file_path.suffix == '.tif' or file_path.suffix == '.tiff':
        # Convert TIFF to PNG for browser display
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
    """List all available samples."""
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'SEA API Server is running'
    })


@app.route('/api/cygnus/health', methods=['GET'])
def cygnus_health():
    return jsonify({"success": True, "message": "Cygnus API is ready."})


@app.route('/api/cygnus/run', methods=['POST'])
def run_cygnus_pipeline():
    from cygnus_upload_merge import merge_cygnus_upload_files

    uploads = [f for f in request.files.getlist("files") if getattr(f, "filename", None)]
    if not uploads:
        f0 = request.files.get("file")
        if f0 and f0.filename:
            uploads = [f0]

    if not uploads:
        return jsonify({"success": False, "message": "No file provided."}), 400

    tmp_path = None
    output_dir = None
    try:
        t_pipe = time.perf_counter()
        from run_pipeline import cygnus_pipeline_log, run_full_pipeline

        cygnus_pipeline_log("merge_files_start", t_pipe, detail=f"upload_count={len(uploads)}")
        merged = merge_cygnus_upload_files(uploads, _openpyxl_available)
        cygnus_pipeline_log(
            "merge_files_done",
            t_pipe,
            detail=f"rows={len(merged)} cols={merged.shape[1]}",
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
            tmp_path = tmp.name
        merged.to_csv(tmp_path, index=False)
        cygnus_pipeline_log("temp_csv_written", t_pipe, detail=f"path={tmp_path}")

        output_dir = tempfile.mkdtemp()
        cygnus_pipeline_log("temp_output_dir", t_pipe, detail=f"path={output_dir}")

        run_full_pipeline(filepath=tmp_path, output_dir=output_dir, pipeline_t0=t_pipe)

        report_path = os.path.join(output_dir, "cygnus_report.html")
        if not os.path.isfile(report_path):
            return jsonify({"success": False, "message": "Report was not generated."}), 500

        with open(report_path, encoding="utf-8") as rfile:
            report_html = rfile.read()

        cygnus_pipeline_log(
            "response_ready",
            t_pipe,
            detail=f"report_html_chars={len(report_html)}",
        )
        return jsonify({"success": True, "report_html": report_html})

    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"{str(e)}\n{traceback.format_exc()}",
        }), 500
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if output_dir:
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except OSError:
                pass


if __name__ == '__main__':
    print("Starting SEA API Server...")
    print(f"Output directory: {OUTPUT_ROOT.absolute()}")
    print(f"Server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET /api/health - Health check")
    print("  GET /api/cygnus/health - Cygnus pipeline API health")
    print("  POST /api/cygnus/run - Run Cygnus pipeline (multipart CSV)")
    print("  GET /api/samples - List all samples")
    print("  GET /api/alignment/<sample_name> - Get alignment data")
    print("  GET /api/images/<sample_name>/<subfolder>/<filename> - Get image")
    print("  GET /api/results/<sample_name>/<filepath> - Get result file")
    
    app.run(debug=True, port=5000)

