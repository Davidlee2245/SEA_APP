"""
Agent Chat API - OpenAI-powered imaging preprocessing assistant
Provides intelligent recommendations for image preprocessing parameters
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
from openai import OpenAI
import json
from typing import List, Dict, Any, Tuple, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path
import hashlib

# Try to import tifffile (optional, better for multi-page and 16-bit TIFF)
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("⚠️  tifffile not available. Using PIL for TIFF conversion (may have limitations with 16-bit and multi-page TIFF).")

# Import manifest management
from core.manifest import ManifestManager, generate_image_id, generate_run_id, ValidationError

app = Flask(__name__)
CORS(app)

# Global manifest manager (will be initialized per run_id)
_manifest_managers = {}  # run_id -> ManifestManager

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Preview cache directory
PREVIEW_CACHE_DIR = Path("previews")
PREVIEW_CACHE_DIR.mkdir(exist_ok=True)

# Max upload size: 50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

def tiff_to_png_preview(file_bytes: bytes, filename: str = "") -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert TIFF to PNG for browser preview.
    Handles multi-page TIFF (uses page 0), 16-bit normalization (percentile stretch).
    
    Args:
        file_bytes: Raw image file bytes
        filename: Original filename (for metadata)
        
    Returns:
        Tuple of (png_bytes, metadata_dict)
    """
    metadata = {
        'original_filename': filename,
        'is_tiff': False,
        'converted': False,
        'bit_depth': None,
        'pages': 1,
        'width': None,
        'height': None,
        'mime_type': 'image/png'
    }
    
    try:
        img = None
        used_tifffile = False
        
        # Try to open with tifffile first (better for multi-page and 16-bit)
        if HAS_TIFFFILE:
            try:
                with tifffile.TiffFile(BytesIO(file_bytes)) as tif:
                    # Get metadata
                    metadata['is_tiff'] = True
                    metadata['pages'] = len(tif.pages) if hasattr(tif, 'pages') else 1
                    
                    # Read first page
                    img_array = tif.pages[0].asarray()
                    
                    # Get bit depth from dtype
                    if img_array.dtype == np.uint16:
                        metadata['bit_depth'] = 16
                    elif img_array.dtype == np.uint8:
                        metadata['bit_depth'] = 8
                    elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
                        metadata['bit_depth'] = 'float'
                    else:
                        metadata['bit_depth'] = str(img_array.dtype)
                    
                    metadata['width'] = img_array.shape[1] if len(img_array.shape) > 1 else img_array.shape[0]
                    metadata['height'] = img_array.shape[0] if len(img_array.shape) > 1 else 1
                    
                    # Handle grayscale vs RGB
                    if len(img_array.shape) == 2:
                        # Grayscale
                        pass
                    elif len(img_array.shape) == 3:
                        # Multi-channel (RGB, etc.)
                        if img_array.shape[2] > 3:
                            # Take first 3 channels
                            img_array = img_array[:, :, :3]
                    else:
                        raise ValueError(f"Unsupported image shape: {img_array.shape}")
                    
                    # Normalize 16-bit or float to 8-bit using percentile stretch (1-99%)
                    if img_array.dtype in (np.uint16, np.float32, np.float64):
                        # Percentile normalization (1st-99th percentile)
                        p1 = np.percentile(img_array, 1)
                        p99 = np.percentile(img_array, 99)
                        
                        if p99 > p1:
                            img_array = np.clip((img_array - p1) / (p99 - p1) * 255, 0, 255)
                        else:
                            img_array = np.zeros_like(img_array, dtype=np.uint8)
                        
                        img_array = img_array.astype(np.uint8)
                        metadata['converted'] = True
                    
                    # Convert to PIL Image
                    if len(img_array.shape) == 2:
                        img = Image.fromarray(img_array, mode='L')
                        img = img.convert('RGB')  # Convert grayscale to RGB for display
                    else:
                        img = Image.fromarray(img_array, mode='RGB')
                    
                    used_tifffile = True
                    
            except Exception as tiff_error:
                # Fallback to PIL if tifffile fails
                pass
        
        # Use PIL if tifffile not available or failed
        if img is None:
            img = Image.open(BytesIO(file_bytes))
            
            # Check if it's actually a TIFF
            if img.format == 'TIFF':
                metadata['is_tiff'] = True
                if hasattr(img, 'n_frames'):
                    metadata['pages'] = img.n_frames
                img.seek(0)  # Use first page
            elif img.format in ('PNG', 'JPEG', 'JPG'):
                # Not a TIFF, return as-is if already displayable
                buffer = BytesIO()
                img.save(buffer, format=img.format)
                buffer.seek(0)
                metadata['width'] = img.width
                metadata['height'] = img.height
                return buffer.read(), metadata
            else:
                raise ValueError(f"Unsupported image format: {img.format}")
            
            # Handle 16-bit or float images
            if img.mode in ('I', 'I;16', 'F'):
                img_array = np.array(img)
                metadata['bit_depth'] = 16 if img.mode == 'I;16' else 'float'
                
                # Percentile normalization
                p1 = np.percentile(img_array, 1)
                p99 = np.percentile(img_array, 99)
                
                if p99 > p1:
                    img_array = np.clip((img_array - p1) / (p99 - p1) * 255, 0, 255)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                
                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array)
                metadata['converted'] = True
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'RGBA'):
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            if not used_tifffile:
                metadata['width'] = img.width
                metadata['height'] = img.height
        
        # Convert to PNG
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        png_bytes = buffer.read()
        
        metadata['converted'] = True if metadata['is_tiff'] else False
        
        return png_bytes, metadata
        
    except Exception as e:
        raise ValueError(f"Failed to convert TIFF to PNG: {str(e)}")


def get_preview_cache_path(file_bytes: bytes) -> Path:
    """
    Generate cache path for preview based on file hash.
    
    Args:
        file_bytes: Raw file bytes
        
    Returns:
        Path to cached preview file
    """
    # Create hash of file bytes
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    return PREVIEW_CACHE_DIR / f"{file_hash}.png"


# System prompt for the imaging preprocessing agent
def convert_image_to_png_base64(base64_image: str) -> str:
    """
    Convert any image format (including TIFF) to PNG base64 for OpenAI API.
    Handles TIFF, multi-page TIFF, and normalizes the image.
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Open image with PIL
        img = Image.open(BytesIO(image_bytes))
        
        # If TIFF with multiple pages, use the first page
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)  # Go to first frame
        
        # Convert to RGB if needed (TIFF can be grayscale, multi-channel, etc.)
        if img.mode not in ('RGB', 'RGBA'):
            # For grayscale or other modes, convert to RGB
            if img.mode == 'I' or img.mode == 'I;16' or img.mode == 'F':
                # 16-bit or float images - normalize to 8-bit
                img_array = np.array(img)
                
                # Normalize to 0-255 range
                img_min = img_array.min()
                img_max = img_array.max()
                if img_max > img_min:
                    img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                
                img = Image.fromarray(img_array)
            
            # Convert to RGB
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            else:
                img = img.convert('RGB')
        
        # Resize if too large (OpenAI has limits)
        max_size = 2048
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to PNG and encode as base64
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        png_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return png_base64
    
    except Exception as e:
        raise ValueError(f"Failed to convert image: {str(e)}")


AGENT_SYSTEM_PROMPT = """You are an AI preprocessing assistant for cell and fluorescence microscopy images.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:

1. IMAGE REQUIREMENT:
   - You MUST ONLY respond when an image is explicitly provided in the user's message.
   - You MUST NEVER analyze, diagnose, or recommend parameters without an actual image.
   - You MUST NEVER infer or hallucinate image properties.
   - If no image is provided, respond with EXACTLY: "No image detected. Please upload a microscopy image before requesting preprocessing recommendations."

2. RESPONSE FORMAT:
   - When an image IS provided, you MUST use the exact TXT format below.
   - Do NOT output JSON.
   - Do NOT output code.
   - Do NOT include any additional commentary outside the format.

3. ANALYSIS REQUIREMENTS:
   - Each image may require different preprocessing strength depending on blur, contrast, background level, noise, and cell density.
   - You must output concrete numeric parameters, not vague descriptions.
   - The output must be easy for humans to read and suitable for saving as a .txt file.

When an image IS provided, analyze it and respond using the exact TXT format below.

First, briefly diagnose the image quality.
Then, propose exact numeric preprocessing parameters.

Use the following output format exactly (ONLY when image is present):

[IMAGE_DIAGNOSIS]
Contrast: low / medium / high
Sharpness: blurred / moderate / sharp
Background level: low / medium / high
Noise level: low / medium / high
Cell density: sparse / medium / dense

[RECOMMENDED_PREPROCESSING_PARAMETERS]

- Bit depth conversion:
  16-bit → 8-bit (yes / no)

- Denoising:
  Median filter kernel size: <integer>
  Gaussian blur sigma: <float>

- Background subtraction:
  Method: Gaussian blur
  Sigma: <float>

- Intensity normalization:
  Percentile low: <float>
  Percentile high: <float>

- Local contrast enhancement (CLAHE):
  Clip limit: <float>
  Tile grid size: <integer x integer>

- Sharpening (optional):
  Method: unsharp mask
  Blur sigma: <float>
  Strength (alpha): <float>
  Subtraction weight (beta): <float>

[CONFIDENCE]
Overall confidence score (0.0–1.0): <float>

Do not include any additional commentary outside this format."""


def get_manifest_manager(run_id: str) -> ManifestManager:
    """
    Get or create a manifest manager for a run_id.
    Caches managers to avoid repeated file I/O.
    """
    if run_id not in _manifest_managers:
        _manifest_managers[run_id] = ManifestManager(run_id)
    return _manifest_managers[run_id]


def parse_txt_format_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the TXT format response and extract parameters.
    
    Returns:
    {
        "diagnosis": {...},
        "parameters": {...},
        "confidence": float
    }
    """
    import re
    
    result = {
        "diagnosis": {},
        "parameters": {},
        "confidence": 0.8  # Default
    }
    
    # Extract diagnosis section
    diagnosis_match = re.search(r'\[IMAGE_DIAGNOSIS\](.*?)(?=\[RECOMMENDED_PREPROCESSING_PARAMETERS\]|\[CONFIDENCE\]|$)', response_text, re.DOTALL | re.IGNORECASE)
    if diagnosis_match:
        diagnosis_text = diagnosis_match.group(1)
        for line in diagnosis_text.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip().lower()
                result["diagnosis"][key] = value
    
    # Extract parameters section
    params_match = re.search(r'\[RECOMMENDED_PREPROCESSING_PARAMETERS\](.*?)(?=\[CONFIDENCE\]|$)', response_text, re.DOTALL | re.IGNORECASE)
    if params_match:
        params_text = params_match.group(1)
        
        # Bit depth conversion
        bit_match = re.search(r'16-bit → 8-bit\s*\(yes\s*/\s*no\)\s*:\s*(yes|no)', params_text, re.IGNORECASE)
        if bit_match:
            result["parameters"]["bit_depth_conversion"] = bit_match.group(1).lower() == "yes"
        
        # Denoising
        median_match = re.search(r'Median filter kernel size:\s*(\d+)', params_text, re.IGNORECASE)
        if median_match:
            result["parameters"]["median_filter_kernel_size"] = int(median_match.group(1))
        
        gaussian_match = re.search(r'Gaussian blur sigma:\s*([\d.]+)', params_text, re.IGNORECASE)
        if gaussian_match:
            result["parameters"]["gaussian_blur_sigma"] = float(gaussian_match.group(1))
        
        # Background subtraction
        bg_sigma_match = re.search(r'Background subtraction.*?Sigma:\s*([\d.]+)', params_text, re.DOTALL | re.IGNORECASE)
        if bg_sigma_match:
            result["parameters"]["background_subtraction_sigma"] = float(bg_sigma_match.group(1))
        
        # Intensity normalization
        perc_low_match = re.search(r'Percentile low:\s*([\d.]+)', params_text, re.IGNORECASE)
        if perc_low_match:
            result["parameters"]["percentile_low"] = float(perc_low_match.group(1))
        
        perc_high_match = re.search(r'Percentile high:\s*([\d.]+)', params_text, re.IGNORECASE)
        if perc_high_match:
            result["parameters"]["percentile_high"] = float(perc_high_match.group(1))
        
        # CLAHE
        clahe_clip_match = re.search(r'Clip limit:\s*([\d.]+)', params_text, re.IGNORECASE)
        if clahe_clip_match:
            result["parameters"]["clahe_clip_limit"] = float(clahe_clip_match.group(1))
        
        clahe_grid_match = re.search(r'Tile grid size:\s*(\d+)\s*x\s*(\d+)', params_text, re.IGNORECASE)
        if clahe_grid_match:
            grid_size = int(clahe_grid_match.group(1))
            result["parameters"]["clahe_grid_size"] = grid_size
        
        # Sharpening
        sharp_sigma_match = re.search(r'Blur sigma:\s*([\d.]+)', params_text, re.IGNORECASE)
        if sharp_sigma_match:
            result["parameters"]["sharpening_blur_sigma"] = float(sharp_sigma_match.group(1))
        
        sharp_alpha_match = re.search(r'Strength\s*\(alpha\):\s*([\d.]+)', params_text, re.IGNORECASE)
        if sharp_alpha_match:
            result["parameters"]["sharpening_alpha"] = float(sharp_alpha_match.group(1))
        
        sharp_beta_match = re.search(r'Subtraction weight\s*\(beta\):\s*([\d.]+)', params_text, re.IGNORECASE)
        if sharp_beta_match:
            result["parameters"]["sharpening_beta"] = float(sharp_beta_match.group(1))
    
    # Extract confidence
    confidence_match = re.search(r'\[CONFIDENCE\].*?Overall confidence score.*?([\d.]+)', response_text, re.DOTALL | re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
    
    return result


def extract_params_from_agent_output(agent_output: Any) -> Dict[str, Any]:
    """
    Extract preprocessing parameters from agent output.
    Handles both old JSON format and new TXT format.
    
    Returns flat params dict suitable for manifest.
    """
    if isinstance(agent_output, dict):
        # Old JSON format
        if "recommended_parameters" in agent_output:
            recommended = agent_output.get('recommended_parameters', {})
            flat_params = {}
            for param_name, param_info in recommended.items():
                if isinstance(param_info, dict) and 'value' in param_info:
                    flat_params[param_name] = param_info['value']
                else:
                    flat_params[param_name] = param_info
            return flat_params
        # New TXT format (already parsed)
        elif "parameters" in agent_output:
            return agent_output["parameters"]
    
    return {}


@app.route('/api/agent/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_key_set = bool(os.getenv('OPENAI_API_KEY'))
    return jsonify({
        'status': 'ok',
        'openai_configured': api_key_set,
        'message': 'Agent Chat API is running' if api_key_set else 'OpenAI API key not configured'
    })


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """
    Agent chat endpoint - processes user messages with OpenAI
    
    Body:
    {
        "message": "user message text",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "image": "base64_encoded_image" (optional),
        "image_path": "path/to/image.tif" (optional, for saving to manifest),
        "run_id": "unique_run_id" (optional, defaults to 'agent_session')
    }
    """
    # Check if API key is configured
    if not os.getenv('OPENAI_API_KEY'):
        return jsonify({
            'success': False,
            'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'
        }), 500
    
    try:
        data = request.json or {}
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        image_data = data.get('image')  # Base64 encoded image
        image_path = data.get('image_path')  # Optional: original image path for manifest
        run_id = data.get('run_id', 'agent_session')  # Optional: run ID for manifest
        
        # CRITICAL: Enforce image requirement for preprocessing recommendations
        # If the message requests preprocessing analysis, image MUST be present
        is_preprocessing_request = (
            'preprocessing' in user_message.lower() or
            'recommend' in user_message.lower() or
            'analyze' in user_message.lower() or
            'diagnose' in user_message.lower()
        )
        
        if is_preprocessing_request and not image_data:
            # Fail-safe: Return error message without calling AI
            return jsonify({
                'success': True,
                'message': 'No image detected. Please upload a microscopy image before requesting preprocessing recommendations.',
                'parameters': None,
                'model': 'fail-safe',
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                },
                'manifest': {
                    'saved': False,
                    'run_id': None,
                    'image_id': None
                }
            })
        
        if not user_message and not image_data:
            return jsonify({
                'success': False,
                'error': 'Message or image is required'
            }), 400
        
        # Build messages array with system prompt and history
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        
        # Add conversation history (limit to last 10 messages to avoid token limits)
        messages.extend(history[-10:])
        
        # Build user message content
        if image_data:
            # Convert image to PNG format (handles TIFF, 16-bit, etc.)
            try:
                png_base64 = convert_image_to_png_base64(image_data)
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f'Image conversion failed: {str(e)}'
                }), 400
            
            # Use vision model with image
            # CRITICAL: Image is present, so we can proceed with analysis
            # Use user message if provided, otherwise use default prompt
            prompt_text = user_message or "Analyze the uploaded microscopy image and recommend image-specific preprocessing parameters to enhance cell shape visibility and separability for accurate cell counting. Diagnose the image quality first, then provide exact numeric parameters. Output in structured TXT format only."
            user_content = [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{png_base64}",
                        "detail": "high"
                    }
                }
            ]
            model = "gpt-4o"  # Vision-capable model
        else:
            # Text-only
            user_content = user_message
            model = "gpt-4o-mini"  # Fast and cost-effective
        
        messages.append({"role": "user", "content": user_content})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500 if image_data else 1000
        )
        
        # Extract response
        assistant_message = response.choices[0].message.content
        
        # Try to parse parameters from response (TXT format or JSON format)
        parameters = None
        parsed_data = None
        saved_to_manifest = False
        image_id = None
        
        try:
            # First try TXT format (new format)
            if '[RECOMMENDED_PREPROCESSING_PARAMETERS]' in assistant_message:
                parsed_data = parse_txt_format_response(assistant_message)
                parameters = {
                    "preprocessing_required": True,
                    "diagnosis": parsed_data.get("diagnosis", {}),
                    "recommended_parameters": parsed_data.get("parameters", {}),
                    "confidence": parsed_data.get("confidence", 0.8)
                }
            # Fallback to JSON format (old format)
            elif '```json' in assistant_message:
                json_start = assistant_message.find('```json') + 7
                json_end = assistant_message.find('```', json_start)
                if json_end > json_start:
                    json_str = assistant_message[json_start:json_end].strip()
                    parameters = json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse parameters: {e}")
        
        # Auto-save parameters to manifest if we have them and an image_path
        if parameters and image_path:
            try:
                # Get manifest manager
                manifest = get_manifest_manager(run_id)
                
                # Generate image ID
                image_id = generate_image_id(image_path)
                
                # Extract flat params
                flat_params = extract_params_from_agent_output(parameters)
                
                if flat_params:
                    # Build agent output for validation
                    # Include diagnosis info in reasoning if available
                    reasoning_parts = []
                    if "diagnosis" in parameters:
                        diagnosis = parameters["diagnosis"]
                        reasoning_parts.append(f"Diagnosis: {diagnosis}")
                    if not reasoning_parts:
                        reasoning_parts.append("Image analysis completed")
                    
                    agent_output = {
                        "recommended_params": flat_params,
                        "reasoning": " | ".join(reasoning_parts),
                        "confidence": parameters.get("confidence", parsed_data.get("confidence", 0.8) if parsed_data else 0.8),
                        "flags": parameters.get("warnings", []),
                    }
                    
                    # Save to manifest
                    manifest.save_agent_recommendation(
                        image_id=image_id,
                        image_path=image_path,
                        agent_output=agent_output,
                        agent_model=response.model,
                        prompt_version="1.0",
                    )
                    
                    saved_to_manifest = True
                    print(f"✅ Saved parameters to manifest: {run_id}/{image_id}")
                
            except ValidationError as e:
                print(f"⚠️  Validation error saving to manifest: {e}")
                # Don't fail the request, just log
            except Exception as e:
                print(f"⚠️  Error saving to manifest: {e}")
                # Don't fail the request, just log
        
        return jsonify({
            'success': True,
            'message': assistant_message,
            'parameters': parameters,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            'manifest': {
                'saved': saved_to_manifest,
                'run_id': run_id if saved_to_manifest else None,
                'image_id': image_id if saved_to_manifest else None,
            }
        })
    
    except Exception as e:
        print(f"Agent chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/agent/quick-recommend', methods=['POST'])
def quick_recommend():
    """
    Quick preprocessing recommendation endpoint
    Provides fast parameter suggestions based on image characteristics
    
    Body:
    {
        "image": "base64_encoded_image" (required),
        "image_path": "path/to/image.tif" (optional, for saving to manifest),
        "run_id": "unique_run_id" (optional)
    }
    
    Note: This endpoint now requires an image and uses the same TXT format as regular chat.
    """
    try:
        data = request.json or {}
        image_data = data.get('image')
        image_path = data.get('image_path')
        run_id = data.get('run_id', 'agent_session')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'Image is required for quick recommendation'
            }), 400
        
        # Convert image to PNG format
        try:
            png_base64 = convert_image_to_png_base64(image_data)
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Image conversion failed: {str(e)}'
            }), 400
        
        # Use the fixed prompt as specified
        user_content = [
            {
                "type": "text",
                "text": "Analyze the uploaded microscopy image and recommend image-specific preprocessing parameters to enhance cell shape visibility and separability for accurate cell counting. Diagnose the image quality first, then provide exact numeric parameters. Output in structured TXT format only."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{png_base64}",
                    "detail": "high"
                }
            }
        ]
        
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Vision-capable model
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Extract response
        assistant_message = response.choices[0].message.content
        
        # Parse parameters from TXT format
        parameters = None
        parsed_data = None
        saved_to_manifest = False
        image_id = None
        
        try:
            if '[RECOMMENDED_PREPROCESSING_PARAMETERS]' in assistant_message:
                parsed_data = parse_txt_format_response(assistant_message)
                parameters = {
                    "preprocessing_required": True,
                    "diagnosis": parsed_data.get("diagnosis", {}),
                    "recommended_parameters": parsed_data.get("parameters", {}),
                    "confidence": parsed_data.get("confidence", 0.8)
                }
        except Exception as e:
            print(f"Failed to parse parameters: {e}")
        
        # Auto-save parameters to manifest if we have them and an image_path
        if parameters and image_path:
            try:
                manifest = get_manifest_manager(run_id)
                image_id = generate_image_id(image_path)
                flat_params = extract_params_from_agent_output(parameters)
                
                if flat_params:
                    reasoning_parts = []
                    if "diagnosis" in parameters:
                        diagnosis = parameters["diagnosis"]
                        reasoning_parts.append(f"Diagnosis: {diagnosis}")
                    if not reasoning_parts:
                        reasoning_parts.append("Image analysis completed")
                    
                    agent_output = {
                        "recommended_params": flat_params,
                        "reasoning": " | ".join(reasoning_parts),
                        "confidence": parameters.get("confidence", parsed_data.get("confidence", 0.8) if parsed_data else 0.8),
                        "flags": [],
                    }
                    
                    manifest.save_agent_recommendation(
                        image_id=image_id,
                        image_path=image_path,
                        agent_output=agent_output,
                        agent_model=response.model,
                        prompt_version="2.0",  # Updated version for TXT format
                    )
                    
                    saved_to_manifest = True
                    print(f"✅ Saved parameters to manifest: {run_id}/{image_id}")
                
            except ValidationError as e:
                print(f"⚠️  Validation error saving to manifest: {e}")
            except Exception as e:
                print(f"⚠️  Error saving to manifest: {e}")
        
        return jsonify({
            'success': True,
            'message': assistant_message,
            'parameters': parameters,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            'manifest': {
                'saved': saved_to_manifest,
                'run_id': run_id if saved_to_manifest else None,
                'image_id': image_id if saved_to_manifest else None,
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/agent/manifest/<run_id>', methods=['GET'])
def get_manifest(run_id):
    """
    Get manifest for a specific run.
    
    Returns:
    {
        "success": true,
        "manifest": {...}
    }
    """
    try:
        manifest = get_manifest_manager(run_id)
        return jsonify({
            'success': True,
            'manifest': manifest.manifest
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404


@app.route('/api/agent/manifest/<run_id>/image/<image_id>', methods=['GET'])
def get_image_record(run_id, image_id):
    """
    Get a specific image record from manifest.
    
    Returns:
    {
        "success": true,
        "record": {...}
    }
    """
    try:
        manifest = get_manifest_manager(run_id)
        record = manifest.get_image_record(image_id)
        
        if not record:
            return jsonify({
                'success': False,
                'error': f'Image {image_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'record': record
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def create_readable_params_txt(run_id: str, image_id: str, record: dict) -> str:
    """
    Create a human-readable TXT file with approved parameters.
    Saves in the SAME directory as the original image.
    Returns the file path.
    """
    from pathlib import Path
    from datetime import datetime
    import os
    
    # Get the original image path
    image_path = record.get('image_path', '')
    
    # Search for the actual image file in data/input directory tree
    output_dir = None
    base_name = None
    
    if image_path:
        # Just the filename (e.g., "r01c01f03p01-ch1sk1fk1fl1.tiff")
        filename = Path(image_path).name
        base_name = Path(image_path).stem
        
        # Search for this file in data/input directory tree
        data_input_dir = Path("data/input")
        if data_input_dir.exists():
            # Search recursively for the image file
            for file in data_input_dir.rglob(filename):
                # Found it! Use its parent directory
                output_dir = file.parent
                print(f"📁 Found image at: {file}")
                break
        
        # If not found in data/input, check if it's an absolute path
        if not output_dir:
            image_file = Path(image_path)
            if image_file.exists():
                output_dir = image_file.parent
            elif image_file.is_absolute():
                output_dir = image_file.parent
    
    # Fallback to data/input if we couldn't find the image
    if not output_dir:
        output_dir = Path("data/input")
        print(f"⚠️  Could not find image, saving to: {output_dir}")
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_name:
        filename = f"APPROVED_PARAMS_{base_name}_{timestamp}.txt"
    else:
        filename = f"APPROVED_PARAMS_{image_id}_{timestamp}.txt"
    
    filepath = output_dir / filename
    
    # Get parameters
    params = record.get('final_params') or record.get('recommended_params') or {}
    reasoning = record.get('reasoning', 'No reasoning provided')
    confidence = record.get('confidence', 0.0)
    image_path = record.get('image_path', 'Unknown')
    
    # Create content
    content = f"""================================================================================
                    APPROVED PREPROCESSING PARAMETERS
================================================================================

✅ STATUS: APPROVED and ready to process
📅 Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🆔 Run ID: {run_id}
🖼️ Image: {image_path}
🔑 Image ID: {image_id}
🎯 Confidence: {confidence:.1%}

================================================================================
📊 YOUR APPROVED PARAMETERS
================================================================================

"""
    
    # Add each parameter
    param_descriptions = {
        'denoising_strength': {
            'name': 'Denoising Strength',
            'purpose': 'Remove noise while preserving edges',
            'range': '0.0 - 1.0'
        },
        'background_subtraction_radius': {
            'name': 'Background Subtraction Radius',
            'purpose': 'Remove uneven background illumination',
            'range': '1 - 500 pixels'
        },
        'clahe_clip_limit': {
            'name': 'CLAHE Clip Limit',
            'purpose': 'Enhance local contrast',
            'range': '0.1 - 10.0'
        },
        'clahe_grid_size': {
            'name': 'CLAHE Grid Size',
            'purpose': 'Size of local regions for CLAHE',
            'range': '2 - 32'
        },
        'gamma_correction': {
            'name': 'Gamma Correction',
            'purpose': 'Adjust brightness curve',
            'range': '0.1 - 5.0'
        },
        'threshold_method': {
            'name': 'Threshold Method',
            'purpose': 'Convert to binary image',
            'range': 'otsu/adaptive/binary'
        },
        'threshold_value': {
            'name': 'Threshold Value',
            'purpose': 'Manual threshold value',
            'range': '0 - 255'
        },
        'sharpening_strength': {
            'name': 'Sharpening Strength',
            'purpose': 'Enhance edge definition',
            'range': '0.0 - 2.0'
        },
        'contrast_alpha': {
            'name': 'Contrast Alpha',
            'purpose': 'Linear contrast adjustment',
            'range': '0.5 - 3.0'
        },
        'brightness_beta': {
            'name': 'Brightness Beta',
            'purpose': 'Brightness offset',
            'range': '-100 to 100'
        }
    }
    
    for i, (param_name, param_value) in enumerate(params.items(), 1):
        desc = param_descriptions.get(param_name, {
            'name': param_name.replace('_', ' ').title(),
            'purpose': 'Custom parameter',
            'range': 'Varies'
        })
        
        content += f"""Parameter {i}: {desc['name'].upper()}
   Value: {param_value}
   Purpose: {desc['purpose']}
   Range: {desc['range']}

"""
    
    content += f"""
================================================================================
🤖 AI REASONING
================================================================================

{reasoning}

================================================================================
🚀 HOW TO APPLY THESE PARAMETERS
================================================================================

Run this command in your terminal:

   cd /home/david/.cursor-tutor/SEA
   conda activate SEA
   
   python -c "
   from core.manifest import ManifestManager
   from core.preprocess import PreprocessRunner
   
   manifest = ManifestManager('{run_id}')
   runner = PreprocessRunner(manifest)
   summary = runner.run()
   
   print(f'✅ Success: {{summary[\\"success\\"]}} images')
   print(f'❌ Failed: {{summary[\\"failed\\"]}} images')
   "

Output will be saved to:
   outputs/runs/{run_id}/preprocessed/{image_id}.png

================================================================================
📂 FILE LOCATIONS
================================================================================

This file:        {filepath}
Manifest (JSON):  outputs/runs/{run_id}/preprocess_manifest.json
Output directory: outputs/runs/{run_id}/preprocessed/

================================================================================
✅ READY TO PROCESS!
================================================================================

Your parameters are approved and saved.
Run the command above to process your image.

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Status: ✅ APPROVED - Ready to Process
================================================================================
"""
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)


@app.route('/api/agent/manifest/<run_id>/image/<image_id>/approve', methods=['POST'])
def approve_params(run_id, image_id):
    """
    Approve parameters for an image.
    
    Body (optional):
    {
        "final_params": {...}  // If not provided, uses recommended_params
    }
    """
    try:
        data = request.json or {}
        final_params = data.get('final_params')
        
        manifest = get_manifest_manager(run_id)
        manifest.approve_params(image_id, final_params)
        
        # Get the record
        record = manifest.get_image_record(image_id)
        
        # Create readable TXT file automatically
        txt_file_path = create_readable_params_txt(run_id, image_id, record)
        
        print(f"✅ Parameters approved and saved to: {txt_file_path}")
        
        return jsonify({
            'success': True,
            'message': 'Parameters approved',
            'record': record,
            'txt_file': txt_file_path
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/agent/manifest/<run_id>/image/<image_id>/edit', methods=['POST'])
def edit_params(run_id, image_id):
    """
    Edit and approve parameters for an image.
    
    Body:
    {
        "final_params": {...}  // Required
    }
    """
    try:
        data = request.json or {}
        final_params = data.get('final_params')
        
        if not final_params:
            return jsonify({
                'success': False,
                'error': 'final_params is required'
            }), 400
        
        manifest = get_manifest_manager(run_id)
        manifest.approve_params(image_id, final_params)
        
        # Get the record
        record = manifest.get_image_record(image_id)
        
        # Create readable TXT file automatically
        txt_file_path = create_readable_params_txt(run_id, image_id, record)
        
        print(f"✅ Parameters edited and saved to: {txt_file_path}")
        
        return jsonify({
            'success': True,
            'message': 'Parameters edited and approved',
            'record': record,
            'txt_file': txt_file_path
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/agent/runs', methods=['GET'])
def list_runs():
    """
    List all available runs (directories in outputs/runs/).
    
    Returns:
    {
        "success": true,
        "runs": [
            {"run_id": "...", "created_at": "...", "image_count": 5},
            ...
        ]
    }
    """
    try:
        from pathlib import Path
        
        runs_dir = Path("outputs/runs")
        if not runs_dir.exists():
            return jsonify({
                'success': True,
                'runs': []
            })
        
        runs = []
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):  # Most recent first
            if run_dir.is_dir():
                manifest_path = run_dir / "preprocess_manifest.json"
                if manifest_path.exists():
                    try:
                        manifest = get_manifest_manager(run_dir.name)
                        runs.append({
                            'run_id': run_dir.name,
                            'created_at': manifest.manifest.get('created_at'),
                            'updated_at': manifest.manifest.get('updated_at'),
                            'image_count': len(manifest.manifest.get('images', {})),
                        })
                    except:
                        pass
        
        return jsonify({
            'success': True,
            'runs': runs
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/agent/create-run', methods=['POST'])
def create_run():
    """
    Create a new run ID.
    
    Body (optional):
    {
        "run_id": "custom_run_id"  // If not provided, generates a new one
    }
    
    Returns:
    {
        "success": true,
        "run_id": "..."
    }
    """
    try:
        data = request.json or {}
        run_id = data.get('run_id', generate_run_id())
        
        # Initialize manifest
        manifest = get_manifest_manager(run_id)
        
        return jsonify({
            'success': True,
            'run_id': run_id,
            'created_at': manifest.manifest.get('created_at')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/preview', methods=['POST'])
def preview_image():
    """
    Convert uploaded image (especially TIFF) to PNG for browser preview.
    
    Body:
    {
        "image": "base64_encoded_image" (required),
        "filename": "image.tiff" (optional, for metadata)
    }
    
    Returns:
    {
        "success": true,
        "previewUrl": "data:image/png;base64,...",
        "mimeType": "image/png",
        "width": 1024,
        "height": 1024,
        "metadata": {
            "original_filename": "image.tiff",
            "is_tiff": true,
            "converted": true,
            "bit_depth": 16,
            "pages": 1
        }
    }
    OR
    Returns PNG bytes directly with Content-Type: image/png
    """
    try:
        data = request.json or {}
        image_data = data.get('image')  # Base64 encoded
        filename = data.get('filename', '')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'Image data is required'
            }), 400
        
        # Decode base64
        try:
            file_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 data: {str(e)}'
            }), 400
        
        # Check file size
        if len(file_bytes) > MAX_UPLOAD_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024):.0f}MB'
            }), 400
        
        # Check if it's a TIFF
        is_tiff = (
            filename.lower().endswith(('.tif', '.tiff')) or
            file_bytes[:4] == b'II*\x00' or  # TIFF little-endian
            file_bytes[:4] == b'MM\x00*'      # TIFF big-endian
        )
        
        # Check cache first
        cache_path = get_preview_cache_path(file_bytes)
        if cache_path.exists():
            # Return cached preview
            with open(cache_path, 'rb') as f:
                png_bytes = f.read()
            
            # Get metadata from cache (store as JSON alongside)
            metadata_path = cache_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Fallback metadata
                img = Image.open(BytesIO(png_bytes))
                metadata = {
                    'original_filename': filename,
                    'is_tiff': is_tiff,
                    'converted': is_tiff,
                    'bit_depth': None,
                    'pages': 1,
                    'width': img.width,
                    'height': img.height
                }
        else:
            # Convert image
            if is_tiff:
                png_bytes, metadata = tiff_to_png_preview(file_bytes, filename)
            else:
                # For non-TIFF, try to open and convert to PNG if needed
                try:
                    img = Image.open(BytesIO(file_bytes))
                    # If already PNG/JPEG, return as-is
                    if img.format in ('PNG', 'JPEG', 'JPG'):
                        png_bytes = file_bytes
                        metadata = {
                            'original_filename': filename,
                            'is_tiff': False,
                            'converted': False,
                            'bit_depth': 8,
                            'pages': 1,
                            'width': img.width,
                            'height': img.height
                        }
                    else:
                        # Convert to PNG
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        buffer.seek(0)
                        png_bytes = buffer.read()
                        metadata = {
                            'original_filename': filename,
                            'is_tiff': False,
                            'converted': True,
                            'bit_depth': 8,
                            'pages': 1,
                            'width': img.width,
                            'height': img.height
                        }
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Failed to process image: {str(e)}'
                    }), 400
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(png_bytes)
            
            # Save metadata
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Encode PNG as base64 for frontend
        preview_base64 = base64.b64encode(png_bytes).decode('utf-8')
        preview_url = f"data:image/png;base64,{preview_base64}"
        
        # Get image dimensions
        img = Image.open(BytesIO(png_bytes))
        
        return jsonify({
            'success': True,
            'previewUrl': preview_url,
            'mimeType': 'image/png',
            'width': img.width,
            'height': img.height,
            'metadata': metadata
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Preview conversion failed: {str(e)}'
        }), 500


@app.route('/api/agent/preprocess/<run_id>', methods=['POST'])
def run_preprocessing(run_id):
    """
    Run preprocessing on all images in a manifest.
    
    Body (optional):
    {
        "status_filter": ["recommended", "approved", "edited"],
        "force_reprocess": false
    }
    
    Returns:
    {
        "success": true,
        "summary": {
            "total": 5,
            "success": 4,
            "failed": 1,
            "cancelled": 0
        },
        "logs": ["..."]
    }
    """
    try:
        from core.preprocess import PreprocessRunner
        
        data = request.json or {}
        status_filter = data.get('status_filter')
        force_reprocess = data.get('force_reprocess', False)
        
        # Get manifest
        manifest = get_manifest_manager(run_id)
        
        # Collect logs
        logs = []
        def log_callback(msg):
            logs.append(msg)
            print(msg)
        
        # Run preprocessing
        runner = PreprocessRunner(
            manifest_manager=manifest,
            log_callback=log_callback
        )
        
        summary = runner.run(
            status_filter=status_filter,
            force_reprocess=force_reprocess
        )
        
        return jsonify({
            'success': True,
            'summary': summary,
            'logs': logs
        })
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Agent Chat API Server")
    print("=" * 60)
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API key configured (length: {len(api_key)})")
    else:
        print("⚠️  OpenAI API key NOT configured!")
        print("   Set environment variable: export OPENAI_API_KEY='your-key-here'")
    
    print("")
    print("Available endpoints:")
    print("  GET  /api/agent/health - Health check")
    print("  POST /api/agent/chat - Chat with agent (auto-saves to manifest)")
    print("  POST /api/agent/quick-recommend - Quick preprocessing recommendations")
    print("  GET  /api/agent/runs - List all runs")
    print("  POST /api/agent/create-run - Create new run")
    print("  GET  /api/agent/manifest/<run_id> - Get manifest for run")
    print("  GET  /api/agent/manifest/<run_id>/image/<image_id> - Get image record")
    print("  POST /api/agent/manifest/<run_id>/image/<image_id>/approve - Approve params")
    print("  POST /api/agent/manifest/<run_id>/image/<image_id>/edit - Edit params")
    print("  POST /api/agent/preprocess/<run_id> - Run preprocessing on manifest")
    print("  POST /api/preview - Convert TIFF to PNG for browser preview")
    print("=" * 60)
    
    app.run(debug=True, port=5001)  # Different port from main API

