"""Flask preview app for etchgen.

Provides a web interface for uploading images, tuning parameters with live
preview, and saving final linesets to rPi/public/art/.
"""

import io
import json
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from ..etchgen.pipeline import Pipeline
from ..etchgen.params import schema as param_schema, defaults as param_defaults
from ..etchgen.preprocess import decode_image
from ..etchgen.serialize import save_lineset_json

app = Flask(__name__, static_folder='static', template_folder='.')

# Session storage (image bytes + metadata)
SESSIONS = {}
SESSIONS_NEXT_ID = 0

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "rPi" / "public" / "art"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the preview UI."""
    return render_template("index.html")


@app.route("/api/image", methods=["POST"])
def upload_image():
    """Upload an image and create a session.

    Returns:
        {"session_id": str, "width": int, "height": int}
    """
    global SESSIONS_NEXT_ID

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        image_data = file.read()
        rgb = decode_image(image_data)
        h, w = rgb.shape[:2]

        session_id = str(SESSIONS_NEXT_ID)
        SESSIONS_NEXT_ID += 1

        SESSIONS[session_id] = {
            "rgb": rgb,
            "name": secure_filename(file.filename).rsplit('.', 1)[0],
        }

        return jsonify({
            "session_id": session_id,
            "width": w,
            "height": h,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/params")
def get_params():
    """Get parameter schema for a style.

    Query:
        style: 'sketch', 'hatch', 'pencil', 'squiggle'

    Returns:
        List of param dicts (label, type, range, default, etc.)
    """
    style = request.args.get('style', 'sketch')
    try:
        return jsonify(param_schema(style))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/generate", methods=["POST"])
def generate():
    """Render with current parameters.

    Body:
        {
            "session_id": str,
            "style": str,
            "params": {key: value, ...},
            "quality": "preview" or "final"
        }

    Returns:
        {
            "points": [[x,y], ...],
            "segment_classes": [0, 0, 1, 2, ...],  # INK=0, RETRACE=1, CONNECTOR=2
            "metrics": {points, total_len, ink_len, ...},
            "stage_timings": {stage: seconds, ...}
        }
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        style = data.get("style", "sketch")
        params = data.get("params", {})
        quality = data.get("quality", "preview")

        if session_id not in SESSIONS:
            return jsonify({"error": "Session not found"}), 404

        session = SESSIONS[session_id]
        rgb = session["rgb"]

        # Create pipeline and render
        pipe = Pipeline()
        pipe.load_image(rgb)
        pipe.set_style_params(style, params)

        result = pipe.render(quality=quality)

        # Convert numpy arrays to lists for JSON serialization
        return jsonify({
            "points": result["points"].tolist(),
            "segment_classes": result["segment_classes"].tolist(),
            "metrics": result["metrics"],
            "stage_timings": result["stage_timings"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/save", methods=["POST"])
def save_drawing():
    """Render at final quality and save to rPi/public/art/.

    Body:
        {
            "session_id": str,
            "style": str,
            "params": {key: value, ...},
            "name": str  # output filename (without .json)
        }

    Returns:
        {"path": str}
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        style = data.get("style", "sketch")
        params = data.get("params", {})
        name = data.get("name", "drawing")

        if session_id not in SESSIONS:
            return jsonify({"error": "Session not found"}), 404

        session = SESSIONS[session_id]
        rgb = session["rgb"]

        # Create pipeline and render final quality
        pipe = Pipeline()
        pipe.load_image(rgb)
        pipe.set_style_params(style, params)

        lineset = pipe.to_lineset(name=name, quality="final")

        # Sanitize filename
        safe_name = secure_filename(name)
        if not safe_name:
            safe_name = "drawing"

        output_file = OUTPUT_DIR / f"{safe_name}.json"
        save_lineset_json(lineset, str(output_file))

        return jsonify({
            "path": str(output_file),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/preview_image/<session_id>")
def get_preview_image(session_id):
    """Return the uploaded image as PNG for preview display."""
    if session_id not in SESSIONS:
        return "Not found", 404

    try:
        from PIL import Image
        session = SESSIONS[session_id]
        rgb = session["rgb"]

        # Convert to PIL and return as PNG
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        return buf.getvalue(), 200, {
            'Content-Type': 'image/png',
            'Cache-Control': 'no-cache',
        }
    except Exception as e:
        return str(e), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
