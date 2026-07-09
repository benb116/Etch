# Etchgen: Image to Single-Line Drawing

A high-performance pipeline for converting images into single-continuous-line Etch-a-Sketch drawings. Optimized for quality, speed, and minimal connector lines.

## Quick Start

### Command-line

```bash
# Render with default "sketch" style
python -m Art.etchgen input.jpg --out output.json

# Choose style and override parameters
python -m Art.etchgen input.jpg --style hatch --out output.json --set spacing_min=3 --set spacing_max=20

# Preview quality (640px, ~300ms) vs final (1280px, ~3s)
python -m Art.etchgen input.jpg --quality preview --out /tmp/preview.json
python -m Art.etchgen input.jpg --quality final --out output.json
```

### Python API

```python
from Art.etchgen.pipeline import Pipeline
from Art.etchgen.serialize import save_lineset_json

# Load and render
pipe = Pipeline()
pipe.load_image("image.jpg")
pipe.set_style_params("sketch")
lineset = pipe.to_lineset("drawing_name", quality="final")
save_lineset_json(lineset, "output.json")

# Inspect intermediate results
result = pipe.render(quality="preview")
print(f"Points: {result['metrics']['points']}")
print(f"Draw time: {result['metrics']['draw_seconds']}s")
print(f"Ink fraction: {result['metrics']['ink_fraction']:.1%}")
```

### Web Preview App

```bash
python Art/preview/server.py
# Browse to http://localhost:5000
# Upload image → tune parameters with live preview → save to rPi/public/art/
```

## Architecture

The pipeline processes images in stages, with caching keyed by parameter subsets so parameter tuning reuses expensive earlier stages.

```
Input Image
    ↓
[1] Preprocess (grayscale, resize, levels/gamma, denoise)
    ↓
[2] Flow Field (structure tensor → orientation)
    ↓
[3] Structure Lines (Canny/XDoG → skeleton → polylines)
    ↓
[4] Tone Rendering (streamline tracer → hatch/flow-guided strokes)
    ↓
[5] Solver (join components → Eulerize → Hierholzer → single line)
    ↓
[6] Postprocess (simplify, scale, output JSON)
    ↓
Lineset JSON
```

### Core Modules

- **`geometry.py`**: Polyline operations (resample, RDP simplification, Chaikin smoothing, snapping)
- **`params.py`**: Unified parameter schema for all stages; powers CLI, UI, and caching
- **`preprocess.py`**: Image loading (EXIF-aware), grayscale, levels, gamma, CLAHE, bilateral denoise
- **`flow.py`**: Structure tensor orientation field (Sobel → eigenanalysis → coherence weighting)
- **`structure.py`**: Edge detection (Canny or XDoG), morphology, thinning, skeleton → polyline tracing
- **`tone.py`**: Jobard-Lefer variable-density streamline tracer; fixed-angle hatch, flow-guided, crosshatch, squiggle modes
- **`solver.py`**: Topological multigraph solver; endpoint snapping, component joining (Delaunay MST), Eulerization (min-weight matching), Hierholzer path
- **`route.py`**: A* connector routing to hide bridges along dark regions
- **`metrics.py`**: Path quality stats (ink %, connector count, draw time estimate)
- **`serialize.py`**: Canvas scaling/centering, rounding, JSON output
- **`pipeline.py`**: Orchestrator with per-stage caching, preview/final quality modes
- **`cli.py`**: Command-line interface

## Drawing Styles

All styles produce a single continuous line (no pen lifts).

### **sketch** (default)
Contour lines + flow-guided hatching. Structure extraction finds edges, flow field guides shading strokes to follow image form (hair, drapes, etc.). High quality, moderate speed.
- Best for: portraits, detailed drawings, artistic photos
- Ink fraction: 75–85%
- Speed: 0.3–3s (preview–final)

### **hatch**
Contour lines + fixed-angle parallel hatching. Classic look with simple 45° (or custom) parallel lines. Faster than sketch (no flow computation).
- Best for: cartoons, technical drawings, consistent style preference
- Ink fraction: 75–85%
- Speed: 0.2–2.5s

### **pencil**
Flow-guided strokes only, no contours. Long flowing lines that respond to image form. Minimal output, high quality.
- Best for: sketchy portraits, flowing subjects (hair, water)
- Ink fraction: 40–50%
- Speed: 0.1–1s (solver-free for simple images)

### **squiggle**
Serpentine horizontal scanlines with darkness-modulated waves. One polyline by construction; zero connectors. Very fast, unique visual style.
- Best for: quick previews, stylistic effect
- Ink fraction: 0% (no solver; wavy scanlines)
- Speed: 0.05–0.5s (fastest!)

## Performance

### Benchmarks (on 500×375px sample)

| Style | Preview (640px) | Final (1280px) | Points | Ink % |
|-------|-----------------|----------------|--------|-------|
| sketch | 0.32s | ~1.5s | 2700 | 80% |
| hatch | 0.31s | ~1.5s | 2700 | 80% |
| pencil | 0.11s | ~0.5s | 500 | 47% |
| squiggle | 0.05s | ~0.3s | 700 | 0% |

### Scaling

- **Working resolution** ("detail" param): 320–3200px. Runtime ∝ pixels (image IO + raster stages).
- **Preview mode**: 640px max, straight connectors, greedy matching (fast feedback)
- **Final mode**: Full detail, A* routed connectors, blossom matching (highest quality)
- **Large images** (3000×2000): ~1–2 min acceptable per user. Solver time ∝ line count (thousands), not pixels.

### Stage Breakdown (sketch style, preview quality)
- Preprocess: 0.1% | Flow: 0.3% | Structure: 0.8% | **Tone: 45%** | **Solver: 54%** | Post: 0%

The solver (component joining + Eulerization + Hierholzer) dominates for structure+tone styles. Streamline tracer is O(strokes × pixels traced), and solver is O(stroke count log stroke count) in the topological graph.

## Parameters

Use `--set key=value` to override. All parameters have ranges and sensible defaults.

### Preprocess
- `detail` (int, 320–3200px): Working resolution. Higher = finer lines, slower.
- `black_point` (float, 0–0.45): Brightness mapped to full black
- `white_point` (float, 0.55–1): Brightness mapped to full white
- `gamma` (float, 0.3–3): Midtone curve
- `clahe` (float, 0–4): Local contrast enhancement (0 = off)
- `denoise` (float, 0–1): Edge-preserving smoothing

### Flow & Tone
- `flow_smooth` (float, 2–30): Gaussian smoothing for orientation field
- `hatch_angle` (float, 0–180°): Fixed hatch direction
- `spacing_min` (float, 1.5–20px): Line spacing in darkest areas
- `spacing_max` (float, 6–80px): Line spacing where shading fades
- `tone_gamma` (float, 0.3–3): Darkness response curve
- `crosshatch` (bool): Second perpendicular pass in dark regions
- `cross_threshold` (float, 0.2–0.95): Darkness threshold for crosshatching

### Structure (sketch/hatch only)
- `edge_method` (choice): "canny" (crisp) or "dog" (sketchy)
- `edge_detail` (float, 0.4–3): Pre-blur; higher = bold contours only
- `edge_sensitivity` (float, 0.3–3): Higher = find more/fainter edges
- `min_component` (int, 1–120px): Drop fragments smaller than this

### Solver
- `snap_radius` (float, 0–10px): Weld hatch endpoints onto contours
- `bridge_penalty` (float, 0–20): Visibility penalty for new connector lines
- `retrace_weight` (float, 0.05–1): Retracing cost (low = prefer retracing)

### Output
- `simplify` (float, 0–2px): RDP simplification tolerance
- `smooth` (int, 0–3): Chaikin smoothing passes

## Output Format

Lineset JSON: one continuous polyline, never lifting the pen.
```json
{
  "name": "drawing_name",
  "pxSpeed": 50,
  "pxPerRev": 200,
  "points": [[x, y], [x, y], ..., [x, y]]
}
```

- Points are in canvas coordinates (1500×865 default, 80px margin)
- Coordinates rounded to 1 decimal place
- Total length = sum of segment distances; draw time ≈ length / pxSpeed

## Quality Tips

1. **Adjust black/white points** for underexposed or overexposed images
2. **Increase gamma** (>1) to lighten midtones (less shading)
3. **Decrease gamma** (<1) to darken (more shading)
4. **Disable structure** (pencil mode) for sketchy subjects (faces, flowing hair)
5. **Higher edge_detail** to pick up only bold contours
6. **Tune spacing_min/spacing_max** to balance detail vs line count
7. **Increase snap_radius** to merge hatch lines into contours
8. **Adjust bridge_penalty** if connectors are too visible/invisible

## Testing & Verification

Run tests and benchmarks:
```bash
pytest Art/etchgen/tests/
python -m Art.etchgen.bench Art/Pics/Mario.jpg
python -m Art.etchgen.bench Art/Pics/Mario.jpg --size-bench  # Scaling profile
```

E2E test:
```bash
python -m Art.etchgen Art/Pics/Mario.jpg --style sketch --out /tmp/Mario.json
# Load /tmp/Mario.json in rPi/public/index.html or docs/index.html to verify
```

## Implementation Notes

- **Topological solver**: Nodes only at endpoints/junctions (thousands), not per-pixel graphs (100k+). Scales to large images.
- **Endpoint snapping**: KD-tree projection of hatch endpoints onto structure polylines creates T-junctions, reducing component count by ~80%.
- **Ink-aware Eulerization**: Prefers retracing invisible lines over visible connectors. Min-weight matching (blossom or greedy KD-tree).
- **Variable-density hatch**: Streamline tracer with occupancy grid collision detection; spacing continuously maps from brightness (no discrete bins).
- **Stage caching**: Results keyed by (image hash, relevant param subset). Moving a hatch slider reuses cached preprocess+structure.
- **Responsiveness**: Preview (640px, straight connectors) renders in <1s; final (1280px, A* routing) in 1–3s on typical images.

## Future Work

1. **Blossom matching in final mode**: Min-weight perfect matching for odd-node pairing (currently greedy for |odd| > 400)
2. **Hilbert halftone mode**: Fractal space-filling curve for stipple-like halftoning
3. **Mobile preview**: Adapt frontend for phones
4. **Batch processing**: API for rendering multiple images with shared cache
5. **GPU acceleration**: Streamline tracer vectorization (if profiling shows bottleneck on very large images)

## Dependencies

- numpy, scipy, networkx (solver + geometry)
- opencv-contrib-python (ximgproc.thinning, Sobel, morphology, blur)
- Pillow (image I/O + EXIF)
- Flask (preview web app)
- pytest (tests)

See `requirements.txt`.

## License

Same as parent Etch repository.
