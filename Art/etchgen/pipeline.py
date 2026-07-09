"""Pipeline orchestrator with per-stage caching.

The pipeline runs stages sequentially, caching results keyed by parameter subsets
so that parameter tuning reuses expensive earlier stages (e.g., moving a hatch
slider doesn't re-run structure extraction).
"""

import hashlib
import pickle
import tempfile
from pathlib import Path

import numpy as np

from . import params as p_module
from .preprocess import preprocess
from .flow import orientation_field, constant_field
from .structure import extract_structure
from .tone import render_streamline_tone, render_squiggle
from .solver import solve
from .geometry import chaikin
from .serialize import lineset_from_polylines
from .metrics import compute_metrics


class PipelineCache:
    """In-memory + on-disk cache for pipeline stages."""

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "etchgen_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}

    def _key_hash(self, key):
        """Convert a hashable key to a string hash."""
        key_bytes = pickle.dumps(key)
        return hashlib.sha256(key_bytes).hexdigest()[:16]

    def get(self, key):
        """Retrieve from memory or disk cache."""
        key_str = self._key_hash(key)
        if key_str in self._memory_cache:
            return self._memory_cache[key_str]

        cache_file = self.cache_dir / f"{key_str}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    val = pickle.load(f)
                    self._memory_cache[key_str] = val
                    return val
            except Exception:
                pass
        return None

    def set(self, key, value):
        """Store to memory and disk cache."""
        key_str = self._key_hash(key)
        self._memory_cache[key_str] = value

        cache_file = self.cache_dir / f"{key_str}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass


class Pipeline:
    """Full render pipeline with stage caching and configurable quality."""

    def __init__(self, cache_dir=None):
        self.cache = PipelineCache(cache_dir)
        self.style = "sketch"
        self.params = {}
        self.rgb = None
        self.preprocessed = None
        self.structure_polylines = None
        self.structure_edges = None
        self.field_uv = None
        self.tone_strokes = None
        self.solved_stroke = None
        self.seg_class = np.array([], dtype=np.uint8)

    def load_image(self, image_path_or_array):
        """Load RGB image from file or array."""
        if isinstance(image_path_or_array, str):
            from .preprocess import load_image
            self.rgb = load_image(image_path_or_array)
        else:
            self.rgb = np.asarray(image_path_or_array)

    def set_style_params(self, style, param_overrides=None):
        """Set style and parameters."""
        self.style = style
        self.params = p_module.coerce(style, param_overrides or {})

    def stage_key(self, stage):
        """Get cache key for a stage (params relevant up to and including this stage)."""
        return p_module.stage_key(self.style, self.params, stage)

    def _run_preprocess(self):
        """Preprocess: image conditioning."""
        key = self.stage_key("preprocess")
        result = self.cache.get(key)
        if result is None:
            result = preprocess(self.rgb, self.params)
            self.cache.set(key, result)
        self.preprocessed = result
        return result

    def _run_flow(self):
        """Flow: orientation field."""
        key = self.stage_key("flow")
        result = self.cache.get(key)
        if result is None:
            gray = self.preprocessed["gray"]
            if self.params.get("tone_mode") == "fixed":
                angle = self.params.get("hatch_angle", 45.0)
                result = constant_field(gray.shape, angle)
            else:
                smooth = self.params.get("flow_smooth", 8.0)
                result = orientation_field(gray, smooth_sigma=float(smooth))
            self.cache.set(key, result)
        self.field_uv = result
        return result

    def _run_structure(self):
        """Structure: edge detection + skeleton tracing."""
        key = self.stage_key("structure")
        result = self.cache.get(key)
        if result is None:
            gray = self.preprocessed["gray"]
            result = extract_structure(gray, self.params)
            self.cache.set(key, result)
        self.structure_polylines, self.structure_edges = result
        return result

    def _run_tone(self):
        """Tone: render strokes (flow-guided, fixed-angle, or squiggle)."""
        tone_mode = self.params.get("tone_mode", "flow")

        if tone_mode == "squiggle":
            # Squiggle mode bypasses solver; it's one line by construction
            self.tone_strokes = render_squiggle(self.preprocessed, self.params)
            return self.tone_strokes

        # Flow or fixed hatch: use streamline tracer
        structure_pl = self.structure_polylines if self.params.get("use_structure") else None
        self.tone_strokes = render_streamline_tone(
            self.preprocessed, self.field_uv, self.params, structure_polylines=structure_pl
        )
        return self.tone_strokes

    def _run_solver(self):
        """Solver: join strokes into one line."""
        tone_mode = self.params.get("tone_mode", "flow")

        if tone_mode == "squiggle":
            # Squiggle is already one line; no solver needed
            self.solved_stroke = self.tone_strokes[0] if self.tone_strokes else np.zeros((0, 2))
            self.seg_class = np.array([], dtype=np.uint8)
            return self.solved_stroke

        # Combine structure + tone strokes, solve for single polyline
        all_strokes = (self.structure_polylines or []) + (self.tone_strokes or [])
        if not all_strokes:
            self.solved_stroke = np.zeros((0, 2), dtype=np.float32)
            self.seg_class = np.array([], dtype=np.uint8)
            return self.solved_stroke

        tone_map = self.preprocessed.get("tone") if self.params.get("use_tone_bias") else None
        result = solve(
            all_strokes,
            tone=tone_map,
            **{k: self.params[k] for k in ["snap_radius", "bridge_penalty", "retrace_weight"]
               if k in self.params}
        )
        self.solved_stroke = result.points
        self.seg_class = result.seg_class
        return self.solved_stroke

    def _run_post(self):
        """Postprocess: smoothing + simplification."""
        points = self.solved_stroke
        if len(points) < 2:
            return points

        # Chaikin smoothing (optional)
        smooth_passes = int(self.params.get("smooth", 0))
        if smooth_passes > 0:
            points = chaikin(points, smooth_passes)

        return points

    def render(self, quality="preview"):
        """Render the full pipeline.

        Args:
            quality: 'preview' (640px, fast) or 'final' (1280px, high quality)

        Returns:
            dict with 'points', 'segment_classes', 'metrics', 'stage_timings'
        """
        import time

        if self.rgb is None:
            raise ValueError("No image loaded; call load_image() first")

        # Preview uses lower detail; final uses full detail
        if quality == "preview":
            orig_detail = self.params.get("detail", 1024)
            self.params["detail"] = min(640, orig_detail)
        # (final uses original detail)

        timings = {}

        # Preprocess
        t0 = time.time()
        self._run_preprocess()
        timings["preprocess"] = time.time() - t0

        # Flow
        t0 = time.time()
        self._run_flow()
        timings["flow"] = time.time() - t0

        # Structure (only for styles that use it)
        if self.params.get("use_structure", True):
            t0 = time.time()
            self._run_structure()
            timings["structure"] = time.time() - t0
        else:
            self.structure_polylines = []
            timings["structure"] = 0.0

        # Tone
        t0 = time.time()
        self._run_tone()
        timings["tone"] = time.time() - t0

        # Solve
        t0 = time.time()
        self._run_solver()
        timings["solve"] = time.time() - t0

        # Postprocess
        t0 = time.time()
        points = self._run_post()
        timings["post"] = time.time() - t0

        # Compute metrics
        seg_class = self.seg_class if hasattr(self, 'seg_class') else np.array([])
        metrics = compute_metrics(points, seg_class, px_speed=50.0)

        return {
            "points": points,
            "segment_classes": seg_class,
            "metrics": metrics,
            "stage_timings": timings,
        }

    def to_lineset(self, name="drawing", quality="final"):
        """Render and convert to lineset JSON format."""
        result = self.render(quality=quality)
        points = result["points"]

        # Apply canvas scaling + serialization
        lineset = lineset_from_polylines(
            [points],
            name=name,
            simplify_tol=self.params.get("simplify", 0.4)
        )
        return lineset
