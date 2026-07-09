"""Parameter schema: one definition drives the preview UI sliders, the CLI
--set flags, defaults, and the pipeline's per-stage cache keys."""

from dataclasses import dataclass

import numpy as np

STAGES = ["preprocess", "flow", "structure", "tone", "solve", "post"]

STYLES = {
    # flagship: contour lines + tone strokes that follow the image's flow
    "sketch": {"use_structure": True, "tone_mode": "flow"},
    # classic look: contour lines + fixed-angle hatching, continuous spacing
    "hatch": {"use_structure": True, "tone_mode": "fixed"},
    # tone-only, long flowing strokes
    "pencil": {"use_structure": False, "tone_mode": "flow"},
    # serpentine scanlines with darkness-modulated waves; one line by construction
    "squiggle": {"use_structure": False, "tone_mode": "squiggle"},
}


@dataclass
class Param:
    key: str
    label: str
    type: str            # 'float' | 'int' | 'bool' | 'choice'
    default: object
    stage: str
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    choices: tuple = ()
    tooltip: str = ""
    styles: tuple = ()   # empty = applies to all styles


PARAMS = [
    # -- preprocess ---------------------------------------------------------
    Param("detail", "Detail (working px)", "int", 1024, "preprocess", 320, 3200, 32,
          tooltip="Internal working resolution (longest side). More = finer lines, slower."),
    Param("black_point", "Black point", "float", 0.0, "preprocess", 0.0, 0.45, 0.01,
          tooltip="Brightness mapped to full black."),
    Param("white_point", "White point", "float", 1.0, "preprocess", 0.55, 1.0, 0.01,
          tooltip="Brightness mapped to full white."),
    Param("gamma", "Gamma", "float", 1.0, "preprocess", 0.3, 3.0, 0.05,
          tooltip="Midtone curve. <1 darkens (more shading), >1 lightens."),
    Param("clahe", "Local contrast", "float", 0.0, "preprocess", 0.0, 4.0, 0.1,
          tooltip="CLAHE strength for flat/low-contrast photos. 0 = off."),
    Param("denoise", "Denoise", "float", 0.3, "preprocess", 0.0, 1.0, 0.05,
          tooltip="Edge-preserving smoothing; reduces speckle lines."),
    Param("invert", "Invert", "bool", False, "preprocess",
          tooltip="Draw the negative (for white-on-dark art)."),

    # -- flow ---------------------------------------------------------------
    Param("flow_smooth", "Flow smoothness", "float", 8.0, "flow", 2.0, 30.0, 0.5,
          tooltip="How smoothly stroke direction follows the image.",
          styles=("sketch", "pencil")),

    # -- structure ----------------------------------------------------------
    Param("edge_method", "Edge method", "choice", "canny", "structure",
          choices=("canny", "dog", "none"),
          tooltip="Contour extraction. canny = crisp, dog = sketchier.",
          styles=("sketch", "hatch")),
    Param("edge_detail", "Edge scale", "float", 1.2, "structure", 0.4, 3.0, 0.1,
          tooltip="Pre-blur before edge detection. Higher = only bold contours.",
          styles=("sketch", "hatch")),
    Param("edge_sensitivity", "Edge sensitivity", "float", 1.0, "structure", 0.3, 3.0, 0.05,
          tooltip="Higher finds more/fainter edges.",
          styles=("sketch", "hatch")),
    Param("min_component", "Min edge size", "int", 12, "structure", 1, 120, 1,
          tooltip="Drop edge fragments smaller than this (pixels).",
          styles=("sketch", "hatch")),
    Param("structure_simplify", "Contour smoothing", "float", 1.0, "structure", 0.2, 3.0, 0.1,
          tooltip="Simplification tolerance for contour polylines (px).",
          styles=("sketch", "hatch")),

    # -- tone ---------------------------------------------------------------
    Param("spacing_min", "Darkest spacing", "float", 4.0, "tone", 1.5, 20.0, 0.5,
          tooltip="Line spacing in the darkest areas (px).",
          styles=("sketch", "hatch", "pencil")),
    Param("spacing_max", "Lightest spacing", "float", 24.0, "tone", 6.0, 80.0, 1.0,
          tooltip="Line spacing where tone fades out (px).",
          styles=("sketch", "hatch", "pencil")),
    Param("white_cutoff", "White cutoff", "float", 0.88, "tone", 0.5, 1.0, 0.01,
          tooltip="Brightness above this gets no shading lines.",
          styles=("sketch", "hatch", "pencil")),
    Param("tone_gamma", "Shading curve", "float", 1.0, "tone", 0.3, 3.0, 0.05,
          tooltip="Darkness response. <1 = heavier shading.",
          styles=("sketch", "hatch", "pencil")),
    Param("hatch_angle", "Hatch angle", "float", 45.0, "tone", 0.0, 180.0, 5.0,
          tooltip="Hatch direction (degrees). Also the fallback where flow is flat.",
          styles=("sketch", "hatch", "pencil")),
    Param("crosshatch", "Crosshatch", "bool", True, "tone",
          tooltip="Second, perpendicular pass in the darkest areas.",
          styles=("sketch", "hatch", "pencil")),
    Param("cross_threshold", "Crosshatch darkness", "float", 0.55, "tone", 0.2, 0.95, 0.01,
          tooltip="Darkness above which crosshatching kicks in.",
          styles=("sketch", "hatch", "pencil")),
    Param("min_stroke", "Min stroke length", "float", 8.0, "tone", 2.0, 40.0, 1.0,
          tooltip="Discard tone strokes shorter than this (px).",
          styles=("sketch", "hatch", "pencil")),
    Param("rows", "Rows", "int", 60, "tone", 16, 240, 2,
          tooltip="Number of scanline rows.", styles=("squiggle",)),
    Param("squiggle_freq", "Wave frequency", "float", 0.35, "tone", 0.05, 1.0, 0.01,
          tooltip="Max wave cycles per pixel in the darkest areas.",
          styles=("squiggle",)),

    # -- solve --------------------------------------------------------------
    Param("snap_radius", "Weld radius", "float", 3.0, "solve", 0.0, 10.0, 0.5,
          tooltip="Dangling line ends within this distance get welded onto nearby lines.",
          styles=("sketch", "hatch", "pencil")),
    Param("bridge_penalty", "Bridge penalty", "float", 6.0, "solve", 0.0, 20.0, 0.5,
          tooltip="How strongly visible connector lines are avoided (vs retracing).",
          styles=("sketch", "hatch", "pencil")),
    Param("retrace_weight", "Retrace cost", "float", 0.35, "solve", 0.05, 1.0, 0.05,
          tooltip="Lower = happier to retrace existing lines to stay invisible.",
          styles=("sketch", "hatch", "pencil")),

    # -- post ---------------------------------------------------------------
    Param("simplify", "Simplify output", "float", 0.4, "post", 0.0, 2.0, 0.05,
          tooltip="Point-reduction tolerance (px). Higher = smaller files."),
    Param("smooth", "Smooth output", "int", 0, "post", 0, 3, 1,
          tooltip="Chaikin smoothing passes on the final stroke."),
]

_BY_KEY = {p.key: p for p in PARAMS}


def params_for_style(style):
    return [p for p in PARAMS if not p.styles or style in p.styles]


def defaults(style):
    return {p.key: p.default for p in params_for_style(style)}


def coerce(style, overrides):
    """Merge user overrides onto defaults, clamped/typed to the schema.

    Also handles style-specific metadata like use_structure and tone_mode.
    """
    out = defaults(style)

    # Merge style preset values (use_structure, tone_mode)
    style_preset = STYLES.get(style, {})
    for k, v in style_preset.items():
        if k not in _BY_KEY:
            out[k] = v

    for k, v in (overrides or {}).items():
        p = _BY_KEY.get(k)
        if p is None:
            # Allow style metadata through (use_structure, tone_mode, etc.)
            out[k] = v
            continue
        if p.styles and style not in p.styles:
            continue
        if p.type == "bool":
            out[k] = v in (True, "true", "True", 1, "1", "on")
        elif p.type == "choice":
            out[k] = v if v in p.choices else p.default
        elif p.type == "int":
            out[k] = int(np.clip(int(float(v)), p.lo, p.hi)) if _num(v) else p.default
        else:
            out[k] = float(np.clip(float(v), p.lo, p.hi)) if _num(v) else p.default
    return out


def _num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def schema(style):
    """JSON-friendly schema for the UI."""
    out = []
    for p in params_for_style(style):
        out.append({
            "key": p.key, "label": p.label, "type": p.type,
            "default": p.default, "stage": p.stage, "lo": p.lo, "hi": p.hi,
            "step": p.step, "choices": list(p.choices), "tooltip": p.tooltip,
        })
    return out


def stage_key(style, params, stage):
    """Cache key covering this stage and everything upstream of it."""
    idx = STAGES.index(stage)
    relevant = []
    for p in params_for_style(style):
        if STAGES.index(p.stage) <= idx:
            relevant.append((p.key, params.get(p.key, p.default)))
    return (style, stage, tuple(sorted(relevant)))
