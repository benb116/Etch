"""Convert solved strokes to lineset JSON format matching the Etch-a-Sketch output contract."""

import json
import numpy as np

from .geometry import rdp


def scale_and_center(points, canvas_size=(1500, 865), margin=80):
    """Scale and center points to canvas, preserving aspect ratio.

    Args:
        points: (N,2) float array of x,y coordinates
        canvas_size: (width, height) target canvas size
        margin: border margin in pixels

    Returns:
        (N,2) float32 array scaled and centered
    """
    if len(points) < 2:
        return points.astype(np.float32)

    points = np.asarray(points, dtype=np.float64)

    # Find bounds
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    if width < 1e-6 or height < 1e-6:
        # Degenerate case: all points at or near same location
        return points.astype(np.float32)

    # Available canvas space (after margin)
    cw = canvas_size[0] - 2 * margin
    ch = canvas_size[1] - 2 * margin

    # Scale to fit, preserving aspect ratio
    scale = min(cw / width, ch / height)

    # Center in available space
    scaled = points - [x_min, y_min]  # translate to origin
    scaled = scaled * scale

    # Center in canvas
    sx_min, sy_min = scaled.min(axis=0)
    sx_max, sy_max = scaled.max(axis=0)
    sx_center = (sx_min + sx_max) / 2.0
    sy_center = (sy_min + sy_max) / 2.0

    cx_center = cw / 2.0
    cy_center = ch / 2.0

    scaled = scaled + [cx_center - sx_center, cy_center - sy_center]
    scaled = scaled + [margin, margin]

    return scaled.astype(np.float32)


def round_points(points, decimals=1):
    """Round points to specified decimal places (reduce JSON size)."""
    points = np.asarray(points, dtype=np.float32)
    factor = 10 ** decimals
    return (np.round(points * factor) / factor).astype(np.float32)


def points_to_lineset(points, name="drawing", px_speed=50, px_per_rev=200,
                      canvas_size=(1500, 865), margin=80, decimals=1):
    """Convert points to lineset JSON format.

    Args:
        points: (N,2) array or list of x,y coordinates
        name: lineset name
        px_speed: motor speed (pixels per time unit)
        px_per_rev: pixels per motor revolution
        canvas_size: target canvas (width, height)
        margin: canvas margin
        decimals: rounding precision

    Returns:
        dict with lineset structure
    """
    points = np.asarray(points)
    if len(points) == 0:
        points = np.zeros((0, 2), dtype=np.float32)

    # Scale and center
    points = scale_and_center(points, canvas_size, margin)

    # Round to reduce JSON size
    points = round_points(points, decimals)

    # Convert to list of [x, y] pairs
    point_list = points.tolist()

    return {
        "name": name,
        "pxSpeed": px_speed,
        "pxPerRev": px_per_rev,
        "points": point_list,
    }


def save_lineset_json(lineset, path):
    """Save lineset dict to JSON file."""
    with open(path, 'w') as f:
        json.dump(lineset, f, separators=(',', ':'))


def lineset_from_polylines(polylines, name="drawing", simplify_tol=0.4, **kwargs):
    """Convert list of polylines to a single lineset.

    Args:
        polylines: list of (N,2) arrays
        name: output name
        simplify_tol: RDP tolerance for final simplification
        **kwargs: passed to points_to_lineset

    Returns:
        lineset dict
    """
    if not polylines:
        return points_to_lineset([], name=name, **kwargs)

    # Concatenate all polylines
    all_points = np.concatenate(polylines, axis=0)

    # Simplify if requested
    if simplify_tol > 0:
        all_points = rdp(all_points, simplify_tol)

    return points_to_lineset(all_points, name=name, **kwargs)
