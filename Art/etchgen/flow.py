"""Orientation field for flow-guided strokes.

Structure tensor gives, per pixel, the dominant edge tangent direction and
a coherence measure (how directional the neighborhood actually is). Where
coherence is low (flat regions), the field blends toward a base hatch
angle so strokes stay orderly instead of following noise.

Angles have a 180-degree ambiguity, so all blending/sampling happens in
doubled-angle vector space (cos 2t, sin 2t).
"""

import cv2
import numpy as np


def orientation_field(gray, smooth_sigma=8.0, base_angle_deg=45.0, blend=1.0):
    """Returns (U, V): doubled-angle unit vectors of the stroke direction.

    Direction follows image edge *tangents* (strokes run along features).
    blend in [0,1]: how much low-coherence areas fall back to base_angle.
    """
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), smooth_sigma)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), smooth_sigma)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), smooth_sigma)

    # gradient orientation (major eigenvector) in doubled-angle space
    u_g = jxx - jyy
    v_g = 2.0 * jxy
    mag = np.hypot(u_g, v_g)
    trace = jxx + jyy
    coherence = np.where(trace > 1e-8, mag / (trace + 1e-8), 0.0).astype(np.float32)

    # tangent = gradient rotated 90 deg; in doubled-angle space that's a sign flip
    with np.errstate(invalid="ignore", divide="ignore"):
        u_t = np.where(mag > 1e-8, -u_g / (mag + 1e-12), 0.0)
        v_t = np.where(mag > 1e-8, -v_g / (mag + 1e-12), 0.0)

    a2 = np.deg2rad(2.0 * base_angle_deg)
    w = np.clip(coherence, 0.0, 1.0) ** 0.75 if blend > 0 else np.ones_like(coherence)
    u = w * u_t + (1.0 - w) * blend * np.cos(a2) + (1.0 - w) * (1.0 - blend) * u_t
    v = w * v_t + (1.0 - w) * blend * np.sin(a2) + (1.0 - w) * (1.0 - blend) * v_t
    n = np.hypot(u, v)
    u = np.where(n > 1e-8, u / (n + 1e-12), np.cos(a2)).astype(np.float32)
    v = np.where(n > 1e-8, v / (n + 1e-12), np.sin(a2)).astype(np.float32)
    return u, v


def constant_field(shape, angle_deg):
    """Uniform direction field at angle_deg (doubled-angle form)."""
    a2 = np.deg2rad(2.0 * angle_deg)
    u = np.full(shape, np.cos(a2), dtype=np.float32)
    v = np.full(shape, np.sin(a2), dtype=np.float32)
    return u, v


def rotate_field(u, v, delta_deg):
    """Rotate a doubled-angle field by delta degrees."""
    d2 = np.deg2rad(2.0 * delta_deg)
    c, s = np.cos(d2), np.sin(d2)
    return (u * c - v * s).astype(np.float32), (u * s + v * c).astype(np.float32)
