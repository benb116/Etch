"""Tone (shading) renderers.

The workhorse is a variable-density streamline tracer (Jobard-Lefer style):
strokes are integrated through a direction field and die when they come
within the local separation distance `dsep` of an already-placed stroke.
dsep is mapped from image darkness, so spacing varies *continuously* with
tone - no discrete brightness bins.

Occupancy is a raster: every committed stroke stamps disks of radius
~0.5*dsep along itself, so the collision test during tracing is a single
array lookup per step.
"""

from collections import deque

import cv2
import numpy as np

from .geometry import rdp


def darkness_maps(tone, p):
    """tone (brightness 0..1) -> (dsep map float32, allow mask bool)."""
    darkness = np.clip(1.0 - tone, 0.0, 1.0)
    g = float(p["tone_gamma"])
    if abs(g - 1.0) > 1e-3:
        darkness = np.power(darkness, g, dtype=np.float32)
    s_min = float(p["spacing_min"])
    s_max = max(float(p["spacing_max"]), s_min + 0.5)
    dsep = (s_max + (s_min - s_max) * darkness).astype(np.float32)
    allow = tone < float(p["white_cutoff"])
    return dsep, allow


class _Stamper:
    """Disk-stamping occupancy raster with cached integer-radius offsets."""

    def __init__(self, shape):
        self.occ = np.zeros(shape, dtype=bool)
        self.h, self.w = shape
        self._disks = {}

    def _disk(self, r):
        d = self._disks.get(r)
        if d is None:
            rr = np.arange(-r, r + 1)
            dy, dx = np.meshgrid(rr, rr, indexing="ij")
            m = dy * dy + dx * dx <= r * r
            d = (dy[m], dx[m])
            self._disks[r] = d
        return d

    def stamp_points(self, xs, ys, radii):
        for x, y, r in zip(xs, ys, radii):
            dy, dx = self._disk(int(r))
            yy = np.clip(int(y) + dy, 0, self.h - 1)
            xx = np.clip(int(x) + dx, 0, self.w - 1)
            self.occ[yy, xx] = True

    def stamp_polyline(self, pts, radius_map, kill):
        pts = np.asarray(pts)
        if len(pts) == 0:
            return
        xs = np.clip(pts[:, 0], 0, self.w - 1)
        ys = np.clip(pts[:, 1], 0, self.h - 1)
        radii = np.maximum(1, (radius_map[ys.astype(np.intp), xs.astype(np.intp)]
                               * kill).astype(np.intp))
        # stamping every ~r/2 samples is enough coverage at 1px step spacing
        keep = np.zeros(len(pts), dtype=bool)
        i = 0
        while i < len(pts):
            keep[i] = True
            i += max(1, int(radii[i] // 2))
        keep[-1] = True
        self.stamp_points(xs[keep], ys[keep], radii[keep])


def trace_streamlines(field_uv, dsep, allow, p, seed_structure=None,
                      kill=0.5, step_len=1.0, max_points=1_500_000):
    """Trace evenly-spaced strokes through a doubled-angle field.

    field_uv: (U, V) doubled-angle direction maps.
    dsep/allow: from darkness_maps.
    seed_structure: polylines already drawn (contours); stamped into the
        occupancy raster so shading keeps a standoff from them.
    Returns list of (N,2) float32 polylines.
    """
    U, V = field_uv
    h, w = dsep.shape
    occ = _Stamper((h, w))
    if seed_structure:
        for pl in seed_structure:
            occ.stamp_polyline(pl, dsep, kill * 0.6)

    min_stroke = float(p["min_stroke"])
    max_steps = 6000

    # initial seeds: jittered grid, darkest first
    g = float(np.clip(float(p["spacing_min"]) * 1.5, 3.0, 12.0))
    gy, gx = np.mgrid[0:h:g, 0:w:g]
    rng = np.random.default_rng(1234)
    sx = np.clip(gx.ravel() + rng.uniform(-g / 3, g / 3, gx.size), 0, w - 1)
    sy = np.clip(gy.ravel() + rng.uniform(-g / 3, g / 3, gy.size), 0, h - 1)
    okmask = allow[sy.astype(np.intp), sx.astype(np.intp)]
    sx, sy = sx[okmask], sy[okmask]
    order = np.argsort(dsep[sy.astype(np.intp), sx.astype(np.intp)])
    seeds = deque(zip(sx[order], sy[order]))

    occ_arr = occ.occ
    strokes = []
    total_pts = 0

    def direction(x, y, prev):
        u = U[int(y), int(x)]
        v = V[int(y), int(x)]
        ang = 0.5 * np.arctan2(v, u)
        dx, dy = np.cos(ang), np.sin(ang)
        if prev is not None and (dx * prev[0] + dy * prev[1]) < 0:
            return -dx, -dy
        return dx, dy

    def trace_dir(x0, y0, sign):
        pts = []
        x, y = x0, y0
        prev = None
        for _ in range(max_steps):
            d = direction(x, y, prev)
            if prev is None and sign < 0:
                d = (-d[0], -d[1])
            # midpoint step
            mx, my = x + 0.5 * step_len * d[0], y + 0.5 * step_len * d[1]
            if not (0 <= mx < w and 0 <= my < h):
                break
            d2 = direction(mx, my, d)
            nx, ny = x + step_len * d2[0], y + step_len * d2[1]
            if not (0 <= nx < w and 0 <= ny < h):
                break
            iy, ix = int(ny), int(nx)
            if not allow[iy, ix] or occ_arr[iy, ix]:
                break
            pts.append((nx, ny))
            prev = d2
            x, y = nx, ny
        return pts

    spawn_gap = 2.0  # spawn candidates every ~spawn_gap * dsep along strokes

    while seeds and total_pts < max_points:
        x0, y0 = seeds.popleft()
        ix, iy = int(x0), int(y0)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        if occ_arr[iy, ix] or not allow[iy, ix]:
            continue
        fwd = trace_dir(x0, y0, +1)
        bwd = trace_dir(x0, y0, -1)
        pts = bwd[::-1] + [(x0, y0)] + fwd
        if len(pts) * step_len < min_stroke:
            continue
        pl = np.array(pts, dtype=np.float32)
        occ.stamp_polyline(pl, dsep, kill)
        pl = rdp(pl, 0.25)
        strokes.append(pl)
        total_pts += len(pl)

        # spawn new seed candidates perpendicular to this stroke
        i = 0
        raw = pts
        while i < len(raw):
            x, y = raw[i]
            d = dsep[int(y), int(x)]
            if i + 1 < len(raw):
                tx, ty = raw[i + 1][0] - x, raw[i + 1][1] - y
            elif i > 0:
                tx, ty = x - raw[i - 1][0], y - raw[i - 1][1]
            else:
                tx, ty = 1.0, 0.0
            n = np.hypot(tx, ty)
            if n > 1e-9:
                px, py = -ty / n, tx / n
                off = d * 1.15
                seeds.append((x + px * off, y + py * off))
                seeds.append((x - px * off, y - py * off))
            i += max(2, int(d * spawn_gap))

    return strokes


def render_streamline_tone(pre, field_uv, p, structure_polylines=None):
    """Main tone pass (+ optional perpendicular crosshatch pass)."""
    from .flow import rotate_field
    dsep, allow = darkness_maps(pre["tone"], p)
    strokes = trace_streamlines(field_uv, dsep, allow, p,
                                seed_structure=structure_polylines)
    if p.get("crosshatch"):
        darkness = 1.0 - pre["tone"]
        g = float(p["tone_gamma"])
        if abs(g - 1.0) > 1e-3:
            darkness = np.power(np.clip(darkness, 0, 1), g, dtype=np.float32)
        cross_allow = darkness >= float(p["cross_threshold"])
        if cross_allow.any():
            cu, cv_ = rotate_field(*field_uv, 90.0)
            cross_dsep = np.maximum(dsep * 1.25, float(p["spacing_min"]))
            strokes += trace_streamlines(
                (cu, cv_), cross_dsep.astype(np.float32), cross_allow, p,
                seed_structure=structure_polylines)
    return strokes


def render_squiggle(pre, p):
    """Serpentine scanlines with darkness-modulated waves. One polyline by
    construction - bypasses the solver entirely."""
    tone = pre["tone"]
    h, w = tone.shape
    rows = int(p["rows"])
    freq = float(p["squiggle_freq"])
    pitch = h / rows
    parts = []
    for r in range(rows):
        y0 = (r + 0.5) * pitch
        xs = np.arange(0, w, 1.0)
        iy = min(int(y0), h - 1)
        dark = 1.0 - tone[iy, xs.astype(np.intp)]
        dark = np.clip(dark, 0.0, 1.0)
        amp = (pitch * 0.45) * dark
        phase = np.cumsum(2 * np.pi * freq * dark)
        ys = y0 + amp * np.sin(phase)
        row = np.stack([xs, ys], axis=1)
        if r % 2 == 1:
            row = row[::-1]
        parts.append(row)
    pts = []
    for i, row in enumerate(parts):
        if i > 0:
            pts.append(np.array([[parts[i - 1][-1][0], parts[i - 1][-1][1]],
                                 [row[0][0], row[0][1]]]))
        pts.append(row)
    line = np.concatenate(pts).astype(np.float32)
    return [rdp(line, 0.25)]
