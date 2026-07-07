"""Polyline operations. A polyline is an (N, 2) float array of x,y points."""

import numpy as np


def as_polyline(pts):
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def dedupe(pts, tol=1e-6):
    """Drop consecutive points closer than tol."""
    pts = as_polyline(pts)
    if len(pts) < 2:
        return pts
    d = np.hypot(*(np.diff(pts, axis=0).T))
    keep = np.concatenate(([True], d > tol))
    return pts[keep]


def polyline_length(pts):
    pts = as_polyline(pts)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*(np.diff(pts, axis=0).T)).sum())


def cum_arclength(pts):
    """Cumulative arc length at each vertex, starting at 0."""
    pts = np.asarray(pts, dtype=np.float64)
    d = np.hypot(*(np.diff(pts, axis=0).T))
    return np.concatenate(([0.0], np.cumsum(d)))


def resample(pts, step):
    """Evenly respace vertices ~step apart along the polyline (endpoints kept)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.astype(np.float32)
    s = cum_arclength(pts)
    total = s[-1]
    if total <= step:
        return pts[[0, -1]].astype(np.float32)
    n = int(np.ceil(total / step)) + 1
    si = np.linspace(0.0, total, n)
    x = np.interp(si, s, pts[:, 0])
    y = np.interp(si, s, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def densify(pts, max_step):
    """Insert vertices so no segment is longer than max_step.

    Unlike resample(), original vertices are all preserved, so the shape
    is exact; only long segments get subdivided.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts.astype(np.float32)
    out = [pts[:1]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = np.hypot(*(b - a))
        if d > max_step:
            n = int(np.ceil(d / max_step))
            t = np.linspace(0.0, 1.0, n + 1)[1:]
            out.append(a + t[:, None] * (b - a))
        else:
            out.append(b[None, :])
    return np.concatenate(out).astype(np.float32)


def rdp(pts, eps):
    """Ramer-Douglas-Peucker simplification (iterative, numpy distances)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3 or eps <= 0:
        return pts.astype(np.float32)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        seg = pts[i + 1:j]
        length = np.hypot(*ab)
        if length < 1e-12:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / length
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep].astype(np.float32)


def chaikin(pts, iterations=1):
    """Chaikin corner-cutting smoothing; endpoints are preserved."""
    pts = np.asarray(pts, dtype=np.float64)
    for _ in range(max(0, iterations)):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * (len(pts) - 1), 2))
        mid[0::2] = q
        mid[1::2] = r
        pts = np.concatenate([pts[:1], mid, pts[-1:]])
    return pts.astype(np.float32)
