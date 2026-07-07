"""Contour ("structure") line extraction: edge detect -> clean -> thin to
1px skeleton -> trace into polylines with shared junction points."""

import cv2
import numpy as np

from .geometry import rdp

# 8-neighborhood, ordered so adjacent entries are adjacent directions
_NBRS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def edge_map(gray, p):
    """Binary edge image from the conditioned brightness map."""
    method = p["edge_method"]
    if method == "none":
        return np.zeros(gray.shape, dtype=np.uint8)
    u8 = (gray * 255).astype(np.uint8)
    sigma = float(p["edge_detail"])
    blurred = cv2.GaussianBlur(u8, (0, 0), sigma)
    sens = float(p["edge_sensitivity"])
    if method == "canny":
        v = float(np.median(blurred))
        lo = max(4.0, (1.0 - 0.33 * sens) * v / max(sens, 0.4))
        hi = max(lo * 2.0, (1.0 + 0.33) * v / max(sens, 0.4))
        edges = cv2.Canny(blurred, lo, hi, L2gradient=True)
    else:  # 'dog' - difference of Gaussians band on the dark side of edges
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
        d = g1 - g2
        tau = float(d.std()) / max(sens, 0.05)
        edges = ((d < -tau) * 255).astype(np.uint8)

    # bridge 1px gaps, then drop dust
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    min_area = int(p["min_component"])
    if min_area > 1 and n > 1:
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
        keep = np.ones(n, dtype=bool)
        keep[small] = False
        keep[0] = False
        edges = np.where(keep[labels], 255, 0).astype(np.uint8)
    return edges


def thin(edges):
    """1px skeleton (Zhang-Suen via opencv-contrib)."""
    if not edges.any():
        return edges.astype(bool)
    return cv2.ximgproc.thinning(edges).astype(bool)


def trace_skeleton(skel):
    """Skeleton bool image -> list of (N,2) float32 polylines in x,y.

    Pixels with != 2 neighbors are nodes (tips and junctions); paths are
    walked node-to-node consuming interior pixels, then leftover pure
    cycles are walked from an arbitrary start.
    """
    h, w = skel.shape
    sk = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    deg = cv2.filter2D(sk, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    is_node = skel & (deg != 2)

    visited = np.zeros_like(skel, dtype=bool)   # interior path pixels consumed
    emitted = set()                              # node-node adjacencies already taken
    polylines = []

    def neighbors(y, x):
        for dy, dx in _NBRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                yield ny, nx

    def walk(sy, sx, ny, nx):
        """Walk from node (sy,sx) into interior pixel (ny,nx) until a node."""
        path = [(sx, sy), (nx, ny)]
        visited[ny, nx] = True
        py, px = sy, sx
        cy, cx = ny, nx
        while True:
            nxt = None
            for wy, wx in neighbors(cy, cx):
                if (wy, wx) == (py, px):
                    continue
                if is_node[wy, wx]:
                    nxt = (wy, wx, True)
                    break
                if not visited[wy, wx]:
                    nxt = (wy, wx, False)
                    break
            if nxt is None:
                return path  # dead end (shouldn't happen on clean skeletons)
            wy, wx, stop = nxt
            path.append((wx, wy))
            if stop:
                return path
            visited[wy, wx] = True
            py, px, cy, cx = cy, cx, wy, wx

    node_pts = np.argwhere(is_node)
    for y, x in node_pts:
        for ny, nx in neighbors(y, x):
            if is_node[ny, nx]:
                key = ((y, x), (ny, nx)) if (y, x) < (ny, nx) else ((ny, nx), (y, x))
                if key not in emitted:
                    emitted.add(key)
                    polylines.append([(x, y), (nx, ny)])
            elif not visited[ny, nx]:
                polylines.append(walk(y, x, ny, nx))

    # leftover pure cycles (no node anywhere on them)
    remaining = skel & ~visited & ~is_node
    ys, xs = np.nonzero(remaining)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        # walk the loop
        path = [(x, y)]
        visited[y, x] = True
        py, px = -1, -1
        cy, cx = y, x
        while True:
            nxt = None
            for wy, wx in neighbors(cy, cx):
                if (wy, wx) == (py, px) or visited[wy, wx]:
                    continue
                nxt = (wy, wx)
                break
            if nxt is None:
                break
            wy, wx = nxt
            path.append((wx, wy))
            visited[wy, wx] = True
            py, px, cy, cx = cy, cx, wy, wx
        if len(path) > 2:
            path.append((x, y))  # close the loop
            polylines.append(path)

    return [np.array(pl, dtype=np.float32) for pl in polylines if len(pl) >= 2]


def extract_structure(gray, p):
    """gray -> (polylines, edge image). The main structure-stage entry."""
    edges = edge_map(gray, p)
    skel = thin(edges)
    polylines = trace_skeleton(skel)
    tol = float(p["structure_simplify"])
    if tol > 0:
        polylines = [rdp(pl, tol) for pl in polylines]
    return polylines, edges
