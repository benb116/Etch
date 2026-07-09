"""Cost-field A* routing for connector lines.

Bridges connecting disjoint components are routed using A* search on a
downsampled cost grid, where cost = 1 + β·brightness. This hides connectors
along dark regions or already-drawn lines instead of slashing across white space.
"""

import heapq

import numpy as np
from scipy import ndimage


def build_cost_field(tone, drawn_mask=None, darkness_weight=10.0, downsampling=4):
    """Build a cost field for A* routing.

    Args:
        tone: brightness map (0=black, 1=white), float32
        drawn_mask: optional boolean mask of already-drawn regions (high-cost)
        darkness_weight: weight for darkness (0=ignore, high=prefer dark)
        downsampling: reduction factor (4 = 4× smaller grid)

    Returns:
        (cost_grid, origin) where origin is the (row, col) offset of cost_grid
        in the original space
    """
    h, w = tone.shape

    # Downsample tone
    ds_h = (h + downsampling - 1) // downsampling
    ds_w = (w + downsampling - 1) // downsampling

    # Resample tone to downsampled resolution (nearest neighbor)
    tone_ds = tone[::downsampling, ::downsampling][:ds_h, :ds_w]

    # Base cost: 1 + darkness weight × (1 - brightness)
    darkness = np.clip(1.0 - tone_ds, 0.0, 1.0)
    cost = 1.0 + darkness_weight * darkness

    # If drawn_mask provided, reduce cost along drawn regions
    if drawn_mask is not None:
        drawn_ds = drawn_mask[::downsampling, ::downsampling][:ds_h, :ds_w]
        cost[drawn_ds] *= 0.1  # Very low cost along already-drawn areas

    return cost.astype(np.float32), (0, 0)


def heuristic(a, b):
    """Euclidean distance heuristic for A*."""
    return np.hypot(a[0] - b[0], a[1] - b[1])


def a_star_path(cost_grid, start, end):
    """A* search for low-cost path on a grid.

    Args:
        cost_grid: (H, W) cost array
        start: (row, col) starting position
        end: (row, col) goal position

    Returns:
        list of (row, col) waypoints in order from start to end
        Returns empty list if no path found
    """
    h, w = cost_grid.shape

    # Clamp to valid range
    start = (max(0, min(start[0], h - 1)), max(0, min(start[1], w - 1)))
    end = (max(0, min(end[0], h - 1)), max(0, min(end[1], w - 1)))

    if start == end:
        return [start]

    # Priority queue: (f_score, counter, position, path)
    counter = 0
    open_set = [(0, counter, start, [start])]
    visited = set()

    g_score = {start: 0}

    while open_set:
        _, _, current, path = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)

        if current == end:
            return path

        cy, cx = current

        # Explore 4-neighbors (no diagonals for simplicity)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx

            if not (0 <= ny < h and 0 <= nx < w):
                continue

            if (ny, nx) in visited:
                continue

            tentative_g = g_score[current] + cost_grid[ny, nx]

            if (ny, nx) not in g_score or tentative_g < g_score[(ny, nx)]:
                g_score[(ny, nx)] = tentative_g
                f_score = tentative_g + heuristic((ny, nx), end)

                counter += 1
                heapq.heappush(open_set, (f_score, counter, (ny, nx), path + [(ny, nx)]))

    return []  # No path found


def route_connector(start, end, tone, drawn_mask=None, darkness_weight=10.0, downsampling=4):
    """Route a connector line from start to end using A* on cost field.

    Args:
        start: (x, y) starting point in original space
        end: (x, y) goal point in original space
        tone: brightness map (float32, 0=black, 1=white)
        drawn_mask: optional boolean mask of drawn regions
        darkness_weight: preference for dark regions
        downsampling: grid downsampling factor

    Returns:
        (N, 2) array of waypoints from start to end in original space
        If routing fails, returns straight line
    """
    if tone is None:
        # No tone info; use straight line
        return np.array([start, end], dtype=np.float32)

    try:
        # Convert to downsampled grid coordinates
        start_grid = (int(start[1] / downsampling), int(start[0] / downsampling))
        end_grid = (int(end[1] / downsampling), int(end[0] / downsampling))

        # Build cost field
        cost_grid, _ = build_cost_field(tone, drawn_mask, darkness_weight, downsampling)

        # A* search
        path = a_star_path(cost_grid, start_grid, end_grid)

        if not path:
            # Routing failed; return straight line
            return np.array([start, end], dtype=np.float32)

        # Convert back to original space
        waypoints = np.array([(p[1] * downsampling, p[0] * downsampling) for p in path], dtype=np.float32)

        # Ensure start and end match exactly
        waypoints[0] = start
        waypoints[-1] = end

        return waypoints

    except Exception:
        # Fall back to straight line on any error
        return np.array([start, end], dtype=np.float32)


def rasterize_polylines(polylines, shape):
    """Rasterize polylines into a binary mask.

    Args:
        polylines: list of (N, 2) float arrays
        shape: (height, width) output shape

    Returns:
        Boolean mask where True = drawn
    """
    mask = np.zeros(shape, dtype=bool)

    for polyline in polylines:
        if len(polyline) < 2:
            continue

        pts = polyline.astype(np.int32)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]

            # Bresenham line to mark mask
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x1 > x0 else -1
            sy = 1 if y1 > y0 else -1
            err = dx - dy

            x, y = x0, y0
            while True:
                x = max(0, min(x, shape[1] - 1))
                y = max(0, min(y, shape[0] - 1))
                mask[y, x] = True

                if x == x1 and y == y1:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    return mask
