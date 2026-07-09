"""Path quality metrics for a solved stroke."""

import numpy as np

from .solver import INK, RETRACE, CONNECTOR


def compute_metrics(points, seg_class, px_speed=50.0):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return {
            "points": int(len(points)), "total_len": 0.0, "ink_len": 0.0,
            "retrace_len": 0.0, "connector_len": 0.0, "connector_count": 0,
            "ink_fraction": 1.0, "draw_seconds": 0.0,
        }
    seglen = np.hypot(*(np.diff(points, axis=0).T))
    seg_class = np.asarray(seg_class)
    total = float(seglen.sum())
    ink = float(seglen[seg_class == INK].sum())
    retrace = float(seglen[seg_class == RETRACE].sum())
    connector = float(seglen[seg_class == CONNECTOR].sum())
    # count connector runs, not segments
    is_conn = (seg_class == CONNECTOR).astype(np.int8)
    runs = int(np.count_nonzero(np.diff(is_conn) == 1) + (is_conn[0] == 1 if len(is_conn) > 0 else 0))
    return {
        "points": int(len(points)),
        "total_len": round(total, 1),
        "ink_len": round(ink, 1),
        "retrace_len": round(retrace, 1),
        "connector_len": round(connector, 1),
        "connector_count": runs,
        "ink_fraction": round(ink / total, 4) if total else 1.0,
        "draw_seconds": round(total / px_speed, 1) if px_speed else 0.0,
    }
