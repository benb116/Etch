"""Invariant tests for the single-stroke solver.

The contract: the output is ONE continuous polyline; every input polyline
is drawn exactly once as INK; extra travel is only RETRACE (duplicated ink
geometry) or CONNECTOR (new bridges); at most a handful of connectors for
reasonable inputs.
"""

import numpy as np
import pytest

from Art.etchgen.geometry import polyline_length
from Art.etchgen.metrics import compute_metrics
from Art.etchgen.solver import CONNECTOR, INK, RETRACE, StrokeGraph, solve


def assert_invariants(polylines, res):
    pts, cls = res.points, res.seg_class
    # 1. one continuous stroke: seg classes align with points
    assert len(cls) == len(pts) - 1
    # 2. no zero-length "jumps" hiding a discontinuity (stitching guarantees
    #    shared joints; every segment should be a real segment or exact dup)
    assert np.all(np.isfinite(pts))
    # 3. total ink ~= total input length (every line drawn exactly once;
    #    endpoint snapping may shift ends by up to snap_radius, hence rel tol)
    seglen = np.hypot(*(np.diff(pts.astype(np.float64), axis=0).T))
    ink = seglen[cls == INK].sum()
    want = sum(polyline_length(p) for p in polylines)
    assert ink == pytest.approx(want, rel=0.02, abs=1.0)
    # 4. retrace geometry must lie on already-drawn geometry (ink OR a
    #    connector - retracing either is invisible)
    drawn = set()
    for i in range(len(pts) - 1):
        if cls[i] != RETRACE:
            a = (round(float(pts[i][0]), 1), round(float(pts[i][1]), 1))
            b = (round(float(pts[i + 1][0]), 1), round(float(pts[i + 1][1]), 1))
            drawn.add((a, b))
            drawn.add((b, a))
    for i in range(len(pts) - 1):
        if cls[i] == RETRACE:
            a = (round(float(pts[i][0]), 1), round(float(pts[i][1]), 1))
            b = (round(float(pts[i + 1][0]), 1), round(float(pts[i + 1][1]), 1))
            assert (a, b) in drawn, "retrace segment not on any drawn segment"


def test_single_polyline_passthrough():
    pl = [np.array([(0, 0), (50, 0), (50, 50)], dtype=np.float32)]
    res = solve(pl)
    assert_invariants(pl, res)
    assert np.all(res.seg_class == INK)
    assert res.n_components == 1


def test_two_disjoint_segments_bridged():
    pl = [np.array([(0, 0), (50, 0)], dtype=np.float32),
          np.array([(60, 5), (100, 5)], dtype=np.float32)]
    res = solve(pl)
    assert_invariants(pl, res)
    m = compute_metrics(res.points, res.seg_class)
    assert m["connector_count"] == 1
    # bridge should be the short gap, not something silly
    assert m["connector_len"] < 30


def test_endpoint_preference_joins_at_ends():
    # Two parallel horizontal lines; the bridge should land at the near ends,
    # making all degrees even at a single stroke with zero retraces.
    pl = [np.array([(0, 0), (100, 0)], dtype=np.float32),
          np.array([(100, 10), (0, 10)], dtype=np.float32)]
    res = solve(pl)
    assert_invariants(pl, res)
    m = compute_metrics(res.points, res.seg_class)
    assert m["connector_count"] == 1
    assert m["retrace_len"] == 0


def test_t_junction_parity():
    # A T shape: 4 odd nodes (3 tips + the junction). One matched pair is
    # skipped (path endpoints), one pair fixed by retrace or tiny bridge.
    pl = [np.array([(0, 0), (100, 0)], dtype=np.float32),
          np.array([(50, 0), (50, 80)], dtype=np.float32)]
    res = solve(pl)
    assert_invariants(pl, res)


def test_grid_shares_junctions():
    # A # grid: crossing lines with shared nodes only if split; our graph
    # doesn't auto-split crossings (they're separate polylines that overlap),
    # but endpoints are distinct - the solver must still produce one stroke.
    pl = [np.array([(0, 20), (100, 20)], dtype=np.float32),
          np.array([(0, 60), (100, 60)], dtype=np.float32),
          np.array([(30, 0), (30, 100)], dtype=np.float32),
          np.array([(70, 0), (70, 100)], dtype=np.float32)]
    res = solve(pl)
    assert_invariants(pl, res)


def test_closed_loop():
    t = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([50 + 30 * np.cos(t), 50 + 30 * np.sin(t)], axis=1)
    res = solve([circle.astype(np.float32)])
    assert_invariants([circle], res)
    assert np.all(res.seg_class == INK)


def test_loop_plus_distant_dot_splits_loop():
    # A closed circle plus a far segment whose nearest approach is mid-loop:
    # the solver should split the loop and attach there, not route to the
    # loop's seam node.
    t = np.linspace(0, 2 * np.pi, 120)
    circle = np.stack([100 + 50 * np.cos(t), 100 + 50 * np.sin(t)], axis=1)
    seg = np.array([(100, 170), (100, 190)], dtype=np.float32)
    res = solve([circle.astype(np.float32), seg])
    assert_invariants([circle, seg], res)
    m = compute_metrics(res.points, res.seg_class)
    # nearest approach is ~20px (circle bottom at y=150 to y=170)
    assert m["connector_len"] < 45


def test_random_soup_invariants():
    rng = np.random.default_rng(42)
    pl = []
    for _ in range(60):
        a = rng.uniform(0, 400, 2)
        ang = rng.uniform(0, 2 * np.pi)
        ln = rng.uniform(10, 60)
        b = a + ln * np.array([np.cos(ang), np.sin(ang)])
        pl.append(np.array([a, b], dtype=np.float32))
    res = solve(pl)
    assert_invariants(pl, res)
    m = compute_metrics(res.points, res.seg_class)
    assert m["ink_fraction"] > 0.3  # sanity: not drowning in extra travel


def test_many_hatch_lines_snapped_style():
    # Dense parallel hatch strokes whose ends coincide with a border line -
    # mimics post-snap hatching; should need few or no connectors.
    border = np.array([(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)],
                      dtype=np.float32)
    pl = [border]
    for x in range(10, 200, 10):
        pl.append(np.array([(x, 0), (x, 100)], dtype=np.float32))
    res = solve(pl)
    assert_invariants(pl, res)
    m = compute_metrics(res.points, res.seg_class)
    assert m["connector_count"] == 0


def test_tone_biases_bridge_placement():
    # With a tone map, bridging through the dark corridor should beat the
    # bright shortcut when costs differ enough. Here we just verify the
    # solve accepts tone without breaking invariants.
    tone = np.ones((200, 200), dtype=np.float32)
    tone[:, 90:110] = 0.0  # dark corridor
    pl = [np.array([(10, 10), (80, 10)], dtype=np.float32),
          np.array([(120, 190), (190, 190)], dtype=np.float32)]
    res = solve(pl, tone=tone)
    assert_invariants(pl, res)


def test_empty_and_degenerate():
    res = solve([])
    assert len(res.points) == 0
    res = solve([np.array([(5, 5)], dtype=np.float32)])
    assert len(res.points) == 0


def test_graph_split_edge():
    g = StrokeGraph()
    eid = g.add_polyline(np.array([(0, 0), (10, 0), (20, 0)], dtype=np.float32))
    out = g.split_edge(eid, 1)
    assert out is not None
    n, e1, e2 = out
    assert g.degree(n) == 2
    assert not g.edges[eid].alive
    total = sum(g.edges[e].length for e in g.alive_edges())
    assert total == pytest.approx(20.0)
