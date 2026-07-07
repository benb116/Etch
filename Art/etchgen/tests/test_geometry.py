import numpy as np
import pytest

from Art.etchgen.geometry import (
    chaikin, dedupe, densify, polyline_length, rdp, resample,
)


def test_dedupe_removes_repeats():
    pts = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0)]
    out = dedupe(pts)
    assert len(out) == 3


def test_polyline_length():
    assert polyline_length([(0, 0), (3, 4)]) == pytest.approx(5.0)
    assert polyline_length([(0, 0)]) == 0.0


def test_resample_spacing():
    pts = [(0, 0), (100, 0)]
    out = resample(pts, 10)
    assert np.allclose(out[0], (0, 0)) and np.allclose(out[-1], (100, 0))
    gaps = np.hypot(*np.diff(out, axis=0).T)
    assert np.all(gaps <= 10.0 + 1e-4)
    assert polyline_length(out) == pytest.approx(100.0)


def test_densify_preserves_vertices():
    pts = np.array([(0, 0), (10, 0), (10, 5)], dtype=np.float32)
    out = densify(pts, 3.0)
    for p in pts:
        assert np.min(np.hypot(*(out - p).T)) < 1e-5
    gaps = np.hypot(*np.diff(out, axis=0).T)
    assert np.all(gaps <= 3.0 + 1e-4)
    assert polyline_length(out) == pytest.approx(15.0)


def test_rdp_collapses_collinear():
    pts = [(x, 0) for x in range(50)]
    out = rdp(pts, 0.1)
    assert len(out) == 2


def test_rdp_keeps_corners():
    pts = [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10)]
    out = rdp(pts, 0.5)
    assert len(out) == 3
    assert (10, 0) in [tuple(p) for p in out]


def test_chaikin_keeps_endpoints():
    pts = [(0, 0), (10, 0), (10, 10)]
    out = chaikin(pts, 2)
    assert np.allclose(out[0], (0, 0))
    assert np.allclose(out[-1], (10, 10))
    assert len(out) > len(pts)
