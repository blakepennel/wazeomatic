import numpy as np

from wazeomatic.hood import HoodEstimator


def _frames(n, h=180, w=320, dash_frac=0.0, still=False, seed=0):
    """Synthetic drive: scrolling texture (road) with an optional static band at the bottom."""
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 255, size=(h * 3, w), dtype=np.uint8)
    dash = rng.integers(60, 120, size=(h, w), dtype=np.uint8)
    cut = int(h * (1 - dash_frac))
    for i in range(n):
        off = 0 if still else (i * 7) % (h * 2)
        img = texture[off:off + h].copy()
        if dash_frac > 0:
            img[cut:] = dash[cut:]
        yield np.repeat(img[:, :, None], 3, axis=2)


def test_estimator_finds_a_static_dashboard_band():
    est = HoodEstimator(min_frac=0.10)
    for fr in _frames(60, dash_frac=0.25):
        frac = est.update(fr)
    assert 0.20 <= frac <= 0.30


def test_estimator_never_drops_below_minimum_or_above_maximum():
    est = HoodEstimator(min_frac=0.10, max_frac=0.40)
    for fr in _frames(60, dash_frac=0.0):
        frac = est.update(fr)
    assert frac == 0.10
    est2 = HoodEstimator(min_frac=0.10, max_frac=0.30)
    for fr in _frames(60, dash_frac=0.4):
        frac2 = est2.update(fr)
    assert frac2 == 0.30


def test_estimator_waits_for_min_frames_and_ignores_standstill():
    est = HoodEstimator(min_frac=0.10, min_frames=30)
    for i, fr in enumerate(_frames(20, dash_frac=0.25)):
        assert est.update(fr) == 0.10
    # a still car (no scene motion) must not grow the zone
    est3 = HoodEstimator(min_frac=0.10)
    for fr in _frames(60, dash_frac=0.25, still=True):
        frac = est3.update(fr)
    assert frac == 0.10
