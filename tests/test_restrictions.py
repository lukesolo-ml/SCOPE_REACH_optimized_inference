"""Silence-restriction first-token biases."""

import math

import numpy as np
import pytest

from quick_sco_re.restrictions import FORBID, allowed_mask, silence_restriction_bias

# token id -> (lo, hi] minutes, cocoa's bins
BOUNDS = {50: (5, 15), 51: (15, 60), 52: (60, 120), 53: (120, 360), 54: (360, 720), 55: (720, 1440)}
VOCAB = list(range(60))


def reference(x, mode, min_minutes=None):
    """build_silence_bias_fn from the cocoa runner, which the helper replaces."""
    if x is None or not np.isfinite(x):
        return None
    m = float(np.floor(x + 1e-9))
    mm = min(lo for lo, _ in BOUNDS.values()) if min_minutes is None else min_minutes
    if m < mm:
        return None
    b = {t: -10000.0 for t in VOCAB if t not in BOUNDS}
    for tid, (lo, hi) in BOUNDS.items():
        if hi < m:
            b[tid] = -10000.0
        elif mode == "prop" and lo < m:
            b[tid] = float(np.log((hi - m + 1) / (hi - lo)))
    return b


@pytest.mark.parametrize("mode", ["hard", "prop"])
def test_matches_reference_runner(mode):
    for x in [None, float("nan"), 0, 4.9, 5, 14.99, 15, 15.5, 59.999, 60, 61, 119.2, 700, 1439, 1440, 5000]:
        assert silence_restriction_bias(x, BOUNDS, VOCAB, mode=mode) == reference(x, mode)


def test_hard_allows_only_bins_that_can_hold_the_gap():
    b = silence_restriction_bias(100.7, BOUNDS, VOCAB)
    mask = allowed_mask(b, len(VOCAB))
    assert sorted(np.flatnonzero(mask)) == [52, 53, 54, 55]          # hi >= 100
    assert all(b[t] == FORBID for t in VOCAB if t not in BOUNDS)      # every non-time token forbidden


def test_prop_scales_the_containing_bin():
    b = silence_restriction_bias(100, BOUNDS, VOCAB, mode="prop")
    assert b[52] == pytest.approx(math.log((120 - 100 + 1) / (120 - 60)))
    assert 53 not in b                                                 # bins wholly above keep their probability


def test_no_restriction_below_min_or_unknown():
    assert silence_restriction_bias(4.9, BOUNDS, VOCAB) is None
    assert silence_restriction_bias(20, BOUNDS, VOCAB, min_minutes=30) is None
    assert silence_restriction_bias(float("inf"), BOUNDS, VOCAB) is None
    assert allowed_mask(None, 5).all()


def test_bad_mode():
    with pytest.raises(ValueError):
        silence_restriction_bias(100, BOUNDS, VOCAB, mode="soft")


def test_open_ended_top_bin():
    """cocoa's last spacer (e.g. TIME//6mt+) has no upper edge: always allowed, never scaled."""
    bounds = {10: (5, 15), 11: (15, 60), 12: (60, float("inf"))}
    for mode in ("hard", "prop"):
        b = silence_restriction_bias(600.0, bounds, range(14), mode=mode)
        assert b[10] == FORBID and b[11] == FORBID and 12 not in b
        assert b[0] == FORBID and b[13] == FORBID
