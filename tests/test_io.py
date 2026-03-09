"""Unit tests for trajectory and score persistence in mc_timeline.io."""

import numpy as np
import pytest

from mc_timeline.io import save_scores, load_scores
from mc_timeline.structures import PatientResults


# ---------------------------------------------------------------------------
# save_scores / load_scores roundtrip
# ---------------------------------------------------------------------------


class TestScoresRoundtrip:
    """Test score save/load cycle."""

    def test_roundtrip(self, tmp_path):
        results = [
            PatientResults(m0_samples=[True, False], m1_samples=[0.5, 0.3], m2_samples=[0.8, 0.2]),
            PatientResults(m0_samples=[False, False], m1_samples=[0.1, 0.0], m2_samples=[0.9, 0.7]),
        ]

        out_path = tmp_path / "scores.npz"
        save_scores(results, out_path)
        loaded = load_scores(out_path)

        np.testing.assert_allclose(loaded["M0"], [0.5, 0.0])
        np.testing.assert_allclose(loaded["M1"], [0.4, 0.05])
        np.testing.assert_allclose(loaded["M2"], [0.5, 0.8])

    def test_empty_samples_produce_nan(self, tmp_path):
        results = [
            PatientResults(),  # all empty
        ]
        out_path = tmp_path / "scores.npz"
        save_scores(results, out_path)
        loaded = load_scores(out_path)

        assert np.isnan(loaded["M0"][0])
        assert np.isnan(loaded["M1"][0])
        assert np.isnan(loaded["M2"][0])

    def test_creates_parent_dirs(self, tmp_path):
        out_path = tmp_path / "nested" / "deep" / "scores.npz"
        results = [PatientResults(m0_samples=[True], m1_samples=[1.0], m2_samples=[0.5])]
        save_scores(results, out_path)
        assert out_path.exists()
