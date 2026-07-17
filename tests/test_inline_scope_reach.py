"""Unit tests for inline SCOPE/REACH estimation.

Covers:
  1. Pure-math REACH recursion
  2. Pure-math SCOPE truncated sum
  3. Round-trip persistence with and without inline fields
"""

import numpy as np

from quick_sco_re.io import save_trajectories, load_trajectories
from quick_sco_re.structures import GeneratedTrajectory, TrajectoryType


# ---------------------------------------------------------------------------
# 1. Pure-math REACH recursion
# ---------------------------------------------------------------------------


class TestReachMath:
    """Verify REACH = 1 - prod(1 - p_i) matches the incremental update."""

    def test_known_sequence(self):
        probs = [0.1, 0.2, 0.3, 0.05, 0.15]
        # Brute force
        expected = 1.0 - np.prod([1.0 - p for p in probs])
        # Incremental
        reach = 0.0
        for p in probs:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - expected) < 1e-12

    def test_single_prob(self):
        reach = 0.0
        p = 0.7
        reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - 0.7) < 1e-12

    def test_zero_probs(self):
        reach = 0.0
        for p in [0.0, 0.0, 0.0]:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach) < 1e-12

    def test_all_one_probs(self):
        reach = 0.0
        for p in [1.0, 1.0]:
            reach = 1.0 - (1.0 - reach) * (1.0 - p)
        assert abs(reach - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 2. Pure-math SCOPE truncated sum
# ---------------------------------------------------------------------------


class TestScopeMath:
    """Verify SCOPE = sum of p_k at pre-occurrence steps."""

    def test_no_occurrence(self):
        """All steps contribute when the token never occurs."""
        probs = [0.1, 0.2, 0.3]
        expected = sum(probs)
        scope = sum(probs)  # No truncation
        assert abs(scope - expected) < 1e-12

    def test_with_occurrence_at_step_2(self):
        """Occurrence at step 2 means steps 0, 1, 2 contribute (inclusive)."""
        # Steps: 0 (p=0.1), 1 (p=0.2), 2 (p=0.3 - occurrence step), 3 (p=0.4 - skipped)
        probs = [0.1, 0.2, 0.3, 0.4]
        occurrence_step = 2
        expected = sum(probs[: occurrence_step + 1])  # 0.1 + 0.2 + 0.3
        assert abs(expected - 0.6) < 1e-12

    def test_occurrence_at_first_step(self):
        """Token occurs immediately — only step 0's prob contributes."""
        probs = [0.5, 0.3, 0.2]
        expected = 0.5
        scope = probs[0]
        assert abs(scope - expected) < 1e-12


# ---------------------------------------------------------------------------
# 3. Round-trip persistence
# ---------------------------------------------------------------------------


class TestPersistenceRoundtrip:
    """Save and load trajectories with inline SCOPE/REACH fields."""

    def test_roundtrip_with_inline_sr(self, tmp_path):
        K = 3
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2, 3],
                timeline_terminating_id=3,
                scope_estimates=np.array([0.5, 0.3, 0.1]),
                reach_estimates=np.array([0.4, 0.25, 0.09]),
                occurred_flag=np.array([True, False, False]),
                occurred_index=np.array([2, -1, -1], dtype=np.int64),
                inline_tracked_ids=[10, 20, 30],
                inline_tracked_name="test_set",
            ),
            GeneratedTrajectory(
                patient_idx=1, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=15, output_ids=[4, 5],
                timeline_terminating_id=None,
                scope_estimates=np.array([0.7, 0.2, 0.0]),
                reach_estimates=np.array([0.6, 0.18, 0.0]),
                occurred_flag=np.array([True, True, False]),
                occurred_index=np.array([0, 1, -1], dtype=np.int64),
                inline_tracked_ids=[10, 20, 30],
                inline_tracked_name="test_set",
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert len(loaded) == 2
        for orig, ld in zip(trajs, loaded):
            assert ld.patient_idx == orig.patient_idx
            assert ld.output_ids == orig.output_ids
            np.testing.assert_array_almost_equal(ld.scope_estimates, orig.scope_estimates)
            np.testing.assert_array_almost_equal(ld.reach_estimates, orig.reach_estimates)
            np.testing.assert_array_equal(ld.occurred_flag, orig.occurred_flag)
            np.testing.assert_array_equal(ld.occurred_index, orig.occurred_index)
            assert ld.inline_tracked_ids == orig.inline_tracked_ids
            assert ld.inline_tracked_name == orig.inline_tracked_name

    def test_roundtrip_without_inline_sr(self, tmp_path):
        """Trajectories without inline SR should load with None fields."""
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2, 3],
                timeline_terminating_id=3,
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert loaded[0].scope_estimates is None
        assert loaded[0].reach_estimates is None
        assert loaded[0].occurred_flag is None
        assert loaded[0].occurred_index is None
        assert loaded[0].inline_tracked_ids is None
        assert loaded[0].inline_tracked_name is None

    def test_mixed_trajectories(self, tmp_path):
        """Mix of trajectories with and without inline SR."""
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2],
                timeline_terminating_id=2,
                scope_estimates=np.array([0.5]),
                reach_estimates=np.array([0.4]),
                occurred_flag=np.array([True]),
                occurred_index=np.array([1], dtype=np.int64),
                inline_tracked_ids=[10],
                inline_tracked_name="mixed",
            ),
            GeneratedTrajectory(
                patient_idx=1, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=10, output_ids=[3, 4],
                timeline_terminating_id=None,
                # No inline SR for this one
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)

        assert loaded[0].scope_estimates is not None
        np.testing.assert_array_almost_equal(loaded[0].scope_estimates, [0.5])
        # Second trajectory has no inline SR data — offsets are equal, so None
        assert loaded[1].scope_estimates is None
