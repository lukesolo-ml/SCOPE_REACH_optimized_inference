"""Unit tests for trajectory and score persistence in mc_timeline.io."""

import numpy as np
import pytest

from mc_timeline.io import save_scores, load_scores, save_trajectories, load_trajectories
from mc_timeline.structures import GeneratedTrajectory, GenerationConfig, PatientResults, TrajectoryType


# ---------------------------------------------------------------------------
# save_trajectories / load_trajectories roundtrip
# ---------------------------------------------------------------------------


class TestTrajectoryRoundtrip:
    """Test trajectory save/load cycle."""

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=10, output_ids=[1, 2, 3, 4, 5],
                timeline_terminating_id=5, was_time_truncated=False,
                truncation_idx=None,
            ),
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=10, output_ids=[1, 2, 3],
                timeline_terminating_id=None, was_time_truncated=True,
                truncation_idx=3,
            ),
            GeneratedTrajectory(
                patient_idx=1, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=15, output_ids=[10, 20],
                timeline_terminating_id=20, was_time_truncated=False,
                truncation_idx=None,
            ),
        ]

        save_trajectories(trajs, tmp_path)
        loaded, config = load_trajectories(tmp_path)

        assert config is None
        assert len(loaded) == len(trajs)

        for orig, ld in zip(trajs, loaded):
            assert ld.patient_idx == orig.patient_idx
            assert ld.sample_idx == orig.sample_idx
            assert ld.traj_type == orig.traj_type
            assert ld.prompt_len == orig.prompt_len
            assert ld.output_ids == orig.output_ids
            assert ld.timeline_terminating_id == orig.timeline_terminating_id
            assert ld.was_time_truncated == orig.was_time_truncated
            assert ld.truncation_idx == orig.truncation_idx

    def test_roundtrip_with_config(self, tmp_path):
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                prompt_len=5, output_ids=[1, 2],
                timeline_terminating_id=2,
            ),
        ]
        config = GenerationConfig(
            max_len=100, n_samp=10, target_event_id=42,
            end_token_ids={43, 44},
        )
        save_trajectories(trajs, tmp_path, config=config)
        loaded, loaded_config = load_trajectories(tmp_path)

        assert loaded_config is not None
        assert loaded_config.max_len == 100
        assert loaded_config.target_event_id == 42

    def test_empty_output_ids(self, tmp_path):
        trajs = [
            GeneratedTrajectory(
                patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M2,
                prompt_len=5, output_ids=[],
                timeline_terminating_id=None,
            ),
        ]
        save_trajectories(trajs, tmp_path)
        loaded, _ = load_trajectories(tmp_path)
        assert loaded[0].output_ids == []


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
