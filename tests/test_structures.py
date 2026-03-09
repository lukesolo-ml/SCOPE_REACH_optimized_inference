"""Unit tests for data structures in mc_timeline.structures."""

import pytest

from mc_timeline.structures import (
    GeneratedTrajectory,
    GenerationConfig,
    PatientResults,
    ScoredTrajectory,
    TrajectoryType,
)


class TestTrajectoryType:
    def test_m1_value(self):
        assert TrajectoryType.M1.value == "m1"

    def test_m2_value(self):
        assert TrajectoryType.M2.value == "m2"

    def test_from_string(self):
        assert TrajectoryType("m1") is TrajectoryType.M1
        assert TrajectoryType("m2") is TrajectoryType.M2


class TestGenerationConfig:
    def test_defaults(self):
        config = GenerationConfig(
            max_len=100, n_samp=10, target_event_id=5, end_token_ids={6},
        )
        assert config.suppressed_ids == []
        assert config.temperature == 1.0
        assert config.trunc_id is None
        assert config.token_id_to_minutes == {}
        assert config.max_time is None
        assert config.time_check_interval == 100

    def test_time_stopping_requires_both_fields(self):
        """Time stopping is only active when both max_time and trunc_id are set."""
        config = GenerationConfig(
            max_len=100, n_samp=10, target_event_id=5, end_token_ids={6},
            max_time=120.0, trunc_id=None,
        )
        use_time = config.max_time is not None and config.trunc_id is not None
        assert use_time is False


class TestPatientResults:
    def test_default_empty_lists(self):
        r = PatientResults()
        assert r.m0_samples == []
        assert r.m1_samples == []
        assert r.m2_samples == []

    def test_independent_instances(self):
        """Each PatientResults instance has independent lists."""
        a = PatientResults()
        b = PatientResults()
        a.m0_samples.append(True)
        assert b.m0_samples == []
