"""Unit tests for scheduler aggregation logic."""

import pytest

from quick_sco_re.structures import (
    GeneratedTrajectory,
    ScoredTrajectory,
    TrajectoryType,
)
from quick_sco_re.scheduler import aggregate_results
from .conftest import make_trajectory, TARGET_EVENT_ID


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------


class TestAggregateResults:
    """Tests for scored trajectory aggregation into PatientResults."""

    def test_basic_aggregation(self):
        """M1 trajectories produce m0 + m1 samples; M2 produces m2 samples."""
        scored = [
            # Patient 0, sample 0
            ScoredTrajectory(
                trajectory=make_trajectory(
                    patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M1,
                    output_ids=[1, 2, TARGET_EVENT_ID],
                    timeline_terminating_id=TARGET_EVENT_ID,
                ),
                score=1.5,
            ),
            ScoredTrajectory(
                trajectory=make_trajectory(
                    patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M2,
                    output_ids=[1, 2, 3],
                    timeline_terminating_id=11,
                ),
                score=0.7,
            ),
        ]

        results = aggregate_results(scored, num_patients=1, target_event_id=TARGET_EVENT_ID)
        assert len(results) == 1
        r = results[0]

        assert r.m0_samples == [True]   # target event was terminating token
        assert r.m1_samples == [1.5]
        assert r.m2_samples == [0.7]

    def test_m0_false_when_no_target_event(self):
        """M0 is False when trajectory did not terminate at target event."""
        scored = [
            ScoredTrajectory(
                trajectory=make_trajectory(
                    patient_idx=0, traj_type=TrajectoryType.M1,
                    output_ids=[1, 2, 11],
                    timeline_terminating_id=11,  # not the target
                ),
                score=0.3,
            ),
        ]
        results = aggregate_results(scored, num_patients=1, target_event_id=TARGET_EVENT_ID)
        assert results[0].m0_samples == [False]

    def test_multiple_patients_and_samples(self):
        """Aggregation correctly bins by patient_idx and traj_type."""
        scored = []
        for pid in range(3):
            for sid in range(2):
                scored.append(
                    ScoredTrajectory(
                        trajectory=make_trajectory(
                            patient_idx=pid, sample_idx=sid,
                            traj_type=TrajectoryType.M1,
                            timeline_terminating_id=None,
                        ),
                        score=float(pid + sid),
                    )
                )
                scored.append(
                    ScoredTrajectory(
                        trajectory=make_trajectory(
                            patient_idx=pid, sample_idx=sid,
                            traj_type=TrajectoryType.M2,
                        ),
                        score=float(pid * 10 + sid),
                    )
                )

        results = aggregate_results(scored, num_patients=3, target_event_id=TARGET_EVENT_ID)
        assert len(results) == 3

        # Patient 0: m1 scores = [0.0, 1.0], m2 scores = [0.0, 1.0]
        assert results[0].m1_samples == [0.0, 1.0]
        assert results[0].m2_samples == [0.0, 1.0]

        # Patient 2: m1 scores = [2.0, 3.0], m2 scores = [20.0, 21.0]
        assert results[2].m1_samples == [2.0, 3.0]
        assert results[2].m2_samples == [20.0, 21.0]

    def test_empty_trajectories(self):
        results = aggregate_results([], num_patients=2, target_event_id=TARGET_EVENT_ID)
        assert len(results) == 2
        for r in results:
            assert r.m0_samples == []
            assert r.m1_samples == []
            assert r.m2_samples == []

    def test_m0_none_terminating_id(self):
        """M0 is False when timeline_terminating_id is None (hit max tokens)."""
        scored = [
            ScoredTrajectory(
                trajectory=make_trajectory(
                    patient_idx=0, traj_type=TrajectoryType.M1,
                    timeline_terminating_id=None,
                ),
                score=0.5,
            ),
        ]
        results = aggregate_results(scored, num_patients=1, target_event_id=TARGET_EVENT_ID)
        assert results[0].m0_samples == [False]
