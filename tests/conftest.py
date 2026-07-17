"""Shared fixtures for quick_sco_re unit tests."""

import pytest

from quick_sco_re.structures import (
    GeneratedTrajectory,
    GenerationConfig,
    PatientResults,
    ScoredTrajectory,
    TrajectoryType,
)


# ---------------------------------------------------------------------------
# Token ID constants (arbitrary but consistent across tests)
# ---------------------------------------------------------------------------

TARGET_EVENT_ID = 10
END_TOKEN_IDS = {11, 12}  # e.g. DSCG_Alive, DSCG_Expired
PAD_ID = 0
TRUNC_ID = 99

# Time tokens: ID -> simulated minutes
TIME_TOKEN_MAP = {
    50: 10.0,   # 10 min
    51: 60.0,   # 1 hour
    52: 360.0,  # 6 hours
    53: 1440.0, # 1 day
}


@pytest.fixture
def base_config():
    """A GenerationConfig with time-based stopping enabled."""
    return GenerationConfig(
        max_len=512,
        n_samp=5,
        target_event_id=TARGET_EVENT_ID,
        end_token_ids=END_TOKEN_IDS,
        suppressed_ids=[PAD_ID, TRUNC_ID],
        temperature=1.0,
        trunc_id=TRUNC_ID,
        token_id_to_minutes=TIME_TOKEN_MAP,
        max_time=120.0,  # 2 hours
        time_check_interval=50,
    )


@pytest.fixture
def config_no_time():
    """A GenerationConfig without time-based stopping."""
    return GenerationConfig(
        max_len=512,
        n_samp=5,
        target_event_id=TARGET_EVENT_ID,
        end_token_ids=END_TOKEN_IDS,
        suppressed_ids=[PAD_ID],
        temperature=1.0,
    )


def make_trajectory(
    patient_idx=0,
    sample_idx=0,
    traj_type=TrajectoryType.M1,
    prompt_len=10,
    output_ids=None,
    timeline_terminating_id=None,
    was_time_truncated=False,
    truncation_idx=None,
    scope_estimates=None,
    reach_estimates=None,
    occurred_flag=None,
    occurred_index=None,
    inline_tracked_ids=None,
    inline_tracked_name=None,
):
    """Helper to build a GeneratedTrajectory with sensible defaults."""
    return GeneratedTrajectory(
        patient_idx=patient_idx,
        sample_idx=sample_idx,
        traj_type=traj_type,
        prompt_len=prompt_len,
        output_ids=output_ids if output_ids is not None else [],
        timeline_terminating_id=timeline_terminating_id,
        was_time_truncated=was_time_truncated,
        truncation_idx=truncation_idx,
        scope_estimates=scope_estimates,
        reach_estimates=reach_estimates,
        occurred_flag=occurred_flag,
        occurred_index=occurred_index,
        inline_tracked_ids=inline_tracked_ids,
        inline_tracked_name=inline_tracked_name,
    )
