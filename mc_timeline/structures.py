"""
Core types for Monte Carlo timeline completion and scoring.

Estimators:
- M0: Simple Monte Carlo (binary: did the target event appear?)
- M1 (SCOPE): Sum of P(target_event) at each position in trajectory
- M2 (REACH): P(target_event would have occurred) on counterfactual trajectory
           = 1 - prod(1 - P(target_event)) at each position
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrajectoryType(Enum):
    """Type of trajectory generation strategy.

    M1: Target event token is allowed (used for M0 and M1/SCOPE estimators).
    M2: Target event token is forbidden (used for M2/REACH estimator).
    """

    M1 = "m1"
    M2 = "m2"


@dataclass
class GenerationConfig:
    """Configuration for timeline trajectory generation.

    Attributes:
        max_len: Maximum total sequence length (prompt + generation).
        n_samp: Number of Monte Carlo samples per patient per trajectory type.
        target_event_id: Token ID of the target event (e.g., DSCG_Expired).
        timeline_end_id: Token ID marking the end of a timeline.
        suppressed_ids: Token IDs to suppress via logit bias (e.g., PAD, TRUNC).
            These tokens receive a large negative logit bias during generation.
        temperature: Sampling temperature. Default 1.0 for proper MC estimation.
        trunc_id: Token ID used to signal time-horizon truncation. Required for
            time-based stopping. If trunc_id appears in suppressed_ids, the
            package automatically exempts it from logit suppression so that the
            deferred logit processor can force it when the time limit is reached.
        token_id_to_minutes: Mapping from token ID to elapsed simulated minutes
            (typically the geometric mean of the token's time-bin bounds). Used
            to accumulate elapsed time during generation.
        max_time: Maximum simulated time in minutes. Generation stops (by forcing
            trunc_id) once the sum of token_id_to_minutes values in the generated
            output meets or exceeds this threshold. Requires trunc_id to be set.
        time_check_interval: Number of tokens to generate before first checking
            elapsed time.  The logit processor is a no-op for the first
            time_check_interval tokens, avoiding per-step overhead during the
            early phase of generation where the time horizon cannot possibly be
            reached.  After the first check fires, subsequent checks occur every
            time_check_interval tokens.  Default 100.
    """

    max_len: int
    n_samp: int
    target_event_id: int
    end_token_ids: set[int]
    suppressed_ids: list[int] = field(default_factory=list)
    temperature: float = 1.0
    trunc_id: int | None = None
    token_id_to_minutes: dict[int, float] = field(default_factory=dict)
    max_time: float | None = None
    time_check_interval: int = 100


@dataclass
class GeneratedTrajectory:
    """Result of a single trajectory generation (before scoring).

    Attributes:
        patient_idx: Index of the patient in the input list.
        sample_idx: Index of the Monte Carlo sample.
        traj_type: Whether this is an M1 or M2 trajectory.
        prompt_len: Number of tokens in the prompt (for offset calculations).
        output_ids: Generated token IDs (excluding prompt).
        has_timeline_end: Whether the timeline end token appeared.
        has_target_event: Whether the target event token appeared.
        was_time_truncated: Whether the time horizon fired (trunc_id was forced
            by the logit processor, or post-hoc truncation was applied). When
            True, output_ids has already been trimmed to exclude the time token
            that exceeded the horizon and any subsequent tokens. Scoring does
            not need to drop additional logprobs.
        truncation_idx: When was_time_truncated is True, the index in the
            *original* (pre-trim) output_ids at which the offending time token
            appeared. None otherwise. Useful for diagnostics.
    """

    patient_idx: int
    sample_idx: int
    traj_type: TrajectoryType
    prompt_len: int
    output_ids: list[int]
    timeline_terminating_id: int
    was_time_truncated: bool = False
    truncation_idx: int | None = None


@dataclass
class ScoredTrajectory:
    """A trajectory with its computed score.

    Attributes:
        trajectory: The underlying generated trajectory.
        score: The estimator-appropriate score:
            - For M1: sum of P(target_event) at each position
            - For M2: 1 - prod(1 - P(target_event)) at each position
    """

    trajectory: GeneratedTrajectory
    score: float


@dataclass
class PatientResults:
    """Aggregated Monte Carlo results for a single patient.

    Attributes:
        m0_samples: Binary outcomes (did target event occur?) from M1 trajectories.
        m1_samples: SCOPE scores from M1 trajectories.
        m2_samples: REACH scores from M2 trajectories.
    """

    m0_samples: list[bool] = field(default_factory=list)
    m1_samples: list[float] = field(default_factory=list)
    m2_samples: list[float] = field(default_factory=list)