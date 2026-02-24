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
            time-based stopping. The SGLang engine must be started with
            disable_overlap_schedule=True and enable_custom_logit_processor=True
            when this is set. If trunc_id appears in suppressed_ids, the package
            automatically exempts it from logit suppression so the logit processor
            can force it when the time limit is reached.
        token_id_to_minutes: Mapping from token ID to elapsed simulated minutes
            (typically the geometric mean of the token's time-bin bounds). Used
            by the time-horizon logit processor to accumulate elapsed time.
        max_time: Maximum simulated time in minutes. Generation stops (by forcing
            trunc_id) once the sum of token_id_to_minutes values in the generated
            output meets or exceeds this threshold. Requires trunc_id to be set.
    """

    max_len: int
    n_samp: int
    target_event_id: int
    timeline_end_id: int
    suppressed_ids: list[int] = field(default_factory=list)
    temperature: float = 1.0
    trunc_id: int | None = None
    token_id_to_minutes: dict[int, float] = field(default_factory=dict)
    max_time: float | None = None


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
            by the logit processor). When True, the last two generated tokens
            (the time token that pushed elapsed time over the limit and the forced
            trunc_id itself) must not contribute their probabilities to the
            estimators during scoring.
    """

    patient_idx: int
    sample_idx: int
    traj_type: TrajectoryType
    prompt_len: int
    output_ids: list[int]
    has_timeline_end: bool
    has_target_event: bool
    was_time_truncated: bool = False


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
