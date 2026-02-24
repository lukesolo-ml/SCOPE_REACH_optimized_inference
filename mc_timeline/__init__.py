"""
mc_timeline: Monte Carlo timeline completion and scoring.

A framework for generating and scoring Monte Carlo trajectory completions
from autoregressive generative models. Designed for clinical event prediction
(e.g., mortality from EHR-trained transformers) but applicable to any
autoregressive model where you want to estimate the probability of a
target event via trajectory sampling.

Estimators:
    M0 (Simple MC): Fraction of trajectories containing the target event.
    M1 (SCOPE): Expected count of the target event — sum of P(event) at each step.
    M2 (REACH): Probability the event would occur — 1 - prod(1 - P(event)).

Typical usage:

    import sglang as sgl
    from mc_timeline import GenerationConfig, generate_and_score, save_scores

    engine = sgl.Engine(model_path="...", skip_tokenizer_init=True, context_length=10000)
    config = GenerationConfig(
        max_len=10000, n_samp=20,
        target_event_id=42, timeline_end_id=43,
        suppressed_ids=[0, 1],
    )

    trajectories, results = await generate_and_score(engine, config, patient_tokens)
    save_scores(results, "scores.npz")
"""

from .io import load_scores, load_trajectories, save_scores, save_trajectories
from .scheduler import (
    aggregate_results,
    generate_and_score,
    generate_trajectories,
    score_trajectories,
)
from .types import (
    GeneratedTrajectory,
    GenerationConfig,
    PatientResults,
    ScoredTrajectory,
    TrajectoryType,
)

__all__ = [
    # Config & types
    "GenerationConfig",
    "GeneratedTrajectory",
    "ScoredTrajectory",
    "PatientResults",
    "TrajectoryType",
    # Scheduler functions
    "generate_and_score",
    "generate_trajectories",
    "score_trajectories",
    "aggregate_results",
    # IO
    "save_trajectories",
    "load_trajectories",
    "save_scores",
    "load_scores",
]
