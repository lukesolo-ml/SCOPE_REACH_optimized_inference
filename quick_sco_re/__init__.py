"""
quick_sco_re: Quick SCOPE and REACH estimation.

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

    from quick_sco_re import GenerationConfig, create_engine, generate_and_score, save_scores

    engine = create_engine(model_path="...", max_len=10000)   # needs the [sglang] extra
    config = GenerationConfig(
        max_len=10000, n_samp=20,
        target_event_id=42, 
        suppressed_ids=[0, 1],
    )

    trajectories, results = await generate_and_score(engine, config, patient_tokens)
    save_scores(results, "scores.npz")
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("quick_sco_re")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0+unknown"

from .diagnostics import check_engine_parity, check_rope_config
from .generation import generate_m2_from_m1_trajectory
from .hf_scoring import generation_segments, score_full_vocab
from .models import export_backbone, is_wrapped, require_servable
from .io import load_scores, load_trajectories, save_scores, save_trajectories
from .scheduler import (
    aggregate_results,
    generate_and_score,
    generate_trajectories,
    generate_m2_from_m1_trajectories,
    score_trajectories,
    create_engine
)
from .restrictions import allowed_mask, silence_restriction_bias
from .scoring import score_trajectory
from .structures import (
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
    "generate_m2_from_m1_trajectory",
    "generate_m2_from_m1_trajectories",
    "score_trajectory",
    "score_trajectories",
    "aggregate_results",
    "create_engine",
    # IO
    "save_trajectories",
    "load_trajectories",
    "save_scores",
    "load_scores",
    # First-token restrictions
    "silence_restriction_bias",
    "allowed_mask",
    # Full-vocabulary HF scoring (needs the [hf] extra)
    "generation_segments",
    "score_full_vocab",
    # Model directories (cotorra models with secondary heads)
    "export_backbone",
    "is_wrapped",
    "require_servable",
    # Checks
    "check_engine_parity",
    "check_rope_config",
]
