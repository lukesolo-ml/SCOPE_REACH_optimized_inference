"""Trajectory diagnostics and logging utilities.

Extracted from the monolithic run_tests.py to be reusable across benchmarks
and application code.
"""

import logging
from collections import Counter
from typing import Sequence

import numpy as np

from .structures import GeneratedTrajectory, GenerationConfig, TrajectoryType


def log_trajectory_diagnostics(
    trajectories: Sequence[GeneratedTrajectory],
    config: GenerationConfig,
    phase_name: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Log and return trajectory diagnostics for a generation phase.

    Args:
        trajectories: Generated trajectories to analyze.
        config: The GenerationConfig used during generation.
        phase_name: Label for log messages.
        logger: Logger instance. If None, uses module-level logger.

    Returns:
        Dict of computed diagnostic values for programmatic use.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_total = len(trajectories)
    if n_total == 0:
        logger.info(f"DIAG [{phase_name}]: No trajectories")
        return {}

    n_time_truncated = sum(1 for t in trajectories if t.was_time_truncated)
    n_target_event = sum(
        1 for t in trajectories
        if t.timeline_terminating_id == config.target_event_id
    )
    n_discharged = sum(
        1 for t in trajectories
        if t.timeline_terminating_id is not None
        and t.timeline_terminating_id in config.end_token_ids
    )
    n_hit_max = sum(
        1 for t in trajectories
        if t.timeline_terminating_id is None and not t.was_time_truncated
    )

    lengths = np.array([len(t.output_ids) for t in trajectories])
    prompt_lens = np.array([t.prompt_len for t in trajectories])
    total_lens = prompt_lens + lengths

    logger.info(f"DIAG [{phase_name}]: {n_total} trajectories total")
    logger.info(
        f"DIAG [{phase_name}]: time_truncated={n_time_truncated}/{n_total} "
        f"({100 * n_time_truncated / n_total:.1f}%)"
    )
    logger.info(
        f"DIAG [{phase_name}]: discharged={n_discharged}/{n_total} "
        f"({100 * n_discharged / n_total:.1f}%) "
        f"[target_event={n_target_event}, other={n_discharged - n_target_event}]"
    )
    logger.info(
        f"DIAG [{phase_name}]: hit_max_tokens={n_hit_max}/{n_total} "
        f"({100 * n_hit_max / n_total:.1f}%)"
    )

    term_counts = Counter(
        t.timeline_terminating_id for t in trajectories
        if t.timeline_terminating_id is not None
    )
    if term_counts:
        logger.info(
            f"DIAG [{phase_name}]: terminating_token_id distribution: {dict(term_counts)}"
        )

    logger.info(
        f"DIAG [{phase_name}]: output_ids length — "
        f"min={lengths.min()}, median={np.median(lengths):.0f}, "
        f"max={lengths.max()}"
    )
    logger.info(
        f"DIAG [{phase_name}]: prompt_len — "
        f"min={prompt_lens.min()}, median={np.median(prompt_lens):.0f}, "
        f"max={prompt_lens.max()}"
    )
    logger.info(
        f"DIAG [{phase_name}]: total_seq_len — "
        f"min={total_lens.min()}, median={np.median(total_lens):.0f}, "
        f"max={total_lens.max()}"
    )

    # M1/M2 breakdown
    for label, ttype in [("M1", TrajectoryType.M1), ("M2", TrajectoryType.M2)]:
        subset = [t for t in trajectories if t.traj_type == ttype]
        if not subset:
            continue
        sub_lens = np.array([len(t.output_ids) for t in subset])
        sub_trunc = sum(1 for t in subset if t.was_time_truncated)
        logger.info(
            f"DIAG [{phase_name}][{label}]: n={len(subset)}, "
            f"time_trunc={sub_trunc}, "
            f"len min={sub_lens.min()} med={np.median(sub_lens):.0f} "
            f"max={sub_lens.max()}"
        )

    return {
        "n_total": n_total,
        "n_time_truncated": n_time_truncated,
        "n_target_event": n_target_event,
        "n_discharged": n_discharged,
        "n_hit_max": n_hit_max,
        "output_lengths": lengths,
        "prompt_lengths": prompt_lens,
    }
