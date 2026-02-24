"""
Interleaved scheduler for SCOPE/REACH timeline generation and scoring.

Orchestrates two-phase processing:
  Phase 1: Generate M1 and M2 trajectories interleaved by patient for
           radix cache locality.
  Phase 2: Score trajectories via prefill-only logprob extraction.

The scheduler supports three modes of operation:
  - generate_only: Generate trajectories and return/save them without scoring.
  - generate_and_score: Generate then immediately score (keeps radix cache hot).
  - score_only: Score previously-saved trajectories loaded from disk.
"""

import asyncio
import logging
import time
from typing import Sequence

import sglang as sgl

from .generation import generate_trajectory
from .scoring import score_trajectory
from .types import (
    GeneratedTrajectory,
    GenerationConfig,
    PatientResults,
    ScoredTrajectory,
    TrajectoryType,
)

logger = logging.getLogger(__name__)


async def generate_trajectories(
    engine: sgl.Engine,
    config: GenerationConfig,
    patient_tokens: Sequence[list[int]],
) -> list[GeneratedTrajectory]:
    """Generate all M1 and M2 trajectories for a batch of patients.

    Trajectories are interleaved by patient (M1_p0_s0, M2_p0_s0, M1_p0_s1,
    M2_p0_s1, ..., M1_p1_s0, ...) to maximize SGLang radix cache hits on
    shared patient prefixes.

    Args:
        engine: SGLang inference engine.
        config: Generation configuration.
        patient_tokens: List of tokenized patient timelines (one per patient).

    Returns:
        List of all generated trajectories.
    """
    gen_tasks = []
    for patient_idx, tokens in enumerate(patient_tokens):
        for sample_idx in range(config.n_samp):
            gen_tasks.append(
                generate_trajectory(
                    engine, config, tokens, patient_idx, sample_idx, TrajectoryType.M1
                )
            )
            gen_tasks.append(
                generate_trajectory(
                    engine, config, tokens, patient_idx, sample_idx, TrajectoryType.M2
                )
            )

    logger.info(f"Generating {len(gen_tasks)} trajectories...")
    start = time.time()
    trajectories = await asyncio.gather(*gen_tasks)
    elapsed = time.time() - start
    logger.info(f"Generation completed in {elapsed:.2f}s")

    return list(trajectories)


async def score_trajectories(
    engine: sgl.Engine,
    config: GenerationConfig,
    trajectories: Sequence[GeneratedTrajectory],
    patient_tokens: Sequence[list[int]],
) -> list[ScoredTrajectory]:
    """Score a collection of trajectories by extracting logprobs.

    Args:
        engine: SGLang inference engine.
        config: Generation configuration.
        trajectories: Previously generated trajectories.
        patient_tokens: Original patient prompts (indexed by patient_idx).

    Returns:
        List of scored trajectories.
    """
    score_tasks = [
        score_trajectory(engine, config, traj, patient_tokens[traj.patient_idx])
        for traj in trajectories
    ]

    logger.info(f"Scoring {len(score_tasks)} trajectories...")
    start = time.time()
    scored = await asyncio.gather(*score_tasks)
    elapsed = time.time() - start
    logger.info(f"Scoring completed in {elapsed:.2f}s")

    return list(scored)


def aggregate_results(
    scored_trajectories: Sequence[ScoredTrajectory],
    num_patients: int,
) -> list[PatientResults]:
    """Aggregate scored trajectories into per-patient results.

    Args:
        scored_trajectories: All scored trajectories.
        num_patients: Total number of patients.

    Returns:
        List of PatientResults, one per patient (indexed by patient_idx).
    """
    results = {i: PatientResults() for i in range(num_patients)}

    for st in scored_trajectories:
        traj = st.trajectory
        if traj.traj_type == TrajectoryType.M1:
            results[traj.patient_idx].m0_samples.append(traj.has_target_event)
            results[traj.patient_idx].m1_samples.append(st.score)
        else:
            results[traj.patient_idx].m2_samples.append(st.score)

    return [results[i] for i in range(num_patients)]


async def generate_and_score(
    engine: sgl.Engine,
    config: GenerationConfig,
    patient_tokens: Sequence[list[int]],
) -> tuple[list[GeneratedTrajectory], list[PatientResults]]:
    """Generate and immediately score all trajectories.

    This is the recommended mode when you want scores and the engine is
    already running — scoring immediately after generation keeps the radix
    cache hot for the prefill scoring pass.

    Args:
        engine: SGLang inference engine.
        config: Generation configuration.
        patient_tokens: List of tokenized patient timelines.

    Returns:
        Tuple of (all generated trajectories, per-patient aggregated results).
    """
    total_start = time.time()

    trajectories = await generate_trajectories(engine, config, patient_tokens)
    scored = await score_trajectories(engine, config, trajectories, patient_tokens)
    results = aggregate_results(scored, num_patients=len(patient_tokens))

    total_time = time.time() - total_start
    logger.info(f"Total generate-and-score time: {total_time:.2f}s")

    return trajectories, results
