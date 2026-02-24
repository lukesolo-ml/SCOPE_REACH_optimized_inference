"""
Trajectory scoring via prefill-only logprob extraction.

Scores generated trajectories by running a forward pass over the full sequence
(prompt + generated tokens) and extracting P(target_event) at each generated
position. This is the "second pass" of the two-pass approach.
"""

import numpy as np
import sglang as sgl

from .types import GeneratedTrajectory, GenerationConfig, ScoredTrajectory, TrajectoryType


def _extract_token_logprobs(
    output: dict,
    target_token_id: int,
    source: str = "output",
    skip: int = 0,
) -> list[float]:
    """Extract logprobs for a specific token ID from SGLang output.

    Args:
        output: SGLang generation output dict.
        target_token_id: Token ID to extract logprobs for.
        source: "output" for output_token_ids_logprobs,
                "input" for input_token_ids_logprobs.
        skip: Number of initial position entries to skip.

    Returns:
        List of log-probabilities for the target token at each position.
    """
    meta = output.get("meta_info", {})
    key = (
        "output_token_ids_logprobs"
        if source == "output"
        else "input_token_ids_logprobs"
    )
    token_ids_logprobs = meta.get(key, [])

    logprobs = []
    for position_entry in token_ids_logprobs[skip:]:
        if position_entry is None:
            continue
        for logprob, token_id, _ in position_entry:
            if token_id == target_token_id:
                logprobs.append(logprob)
                break

    return logprobs


async def score_trajectory(
    engine: sgl.Engine,
    config: GenerationConfig,
    traj: GeneratedTrajectory,
    prompt_tokens: list[int],
) -> ScoredTrajectory:
    """Score a trajectory by extracting P(target_event) at each position.

    Runs a prefill-only forward pass over [prompt + generated_tokens] and
    extracts the probability assigned to the target event token at each
    generated position.

    For M1 trajectories: score = sum of P(target_event) at each step (SCOPE).
    For M2 trajectories: score = 1 - prod(1 - P(target_event)) (REACH).

    Args:
        engine: SGLang inference engine.
        config: Generation configuration (used for token IDs, max_len).
        traj: The generated trajectory to score.
        prompt_tokens: The original prompt tokens for this patient.

    Returns:
        A ScoredTrajectory with the computed score.
    """
    if not traj.output_ids:
        return ScoredTrajectory(trajectory=traj, score=0.0)

    # Determine which output tokens to include in the scoring sequence
    if traj.traj_type == TrajectoryType.M1 and traj.has_target_event:
        stop_idx = traj.output_ids.index(config.target_event_id)
        scoring_ids = traj.output_ids[: stop_idx + 1]
    else:
        scoring_ids = traj.output_ids

    # Truncate to fit within context length
    max_scoring_len = config.max_len - len(prompt_tokens) - 10
    if len(scoring_ids) > max_scoring_len:
        scoring_ids = scoring_ids[:max_scoring_len]

    if not scoring_ids:
        return ScoredTrajectory(trajectory=traj, score=0.0)

    full_sequence = prompt_tokens + scoring_ids
    prompt_len = len(prompt_tokens)

    score_output = await engine.async_generate(
        input_ids=full_sequence,
        sampling_params={
            "max_new_tokens": 1,
            "temperature": 1.0,
        },
        return_logprob=True,
        logprob_start_len=prompt_len - 1,
        token_ids_logprob=[config.target_event_id],
    )

    # Collect logprobs from both input and output positions
    dscg_logprobs = _extract_token_logprobs(
        score_output, config.target_event_id, source="input", skip=2
    )
    output_logprobs = _extract_token_logprobs(
        score_output, config.target_event_id, source="output"
    )
    if output_logprobs:
        dscg_logprobs.append(output_logprobs[0])

    if not dscg_logprobs:
        return ScoredTrajectory(trajectory=traj, score=0.0)

    # When the time horizon fired, the last two generated tokens are the time
    # token that pushed elapsed time over the limit and the forced trunc_id.
    # Neither should contribute probability mass to the estimators.
    if traj.was_time_truncated:
        dscg_logprobs = dscg_logprobs[:-2] if len(dscg_logprobs) >= 2 else []

    if not dscg_logprobs:
        return ScoredTrajectory(trajectory=traj, score=0.0)

    probs = np.clip(np.exp(dscg_logprobs), 0.0, 1.0)

    if traj.traj_type == TrajectoryType.M1:
        score = float(np.sum(probs))
    else:
        score = float(1.0 - np.prod(1.0 - probs))

    return ScoredTrajectory(trajectory=traj, score=score)
