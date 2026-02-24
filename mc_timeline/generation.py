"""
Trajectory generation for SCOPE/REACH risk prediction.

Generates M1 (target-event-allowed) and M2 (target-event-forbidden) trajectories
using an SGLang engine. Generation is done without logprobs for speed; scoring
is handled separately.

Time-based stopping
-------------------
When GenerationConfig.max_time is set (together with trunc_id and
token_id_to_minutes), generation uses a stateful custom logit processor
(TimeHorizonStopProcessor) that accumulates elapsed simulated time from the
generated output tokens.  Once elapsed >= max_time the processor forces the
trunc_id token, which is also a stop token, ending the sequence.

Requirements when using time-based stopping:
  - The SGLang engine must be started with:
      disable_overlap_schedule=True
      enable_custom_logit_processor=True
  - trunc_id must NOT be logit-suppressed; the package handles this automatically
    even if trunc_id appears in config.suppressed_ids.
"""

import sglang as sgl
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from .types import GeneratedTrajectory, GenerationConfig, TrajectoryType

# ---------------------------------------------------------------------------
# Time-horizon logit processor (ported from async_pt_sglang_m1_m2.py)
# ---------------------------------------------------------------------------

class TimeHorizonStopProcessor(CustomLogitProcessor):
    """Force trunc_id once the generated timeline exceeds a simulated time limit.

    Reads the already-generated token IDs from param_dict["__req__"].output_ids
    at each decoding step and accumulates elapsed time using a caller-supplied
    token->minutes map.  When elapsed >= time_horizon the entire logit vector is
    set to -inf except for trunc_id, which is set to 0, forcing that token.

    custom_params (passed per request) must contain:
        time_horizon  : float  – maximum simulated minutes
        trunc_id      : int    – token ID to force on overflow
        time_token_map: dict[int, float] – token_id -> elapsed minutes

    Requires the SGLang engine to be started with disable_overlap_schedule=True
    to avoid race conditions on output_ids.
    """

    def __call__(self, logits, custom_param_list):
        import torch  # noqa: F401 – imported for side-effects on logits tensor

        for i, param_dict in enumerate(custom_param_list):
            req = param_dict.get("__req__")
            if req is None:
                continue

            time_horizon: float = param_dict["time_horizon"]
            trunc_id: int = param_dict["trunc_id"]
            time_token_map: dict = param_dict["time_token_map"]

            elapsed = sum(
                time_token_map.get(int(tid), 0.0) for tid in req.output_ids
            )

            if elapsed >= time_horizon:
                logits[i, :] = float("-inf")
                logits[i, trunc_id] = 0.0

        return logits


# Serialized once at module level and reused across all generate_trajectory calls.
_LOGIT_PROCESSOR_STR: str | None = None


def _get_logit_processor_str() -> str:
    global _LOGIT_PROCESSOR_STR
    if _LOGIT_PROCESSOR_STR is None:
        _LOGIT_PROCESSOR_STR = TimeHorizonStopProcessor().to_str()
    return _LOGIT_PROCESSOR_STR


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

async def generate_trajectory(
    engine: sgl.Engine,
    config: GenerationConfig,
    prompt_tokens: list[int],
    patient_idx: int,
    sample_idx: int,
    traj_type: TrajectoryType,
) -> GeneratedTrajectory:
    """Generate a single trajectory without logprobs.

    Args:
        engine: SGLang inference engine.
        config: Generation configuration.
        prompt_tokens: Tokenized patient timeline (the prompt/prefix).
        patient_idx: Index of the patient in the batch.
        sample_idx: Index of the Monte Carlo sample.
        traj_type: M1 (target event allowed) or M2 (target event forbidden).

    Returns:
        A GeneratedTrajectory with the generated token IDs and metadata.
        was_time_truncated is True iff the time horizon fired.
    """
    max_new = config.max_len - len(prompt_tokens) - 1
    use_time_stopping = config.max_time is not None and config.trunc_id is not None

    # Build logit bias from suppressed_ids.  When time-based stopping is active,
    # auto-exclude trunc_id so the logit processor can force it — even if the
    # caller included trunc_id in suppressed_ids.
    logit_bias = {
        tid: -10000
        for tid in config.suppressed_ids
        if not (use_time_stopping and tid == config.trunc_id)
    }

    if traj_type == TrajectoryType.M1:
        # M1: target event is a valid stop token
        stop_token_ids = [config.timeline_end_id, config.target_event_id]
    else:
        # M2: target event is forbidden — suppress it and stop only on timeline end
        # or suppressed tokens (which signal malformed / padding sequences).
        logit_bias[config.target_event_id] = -10000
        stop_token_ids = [config.timeline_end_id] + [
            tid
            for tid in config.suppressed_ids
            if not (use_time_stopping and tid == config.trunc_id)
        ]

    if use_time_stopping:
        stop_token_ids.append(config.trunc_id)

    sampling_params: dict = {
        "max_new_tokens": max_new,
        "temperature": config.temperature,
        "stop_token_ids": stop_token_ids,
        "logit_bias": logit_bias,
    }

    extra_kwargs: dict = {"return_logprob": False}

    if use_time_stopping:
        sampling_params["custom_params"] = {
            "time_horizon": config.max_time,
            "trunc_id": config.trunc_id,
            "time_token_map": config.token_id_to_minutes,
        }
        extra_kwargs["custom_logit_processor"] = _get_logit_processor_str()

    output = await engine.async_generate(
        input_ids=prompt_tokens,
        sampling_params=sampling_params,
        **extra_kwargs,
    )

    meta = output.get("meta_info", {})
    output_ids = list(meta.get("output_ids", output.get("output_ids", [])))

    was_time_truncated = use_time_stopping and config.trunc_id in output_ids

    return GeneratedTrajectory(
        patient_idx=patient_idx,
        sample_idx=sample_idx,
        traj_type=traj_type,
        prompt_len=len(prompt_tokens),
        output_ids=output_ids,
        has_timeline_end=config.timeline_end_id in output_ids,
        has_target_event=config.target_event_id in output_ids,
        was_time_truncated=was_time_truncated,
    )
