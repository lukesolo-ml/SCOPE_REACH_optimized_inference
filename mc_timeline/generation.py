"""
Trajectory generation for SCOPE/REACH risk prediction.

Generates M1 (target-event-allowed) and M2 (target-event-forbidden) trajectories
using an SGLang engine. Generation is done without logprobs for speed; scoring
is handled separately.

TODO: Try to implement ability for simulatenous generation and scoring

Time-based stopping
-------------------
When GenerationConfig.max_time is set (together with trunc_id and
token_id_to_minutes), generation uses a two-layer approach:

1. **Deferred logit processor** A lightweight custom logit processor that is
   a complete no-op for the first ``time_check_interval`` tokens. After that it
   checks elapsed simulated time every ``time_check_interval`` tokens. When it
   detects that the horizon has been exceeded it forces trunc_id, terminating
   generation. This avoids the throughput penalty of checking every step.

2. **Post-hoc exact truncation**  Because the processor only checks
   periodically, the trajectory may overshoot the horizon by up to
   ``time_check_interval`` tokens. After generation, we walk the output tokens
   to find the *exact* position where elapsed time first met or exceeded
   max_time and trim output_ids to end just before that time token. This gives
   exact time-horizon semantics at near-baseline throughput.

Requirements when using time-based stopping:
  - The SGLang engine must be started with:
      disable_overlap_schedule=True
      enable_custom_logit_processor=True
      NOTE: THIS IS CRITICAL
  - trunc_id must NOT be logit-suppressed; the package handles this
    automatically even if trunc_id appears in config.suppressed_ids.
      NOTE: THIS IS ALSO IMPORTANT

Chunked post-hoc generation
----------------------------
An alternative to the deferred logit processor approach. Instead of attaching
a custom logit processor (which imposes ~2x throughput overhead from SGLang's
per-step dispatch), generation proceeds in fixed-size token chunks with
time-horizon checks between rounds in Python.

Advantages:
  - No custom logit processor → no ``__call__`` dispatch overhead
  - No ``disable_overlap_schedule`` needed → overlap scheduler runs freely
  - No ``enable_custom_logit_processor`` needed → vanilla engine codepath
  - SGLang's radix cache makes resubmission prefix cost minimal
  - ``chunk_size`` is the tuning knob: larger = fewer round trips but more
    overshoot tokens; smaller = tighter truncation but more rounds

The chunked approach uses the same post-hoc exact truncation as the logit
processor path, so final time-horizon semantics are identical.
"""

import sglang as sgl
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from .types import GeneratedTrajectory, GenerationConfig, TrajectoryType

# ---------------------------------------------------------------------------
# Deferred time-horizon logit processor
# ---------------------------------------------------------------------------


class DeferredTimeHorizonProcessor(CustomLogitProcessor):
    """Force trunc_id once generated timeline exceeds a simulated time limit.

    Uses two optimizations to minimize overhead:

    1. **Deferred checking**: The processor is a no-op until at least
       ``check_interval`` tokens have been generated, then checks every
       ``check_interval`` tokens. On non-check steps the only work is two
       integer comparisons.

    2. **Incremental accumulation**: Instead of re-summing the entire
       output_ids sequence on every check, the processor maintains a running
       elapsed-time total and a cursor in custom_params (mutated in place).
       Each check only sums tokens generated since the last check, making
       total work O(n) across the full sequence instead of O(n²).

    custom_params (passed per request) must contain:
        time_horizon   : float            -maximum simulated minutes
        trunc_id       : int              -token ID to force on overflow
        time_token_map : dict[int, float] -token_id -> elapsed minutes
        check_interval : int              -tokens between checks (default 100)

    The processor adds two keys to custom_params on first use:
        _elapsed        : float - running total of elapsed simulated minutes
        _cursor         : int   - index into output_ids up to which we've summed

    Requires disable_overlap_schedule=True on the SGLang engine.
    """

    def __call__(self, logits, custom_param_list):
        for i, param_dict in enumerate(custom_param_list):
            req = param_dict.get("__req__")
            if req is None:
                continue

            check_interval: int = param_dict.get("check_interval", 100)
            n_generated = len(req.output_ids)

            # No-op until we've generated enough tokens to bother checking.
            # After the first eligible check, re-check every check_interval.
            if n_generated < check_interval:
                continue
            if n_generated % check_interval != 0:
                continue

            time_token_map: dict = param_dict["time_token_map"]

            # Incremental accumulation: only sum tokens since last check
            cursor: int = param_dict.get("_cursor", 0)
            elapsed: float = param_dict.get("_elapsed", 0.0)

            for j in range(cursor, n_generated):
                elapsed += time_token_map.get(int(req.output_ids[j]), 0.0)

            # Store running state back into custom_params for next check
            param_dict["_cursor"] = n_generated
            param_dict["_elapsed"] = elapsed

            if elapsed >= param_dict["time_horizon"]:
                trunc_id: int = param_dict["trunc_id"]
                logits[i, :] = float("-inf")
                logits[i, trunc_id] = 0.0

        return logits



# Serialized once at module level and reused across all generate_trajectory calls.
_LOGIT_PROCESSOR_STR: str | None = None


def _get_logit_processor_str() -> str:
    global _LOGIT_PROCESSOR_STR
    if _LOGIT_PROCESSOR_STR is None:
        _LOGIT_PROCESSOR_STR = DeferredTimeHorizonProcessor().to_str()
    return _LOGIT_PROCESSOR_STR


# ---------------------------------------------------------------------------
# Post-hoc exact time truncation
# ---------------------------------------------------------------------------


def _apply_time_truncation(
    output_ids: list[int],
    token_id_to_minutes: dict[int, float],
    max_time: float,
) -> tuple[list[int], bool, int | None]:
    """Walk output_ids and trim to the exact time horizon.

    Finds the first time token whose cumulative contribution causes elapsed
    time to meet or exceed max_time, then truncates output_ids to end just
    *before* that token.

    Args:
        output_ids: Raw generated token IDs.
        token_id_to_minutes: Mapping from token ID to elapsed minutes.
        max_time: Maximum simulated time in minutes.

    Returns:
        (trimmed_ids, was_truncated, truncation_idx)
        - trimmed_ids: output_ids truncated at the horizon (or unchanged).
        - was_truncated: True if truncation occurred.
        - truncation_idx: Index of the offending time token in the original
          output_ids, or None if no truncation.
    """
    elapsed = 0.0
    for idx, tid in enumerate(output_ids):
        minutes = token_id_to_minutes.get(tid, 0.0)
        if minutes > 0.0:
            elapsed += minutes
            if elapsed >= max_time:
                # Trim to just before this time token
                return output_ids[:idx], True, idx

    return output_ids, False, None


# ---------------------------------------------------------------------------
# Helpers for computing elapsed simulated time
# ---------------------------------------------------------------------------


def _compute_elapsed_minutes(
    output_ids: list[int],
    token_id_to_minutes: dict[int, float],
) -> float:
    """Sum the simulated elapsed minutes across all tokens in output_ids."""
    elapsed = 0.0
    for tid in output_ids:
        elapsed += token_id_to_minutes.get(tid, 0.0)
    return elapsed


# ---------------------------------------------------------------------------
# Core generation function (logit processor path)
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
        When time-based stopping is active, output_ids are post-hoc trimmed
        to the exact time horizon boundary.
    """
    max_new = config.max_len - len(prompt_tokens) - 1
    use_time_stopping = config.max_time is not None and config.trunc_id is not None

    # Build logit bias from suppressed_ids.  When time-based stopping is active,
    # auto-exclude trunc_id so the logit processor can force it.
    logit_bias = {tid: -10000 for tid in config.suppressed_ids}

    if traj_type == TrajectoryType.M1:
        stop_token_ids = list(config.end_token_ids)
        stop_token_ids.append(config.target_event_id)
    else:
        logit_bias[config.target_event_id] = -10000
        stop_token_ids = list(config.end_token_ids)

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
            "check_interval": config.time_check_interval,
        }
        extra_kwargs["custom_logit_processor"] = _get_logit_processor_str()

    output = await engine.async_generate(
        input_ids=prompt_tokens,
        sampling_params=sampling_params,
        **extra_kwargs,
    )

    meta = output.get("meta_info", {})
    finish_reason = meta.get("finish_reason", {})
    if isinstance(finish_reason, dict) and finish_reason.get("type") == "stop":
        terminal_token_id = finish_reason.get("matched")
    else:
        terminal_token_id = None
    output_ids = list(meta.get("output_ids", output.get("output_ids", [])))
    # --- Post-hoc exact truncation ---
    # The deferred processor may have overshot by up to check_interval tokens,
    # or may not have fired at all if the trajectory ended naturally but still
    # exceeded the horizon. Either way, we walk the tokens for exact trimming.
    was_time_truncated = False
    truncation_idx = None

    if use_time_stopping and output_ids:
        # Strip the trunc_id if the deferred processor forced it — we'll
        # re-derive the exact boundary below.
        if output_ids[-1] == config.trunc_id:
            output_ids = output_ids[:-1]

        output_ids, was_time_truncated, truncation_idx = _apply_time_truncation(
            output_ids,
            config.token_id_to_minutes,
            config.max_time,
        )

    return GeneratedTrajectory(
        patient_idx=patient_idx,
        sample_idx=sample_idx,
        traj_type=traj_type,
        prompt_len=len(prompt_tokens),
        output_ids=output_ids,
        timeline_terminating_id=terminal_token_id,
        was_time_truncated=was_time_truncated,
        truncation_idx=truncation_idx,
    )