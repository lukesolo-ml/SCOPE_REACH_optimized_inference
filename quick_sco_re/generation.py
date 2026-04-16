"""
Trajectory generation for SCOPE/REACH risk prediction.

Generates M1 (target-event-allowed) and M2 (target-event-forbidden) trajectories
using an SGLang engine. Generation is done without logprobs for speed; scoring
is handled separately.

TODO: Double check simultaneous generation and scoring. It is currently much slower under all configs
NOTE: Simultaneous generation and scoring is currently not recommended.

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
      TODO: This isn't particularly robust since not every vocab will have such an id
"""

import atexit
import collections
import os
import shutil
import tempfile
import uuid

import numpy as np
import sglang as sgl
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

from .structures import GeneratedTrajectory, GenerationConfig, TrajectoryType

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
                # Key that we are directly setting the logit since trunc_id
                # is likely in restricted vocab
                logits[i, trunc_id] = 0.0

        return logits


# ---------------------------------------------------------------------------
# Result channel for inline SCOPE/REACH
# ---------------------------------------------------------------------------
#
# Two channels are supported:
#
#   1. **Filesystem** (authoritative for the live engine). SGLang runs the
#      custom logit processor in a scheduler subprocess, so module-level dict
#      mutations in the subprocess don't reach the client. The processor
#      writes per-request .npz files into a tmpdir whose path is passed
#      through custom_params; the client reads/unlinks them after generation.
#
#   2. **In-memory dict** (fallback used by unit tests). When _sr_result_dir
#      is absent from param_dict (direct-call tests that don't go through the
#      engine), the processor writes to this dict instead.

_RESULTS_MAX_SIZE = 10_000

_RESULTS: collections.OrderedDict[str, dict] = collections.OrderedDict()

_RESULT_DIR: str | None = None


def _get_result_dir() -> str:
    """Lazily create (and register for cleanup) a tmpdir for result files."""
    global _RESULT_DIR
    if _RESULT_DIR is None:
        _RESULT_DIR = tempfile.mkdtemp(prefix="scope_reach_inline_")
        atexit.register(_cleanup_result_dir)
    return _RESULT_DIR


def _cleanup_result_dir() -> None:
    global _RESULT_DIR
    if _RESULT_DIR and os.path.isdir(_RESULT_DIR):
        shutil.rmtree(_RESULT_DIR, ignore_errors=True)
    _RESULT_DIR = None


def _store_result(request_id: str, data: dict) -> None:
    """Store inline SCOPE/REACH result in-memory (fallback, tests only)."""
    _RESULTS[request_id] = data
    while len(_RESULTS) > _RESULTS_MAX_SIZE:
        _RESULTS.popitem(last=False)


def _write_result_file(result_dir: str, request_id: str, data: dict) -> None:
    """Atomically write a result .npz file (tmp + rename)."""
    final_path = os.path.join(result_dir, f"{request_id}.npz")
    tmp_path = os.path.join(result_dir, f"{request_id}.{os.getpid()}.tmp")
    try:
        np.savez(
            tmp_path,
            scope=data["scope"],
            reach=data["reach"],
            occurred_flag=data["occurred_flag"],
            occurred_index=data["occurred_index"],
        )
        os.replace(tmp_path, final_path)
    except OSError:
        # Best-effort: if write fails, client will see None and log a warning
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _read_result_file(result_dir: str, request_id: str) -> dict | None:
    """Read and unlink a result .npz file; return None if missing/corrupt."""
    path = os.path.join(result_dir, f"{request_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            result = {
                "scope": data["scope"].copy(),
                "reach": data["reach"].copy(),
                "occurred_flag": data["occurred_flag"].copy(),
                "occurred_index": data["occurred_index"].copy(),
            }
    except (OSError, EOFError, ValueError):
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return result


def pop_inline_result(request_id: str, result_dir: str | None = None) -> dict | None:
    """Pop and return inline SCOPE/REACH result for a completed request.

    If result_dir is provided, checks the filesystem channel first. Falls back
    to the in-memory _RESULTS dict (populated by direct-call unit tests).
    """
    if result_dir is not None:
        file_result = _read_result_file(result_dir, request_id)
        if file_result is not None:
            return file_result
    return _RESULTS.pop(request_id, None)


# ---------------------------------------------------------------------------
# Inline SCOPE/REACH logit processor
# ---------------------------------------------------------------------------


class InlineScopeReachProcessor(CustomLogitProcessor):
    """Compute SCOPE and REACH estimates inline during generation.

    Observational only — returns logits unmodified.

    custom_params (passed per request) must contain:
        tracked_ids  : list[int] — vocab token IDs to estimate
        request_id   : str       — unique ID for result retrieval

    Optional:
        tracked_name : str       — label for downstream keying

    The processor lazily allocates per-request state on first call:
        _sr_tracked_set   : set[int]
        _sr_id_to_idx     : dict[int, int]
        _sr_occurred_flag : np.ndarray[bool]   shape (K,)
        _sr_occurred_index: np.ndarray[int64]  shape (K,)
        _sr_scope         : np.ndarray[float64] shape (K,)
        _sr_reach         : np.ndarray[float64] shape (K,)
        _sr_cursor        : int

    Requires disable_overlap_schedule=True on the SGLang engine.
    """

    def __call__(self, logits, custom_param_list):
        for i, param_dict in enumerate(custom_param_list):
            req = param_dict.get("__req__")
            if req is None:
                continue

            tracked_ids = param_dict.get("tracked_ids")
            if tracked_ids is None:
                continue

            request_id = param_dict.get("request_id")
            result_dir = param_dict.get("_sr_result_dir")

            # --- Lazy initialization ---
            if "_sr_cursor" not in param_dict:
                K = len(tracked_ids)
                param_dict["_sr_tracked_set"] = set(tracked_ids)
                param_dict["_sr_id_to_idx"] = {tid: idx for idx, tid in enumerate(tracked_ids)}
                param_dict["_sr_occurred_flag"] = np.zeros(K, dtype=bool)
                param_dict["_sr_occurred_index"] = np.full(K, -1, dtype=np.int64)
                param_dict["_sr_scope"] = np.zeros(K, dtype=np.float64)
                param_dict["_sr_reach"] = np.zeros(K, dtype=np.float64)
                param_dict["_sr_cursor"] = 0

            tracked_set = param_dict["_sr_tracked_set"]
            id_to_idx = param_dict["_sr_id_to_idx"]
            occurred_flag = param_dict["_sr_occurred_flag"]
            occurred_index = param_dict["_sr_occurred_index"]
            scope = param_dict["_sr_scope"]
            reach = param_dict["_sr_reach"]
            cursor = param_dict["_sr_cursor"]

            n_generated = len(req.output_ids)

            # --- Step 1: Advance cursor, flip occurred flags ---
            for j in range(cursor, n_generated):
                tid = int(req.output_ids[j])
                if tid in tracked_set:
                    k = id_to_idx[tid]
                    if not occurred_flag[k]:
                        occurred_flag[k] = True
                        occurred_index[k] = j

            param_dict["_sr_cursor"] = n_generated

            # --- Early exit if all tracked tokens have occurred ---
            if occurred_flag.all():
                if request_id is not None:
                    payload = {
                        "scope": scope.copy(),
                        "reach": reach.copy(),
                        "occurred_flag": occurred_flag.copy(),
                        "occurred_index": occurred_index.copy(),
                    }
                    if result_dir is not None:
                        _write_result_file(result_dir, request_id, payload)
                    else:
                        _store_result(request_id, payload)
                continue

            # --- Step 2: Compute probabilities for current step ---
            # Use logsumexp + gather for efficiency (avoids full-vocab softmax alloc)
            logits_i = logits[i].float()
            log_Z = torch.logsumexp(logits_i, dim=-1)
            tracked_ids_tensor = torch.tensor(tracked_ids, device=logits.device, dtype=torch.long)
            log_p_tracked = logits_i[tracked_ids_tensor] - log_Z
            p_tracked = log_p_tracked.exp().cpu().numpy().astype(np.float64)

            # --- Step 3: Update SCOPE and REACH for non-occurred tokens ---
            mask = ~occurred_flag
            scope[mask] += p_tracked[mask]
            reach[mask] = 1.0 - (1.0 - reach[mask]) * (1.0 - p_tracked[mask])

            # --- Step 4: Store result for retrieval ---
            if request_id is not None:
                payload = {
                    "scope": scope.copy(),
                    "reach": reach.copy(),
                    "occurred_flag": occurred_flag.copy(),
                    "occurred_index": occurred_index.copy(),
                }
                if result_dir is not None:
                    _write_result_file(result_dir, request_id, payload)
                else:
                    _store_result(request_id, payload)

        return logits


# ---------------------------------------------------------------------------
# Chained processor (composes time-horizon + inline SCOPE/REACH)
# ---------------------------------------------------------------------------


class ChainedProcessor(CustomLogitProcessor):
    """Dispatches to time-horizon and/or inline SCOPE/REACH processors.

    Checks which keys are present in custom_params and runs the appropriate
    sub-processors. This allows a single custom_logit_processor per request
    while supporting both features independently or together.
    """

    def __init__(self):
        super().__init__()
        self._time_horizon = DeferredTimeHorizonProcessor()
        self._scope_reach = InlineScopeReachProcessor()

    def __call__(self, logits, custom_param_list):
        # Time-horizon processor may modify logits (force trunc_id)
        has_time = any("time_horizon" in p for p in custom_param_list)
        has_sr = any("tracked_ids" in p for p in custom_param_list)

        if has_time:
            logits = self._time_horizon(logits, custom_param_list)
        if has_sr:
            logits = self._scope_reach(logits, custom_param_list)

        return logits


# Serialized once at module level and reused across all generate_trajectory calls.
_LOGIT_PROCESSOR_STR: str | None = None


def _get_logit_processor_str() -> str:
    global _LOGIT_PROCESSOR_STR
    if _LOGIT_PROCESSOR_STR is None:
        _LOGIT_PROCESSOR_STR = ChainedProcessor().to_str()
    return _LOGIT_PROCESSOR_STR


# ---------------------------------------------------------------------------
# Post-hoc exact time truncation
# ---------------------------------------------------------------------------


def apply_time_truncation(
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
    use_inline_sr = config.tracked_ids is not None
    use_processor = use_time_stopping or use_inline_sr

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

    # --- Custom params for logit processor(s) ---
    request_id: str | None = None
    custom_params: dict = {}

    if use_time_stopping:
        custom_params.update({
            "time_horizon": config.max_time,
            "trunc_id": config.trunc_id,
            "time_token_map": config.token_id_to_minutes,
            "check_interval": config.time_check_interval,
        })

    if use_inline_sr:
        request_id = str(uuid.uuid4())
        custom_params.update({
            "tracked_ids": config.tracked_ids,
            "request_id": request_id,
            "_sr_result_dir": _get_result_dir(),
        })
        if config.tracked_name is not None:
            custom_params["tracked_name"] = config.tracked_name

    if use_processor:
        sampling_params["custom_params"] = custom_params
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

        output_ids, was_time_truncated, truncation_idx = apply_time_truncation(
            output_ids,
            config.token_id_to_minutes,
            config.max_time,
        )

    # --- Retrieve inline SCOPE/REACH results ---
    scope_estimates = None
    reach_estimates = None
    occurred_flag = None
    occurred_index = None
    inline_tracked_ids = None
    inline_tracked_name = None

    if use_inline_sr and request_id is not None:
        sr_result = pop_inline_result(request_id, result_dir=_get_result_dir())
        if sr_result is not None:
            scope_estimates = sr_result["scope"]
            reach_estimates = sr_result["reach"]
            occurred_flag = sr_result["occurred_flag"]
            occurred_index = sr_result["occurred_index"]
        inline_tracked_ids = list(config.tracked_ids)
        inline_tracked_name = config.tracked_name

    return GeneratedTrajectory(
        patient_idx=patient_idx,
        sample_idx=sample_idx,
        traj_type=traj_type,
        prompt_len=len(prompt_tokens),
        output_ids=output_ids,
        timeline_terminating_id=terminal_token_id,
        was_time_truncated=was_time_truncated,
        truncation_idx=truncation_idx,
        scope_estimates=scope_estimates,
        reach_estimates=reach_estimates,
        occurred_flag=occurred_flag,
        occurred_index=occurred_index,
        inline_tracked_ids=inline_tracked_ids,
        inline_tracked_name=inline_tracked_name,
    )