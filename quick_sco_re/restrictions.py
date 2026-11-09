"""First-token restrictions: logit biases for the first generated token.

Silence restriction
-------------------
A prompt cut at a fixed time (e.g. 24 hours after admission) ends at the patient's
last event, not at the cut. A patient silent for x minutes before the cut must
have their next event at least x minutes later, so the first generated token has
to be a time-gap token whose bin can contain or exceed x. Without this, the
model continues as if the last event had just happened.

Time gaps are floored to whole minutes and the bins are right-closed, (lo, hi],
so with m = floor(x) the allowed bins are those with hi >= m. Every other token
is forbidden on that step.

  mode "hard"  allowed bins keep their probabilities (renormalised by sampling).
  mode "prop"  the bin containing m is also scaled by the share of its
               whole-minute values that are >= m, (hi - m + 1) / (hi - lo).

Below ``min_minutes`` (by default the smallest bin's lower edge: shorter gaps
get no time token) nothing is restricted.

Pass the result as ``first_token_bias`` to generate_trajectory, or a list of them
(one per patient) as ``first_token_biases`` to generate_trajectories /
generate_and_score. The bias forbids every non-time token, so it forbids every
tracked event, as those calls require.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

FORBID = -10000.0


def silence_restriction_bias(
    silence_minutes: float | None,
    time_token_bounds: Mapping[int, tuple[float, float]],
    vocab_ids: Iterable[int],
    mode: str = "hard",
    min_minutes: float | None = None,
) -> dict[int, float] | None:
    """First-token logit bias for a patient silent for ``silence_minutes``.

    Args:
        silence_minutes: Minutes from the last prompt event to the prompt cut.
            None or non-finite means unknown: no restriction.
        time_token_bounds: Time-gap token id -> (lo, hi] in minutes.
        vocab_ids: Every token id of the vocabulary (non-time tokens are forbidden).
        mode: "hard" or "prop" (see module docstring).
        min_minutes: No restriction below this many minutes. Defaults to the
            smallest lower bin edge.

    Returns:
        Token id -> additive logit bias, or None when nothing is restricted.
    """
    if mode not in ("hard", "prop"):
        raise ValueError(f"mode must be 'hard' or 'prop', got {mode!r}")
    if not time_token_bounds:
        raise ValueError("time_token_bounds is empty")
    if silence_minutes is None or not math.isfinite(silence_minutes):
        return None
    floor_min = min(lo for lo, _ in time_token_bounds.values()) if min_minutes is None else float(min_minutes)
    m = math.floor(float(silence_minutes) + 1e-9)
    if m < floor_min:
        return None
    bias = {int(t): FORBID for t in vocab_ids if t not in time_token_bounds}
    for tid, (lo, hi) in time_token_bounds.items():
        if hi < m:
            bias[int(tid)] = FORBID
        elif mode == "prop" and lo < m:
            bias[int(tid)] = math.log((hi - m + 1) / (hi - lo))
    return bias


def allowed_mask(bias: Mapping[int, float] | None, vocab_size: int):
    """Boolean numpy mask of the tokens a first-token bias leaves possible
    (all True for None). Biases at or below FORBID / 2 count as forbidden."""
    import numpy as np

    mask = np.ones(vocab_size, dtype=bool)
    if bias:
        for t, b in bias.items():
            if b <= FORBID / 2:
                mask[int(t)] = False
    return mask
