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

async def check_engine_parity(
    engine,
    hf_model,
    prompts: Sequence[Sequence[int]],
    context_window: int | None = None,
    tolerance: float = 0.05,
) -> float:
    """Compare SGLang's next-token distribution after each prompt with the HF model's.

    A wrong attention backend, a misread config (e.g. RoPE scaling) or a different
    model file shows up here before any trajectory is generated. Both read the
    same context: with a rolling window, the last ``context_window - overhead - 2``
    prompt tokens, as generate_trajectory's first request. Use a handful of prompts:
    the full-vocabulary logprob request is slow in SGLang.

    Returns:
        The largest L1 distance between the two distributions.

    Raises:
        RuntimeError: if it exceeds ``tolerance``.
    """
    import asyncio

    import torch

    from .generation import _SGLANG_OVERHEAD

    V = hf_model.config.vocab_size
    keep = None if context_window is None else context_window - _SGLANG_OVERHEAD - 2
    ctxs = [list(p) if keep is None else list(p)[-keep:] for p in prompts]
    outs = await asyncio.gather(*[
        engine.async_generate(input_ids=c, sampling_params={"max_new_tokens": 1, "temperature": 1.0},
                              return_logprob=True, token_ids_logprob=list(range(V)))
        for c in ctxs
    ])
    dev = next(hf_model.parameters()).device
    worst = 0.0
    with torch.no_grad():
        for c, o in zip(ctxs, outs):
            ent = o["meta_info"]["output_token_ids_logprobs"][0]
            sg = np.zeros(V)
            for lp, t, _ in ent:
                sg[t] = np.exp(lp)
            lg = hf_model(input_ids=torch.tensor([c], device=dev)).logits[0, -1].float()
            hf = torch.softmax(lg, -1).cpu().numpy()
            worst = max(worst, float(np.abs(sg - hf).sum()))
    if worst > tolerance:
        raise RuntimeError(f"SGLang and HF next-token distributions differ (max L1 {worst:.4f} > {tolerance}): "
                           "check the attention backend and how the model config is read")
    return worst


def check_rope_config(model_path: str) -> None:
    """Fail if this transformers version reads the model's RoPE settings differently
    from how they were saved (config.json). Older and newer transformers versions
    store theta and scaling under different keys; a silent misread changes every
    position's encoding."""
    import json
    import os

    from transformers import AutoConfig

    from .models import require_servable

    require_servable(model_path)
    with open(os.path.join(model_path, "config.json")) as f:
        saved = json.load(f)
    want = saved.get("rope_parameters") or {"rope_theta": saved.get("rope_theta"), **(saved.get("rope_scaling") or {})}
    cfg = AutoConfig.from_pretrained(model_path)
    rp = getattr(cfg, "rope_parameters", None) or {}
    got_theta = getattr(cfg, "rope_theta", None) or rp.get("rope_theta")
    got_scaling = getattr(cfg, "rope_scaling", None) or rp
    want_type = want.get("rope_type", want.get("type", "default"))
    got_type = (got_scaling or {}).get("rope_type", (got_scaling or {}).get("type", "default"))
    if want.get("rope_theta") is not None and got_theta != want.get("rope_theta"):
        raise RuntimeError(f"RoPE theta read as {got_theta}, saved as {want.get('rope_theta')}")
    if want_type != "default" and got_type != want_type:
        raise RuntimeError(f"RoPE scaling read as {got_type!r}, saved as {want_type!r}")
