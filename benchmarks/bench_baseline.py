"""Benchmark: baseline generation without time-based stopping."""

from .common import (
    PhaseRunner, build_config, log_config, compute_phase_metrics, logger,
)
from .resampling import stratified_resample_auc

import numpy as np


async def run(model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
              patient_tokens, outcome, resample_size, n_resamples):
    config = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=False,
    )
    log_config(config, "baseline")

    async with PhaseRunner(model_path, max_len, use_time_horizon=False) as runner:
        await runner.warmup(config, patient_tokens)
        trajectories, results, wall_time = await runner.run_generate_and_score(
            config, patient_tokens,
        )

    logger.info("=" * 60)
    logger.info("RESULTS — BASELINE (no time stopping)")
    logger.info("=" * 60)
    metrics = compute_phase_metrics(
        trajectories, results, outcome, patient_tokens, config,
        "baseline", wall_time,
    )

    rng = np.random.default_rng(seed=42)
    stratified_resample_auc(
        outcome=outcome,
        estimators={"M0": metrics["M0"], "M1": metrics["M1"], "M2": metrics["M2"]},
        gen_lengths=metrics["gen_lengths"],
        half_n=resample_size // 2,
        n_resamples=n_resamples,
        logger=logger,
        rng=rng,
    )
