"""Benchmark: generation with deferred logit processor time-horizon stopping."""

from .common import (
    PhaseRunner, build_config, log_config, compute_phase_metrics, logger,
)
from .resampling import stratified_resample_auc

import numpy as np


async def run(model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
              token_id_to_minutes, max_time, patient_tokens, outcome,
              resample_size, n_resamples):
    config = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=True,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
    )
    log_config(config, "with_lp")

    async with PhaseRunner(model_path, max_len, use_time_horizon=True) as runner:
        await runner.warmup(config, patient_tokens)
        trajectories, results, wall_time = await runner.run_generate_and_score(
            config, patient_tokens,
        )

    logger.info("=" * 60)
    logger.info("RESULTS — WITH_LP (deferred logit processor)")
    logger.info("=" * 60)
    metrics = compute_phase_metrics(
        trajectories, results, outcome, patient_tokens, config,
        "with_lp", wall_time,
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
