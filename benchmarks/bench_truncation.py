"""Benchmark: validate post-hoc truncation against native time-horizon stopping.

Phase A: Generate WITH time horizon, score normally.
Phase B: Generate WITHOUT time horizon, score with truncate_again=True.
Metrics should be very similar if post-hoc truncation is correct.
"""

import numpy as np

from .common import (
    PhaseRunner, build_config, log_config,
    compute_phase_metrics, log_comparison_table,
    log_score_correlations, logger,
)


async def run(model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
              token_id_to_minutes, max_time, patient_tokens, outcome):
    config_with_time = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=True,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
    )
    config_no_time = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=False,
    )
    # Config for scoring with post-hoc truncation (carries time info)
    config_score = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=True,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
    )

    log_config(config_with_time, "truncation_test/Phase A (native)")
    log_config(config_no_time, "truncation_test/Phase B gen (no time)")

    phase_results = {}

    # Phase A: native time stopping
    logger.info("=" * 60)
    logger.info("TRUNCATION TEST — Phase A: native time stopping")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=True) as runner:
        await runner.warmup(config_with_time, patient_tokens)
        traj_a, res_a, wt_a = await runner.run_generate_and_score(
            config_with_time, patient_tokens,
        )

    phase_results["native"] = compute_phase_metrics(
        traj_a, res_a, outcome, patient_tokens, config_with_time,
        "Phase A (native)", wt_a,
    )

    # Phase B: generate without time, score with post-hoc truncation
    logger.info("=" * 60)
    logger.info("TRUNCATION TEST — Phase B: post-hoc truncation at scoring")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=False) as runner:
        await runner.warmup(config_no_time, patient_tokens)
        traj_b, res_b, wt_b = await runner.run_generate_then_rescore(
            config_no_time, config_score, patient_tokens,
        )

    phase_results["posthoc"] = compute_phase_metrics(
        traj_b, res_b, outcome, patient_tokens, config_score,
        "Phase B (post-hoc)", wt_b,
    )

    log_comparison_table(phase_results, outcome, "TRUNCATION TEST COMPARISON")
    log_score_correlations(phase_results, "native", "posthoc")

    logger.info(
        "If post-hoc truncation is correct, AUCs should be similar and "
        "per-patient correlations should be very high (r > 0.99)."
    )
