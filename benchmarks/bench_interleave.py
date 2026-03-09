"""Benchmark: sequential vs interleaved generate+score pipeline.

Phase A: Standard two-phase (generate all, then score all).
Phase B: Interleaved (score each trajectory immediately after generation).
"""

from .common import (
    PhaseRunner, build_config, log_config,
    compute_phase_metrics, log_comparison_table,
    log_score_correlations, logger,
)


async def run(model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
              use_time_stopping, token_id_to_minutes, max_time,
              patient_tokens, outcome):
    config = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=use_time_stopping,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
    )
    log_config(config, "interleave_test")

    phase_results = {}

    # Phase A: sequential
    logger.info("=" * 60)
    logger.info("INTERLEAVE TEST — Phase A: sequential generate-then-score")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=use_time_stopping) as runner:
        await runner.warmup(config, patient_tokens)
        traj_a, res_a, wt_a = await runner.run_generate_and_score(
            config, patient_tokens,
        )

    phase_results["sequential"] = compute_phase_metrics(
        traj_a, res_a, outcome, patient_tokens, config,
        "Phase A (sequential)", wt_a,
    )

    # Phase B: interleaved
    logger.info("=" * 60)
    logger.info("INTERLEAVE TEST — Phase B: interleaved generate+score")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=use_time_stopping) as runner:
        await runner.warmup(config, patient_tokens)
        traj_b, res_b, wt_b = await runner.run_interleaved(
            config, patient_tokens,
        )

    phase_results["interleaved"] = compute_phase_metrics(
        traj_b, res_b, outcome, patient_tokens, config,
        "Phase B (interleaved)", wt_b,
    )

    log_comparison_table(phase_results, outcome, "INTERLEAVE TEST COMPARISON")
    log_score_correlations(phase_results, "sequential", "interleaved")

    logger.info(
        "AUCs will differ due to sampling stochasticity, but distributions "
        "should be statistically indistinguishable."
    )
