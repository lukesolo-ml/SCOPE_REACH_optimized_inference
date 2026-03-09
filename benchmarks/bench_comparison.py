"""Benchmark: side-by-side comparison of baseline vs logit-processor time stopping."""

from .common import (
    PhaseRunner, build_config, log_config,
    compute_phase_metrics, log_comparison_table, logger,
)


async def run(model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
              use_time_stopping, token_id_to_minutes, max_time,
              patient_tokens, outcome):
    config_no_time = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=False,
    )
    config_with_time = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=use_time_stopping,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
    )

    log_config(config_no_time, "benchmark/baseline")
    log_config(config_with_time, "benchmark/with_lp")

    phase_results = {}

    # Phase A: baseline
    logger.info("=" * 60)
    logger.info("Phase A: BASELINE (no time stopping)")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=False) as runner:
        await runner.warmup(config_no_time, patient_tokens)
        traj_a, res_a, wt_a = await runner.run_generate_and_score(
            config_no_time, patient_tokens,
        )

    phase_results["baseline"] = compute_phase_metrics(
        traj_a, res_a, outcome, patient_tokens, config_no_time,
        "Phase A (baseline)", wt_a,
    )

    # Phase B: with logit processor
    logger.info("=" * 60)
    logger.info("Phase B: WITH_LP (deferred logit processor)")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=True) as runner:
        await runner.warmup(config_with_time, patient_tokens)
        traj_b, res_b, wt_b = await runner.run_generate_and_score(
            config_with_time, patient_tokens,
        )

    phase_results["with_lp"] = compute_phase_metrics(
        traj_b, res_b, outcome, patient_tokens, config_with_time,
        "Phase B (with_lp)", wt_b,
    )

    log_comparison_table(phase_results, outcome, "BENCHMARK COMPARISON")
