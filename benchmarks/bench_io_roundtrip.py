"""Benchmark: validate IO roundtrip — scores from loaded trajectories match originals.

Workflow:
  1. Generate trajectories for the patient cohort.
  2. Score the trajectories directly (from generation results).
  3. Save trajectories to disk via quick_sco_re.io.
  4. Load trajectories back from disk.
  5. Score the loaded trajectories.
  6. Compare aggregated per-patient scores — they must be identical.
"""

import tempfile

import numpy as np

from quick_sco_re import (
    save_trajectories,
    load_trajectories,
    score_trajectories,
    aggregate_results,
)

from .common import (
    PhaseRunner,
    build_config,
    log_config,
    compute_phase_metrics,
    log_comparison_table,
    log_score_correlations,
    logger,
)


async def run(
    model_path, vocab, trunc_id, max_len, n_samp, time_check_interval,
    patient_tokens, outcome,
    token_id_to_minutes=None, max_time=None,
):
    use_time_stopping = max_time is not None
    config = build_config(
        vocab, max_len=max_len, n_samp=n_samp, trunc_id=trunc_id,
        time_check_interval=time_check_interval,
        use_time_stopping=use_time_stopping,
        token_id_to_minutes=token_id_to_minutes or {},
        max_time=max_time,
    )
    log_config(config, "io_roundtrip")

    # ------------------------------------------------------------------
    # Phase A: generate and score directly
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("IO ROUNDTRIP — Phase A: generate & score (direct)")
    logger.info("=" * 60)

    async with PhaseRunner(model_path, max_len, use_time_horizon=use_time_stopping) as runner:
        await runner.warmup(config, patient_tokens)
        traj_a, res_a, wt_a = await runner.run_generate_and_score(
            config, patient_tokens,
        )

        # ------------------------------------------------------------------
        # Save / load roundtrip
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory(prefix="mc_io_roundtrip_") as tmp_dir:
            logger.info(f"Saving {len(traj_a)} trajectories to {tmp_dir}")
            save_trajectories(traj_a, tmp_dir, config=config)

            logger.info("Loading trajectories back from disk")
            traj_loaded, config_loaded = load_trajectories(tmp_dir)

        # ------------------------------------------------------------------
        # Validate loaded trajectories match originals
        # ------------------------------------------------------------------
        assert len(traj_loaded) == len(traj_a), (
            f"Trajectory count mismatch: saved {len(traj_a)}, loaded {len(traj_loaded)}"
        )

        mismatches = []
        for i, (orig, loaded) in enumerate(zip(traj_a, traj_loaded)):
            if orig.patient_idx != loaded.patient_idx:
                mismatches.append(f"[{i}] patient_idx: {orig.patient_idx} vs {loaded.patient_idx}")
            if orig.sample_idx != loaded.sample_idx:
                mismatches.append(f"[{i}] sample_idx: {orig.sample_idx} vs {loaded.sample_idx}")
            if orig.traj_type != loaded.traj_type:
                mismatches.append(f"[{i}] traj_type: {orig.traj_type} vs {loaded.traj_type}")
            if orig.prompt_len != loaded.prompt_len:
                mismatches.append(f"[{i}] prompt_len: {orig.prompt_len} vs {loaded.prompt_len}")
            if orig.output_ids != loaded.output_ids:
                mismatches.append(f"[{i}] output_ids differ (len {len(orig.output_ids)} vs {len(loaded.output_ids)})")
            if orig.timeline_terminating_id != loaded.timeline_terminating_id:
                mismatches.append(f"[{i}] timeline_terminating_id: {orig.timeline_terminating_id} vs {loaded.timeline_terminating_id}")
            if orig.was_time_truncated != loaded.was_time_truncated:
                mismatches.append(f"[{i}] was_time_truncated: {orig.was_time_truncated} vs {loaded.was_time_truncated}")
            if orig.truncation_idx != loaded.truncation_idx:
                mismatches.append(f"[{i}] truncation_idx: {orig.truncation_idx} vs {loaded.truncation_idx}")

        if mismatches:
            for m in mismatches[:20]:
                logger.error(f"MISMATCH: {m}")
            raise AssertionError(
                f"Trajectory roundtrip failed with {len(mismatches)} field mismatches"
            )
        logger.info("All trajectory fields match after roundtrip")

        # ------------------------------------------------------------------
        # Phase B: score loaded trajectories
        # ------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("IO ROUNDTRIP — Phase B: score loaded trajectories")
        logger.info("=" * 60)

        import time as _time
        start = _time.time()
        scored_loaded = await score_trajectories(
            runner.engine, config, traj_loaded, patient_tokens,
        )
        res_b = aggregate_results(
            scored_loaded, num_patients=len(patient_tokens),
            target_event_id=config.target_event_id,
        )
        wt_b = _time.time() - start

    # ------------------------------------------------------------------
    # Compare results
    # ------------------------------------------------------------------
    phase_results = {}
    phase_results["direct"] = compute_phase_metrics(
        traj_a, res_a, outcome, patient_tokens, config,
        "Phase A (direct)", wt_a,
    )
    phase_results["loaded"] = compute_phase_metrics(
        traj_loaded, res_b, outcome, patient_tokens, config,
        "Phase B (loaded)", wt_b,
    )

    log_comparison_table(phase_results, outcome, "IO ROUNDTRIP COMPARISON")
    log_score_correlations(phase_results, "direct", "loaded")

    # ------------------------------------------------------------------
    # Assert exact equality
    # ------------------------------------------------------------------
    for est in ["M0", "M1", "M2"]:
        direct = phase_results["direct"][est]
        loaded = phase_results["loaded"][est]
        if not np.allclose(direct, loaded, atol=1e-12):
            diff = np.abs(direct - loaded)
            logger.error(
                f"{est} scores differ! max_diff={diff.max():.2e}, "
                f"mean_diff={diff.mean():.2e}"
            )
            raise AssertionError(
                f"{est} scores from loaded trajectories do not match direct scores"
            )
        logger.info(f"{est}: EXACT MATCH between direct and loaded scores")

    logger.info("IO roundtrip validation PASSED — all scores identical.")
