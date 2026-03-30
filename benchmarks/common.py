"""Shared utilities for mc_timeline benchmarks.

Consolidates config building, phase running, metric computation, and
comparison table logging that were duplicated across 5 mode runners
in the original run_tests.py.
"""

import asyncio
import gc
import logging
import math
import time
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sglang as sgl

from mc_timeline import (
    GenerationConfig,
    generate_and_score,
    generate_trajectories,
    score_trajectories,
    aggregate_results,
    create_engine,
)
from mc_timeline.diagnostics import log_trajectory_diagnostics


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-token vocabulary (dataset-specific; override for other datasets)
# ---------------------------------------------------------------------------

TIME_TOKEN_MINUTES: dict[str, tuple[float, float]] = {
    "T_5m-15m": (5, 15),
    "T_15m-1h": (15, 60),
    "T_1h-2h": (60, 120),
    "T_2h-6h": (120, 360),
    "T_6h-12h": (360, 720),
    "T_12h-1d": (720, 1440),
    "T_1d-3d": (1440, 4320),
    "T_3d-1w": (4320, 10080),
    "T_1w-2w": (10080, 20160),
    "T_2w-1mt": (20160, 43200),
    "T_1mt-3mt": (43200, 129600),
    "T_3mt-6mt": (129600, 259200),
    "T_6mt+": (259200, 518400),
}


def build_token_id_to_minutes(vocab) -> dict[int, float]:
    """Map token IDs to elapsed minutes using geometric mean of bin bounds."""
    mapping = {}
    for word, (lo, hi) in TIME_TOKEN_MINUTES.items():
        if word in vocab:
            mapping[vocab(word)] = math.sqrt(lo * hi)
    return mapping


# ---------------------------------------------------------------------------
# Unified config builder
# ---------------------------------------------------------------------------


def build_config(
    vocab,
    max_len: int,
    n_samp: int,
    trunc_id: int,
    time_check_interval: int = 100,
    use_time_stopping: bool = False,
    token_id_to_minutes: dict[int, float] | None = None,
    max_time: float | None = None,
) -> GenerationConfig:
    """Build a GenerationConfig with optional time-stopping.

    Replaces the three separate builders (_build_config, _build_config_no_time,
    _build_config_score_only) from the original script.
    """
    return GenerationConfig(
        max_len=max_len,
        n_samp=n_samp,
        target_event_id=vocab("DSCG_Expired"),
        end_token_ids={
            tid for word, tid in vocab.lookup.items()
            if isinstance(word, str) and word.startswith("DSCG")
        },
        suppressed_ids=[vocab("PAD"), trunc_id, vocab("TL_END")],
        trunc_id=trunc_id if use_time_stopping else None,
        token_id_to_minutes=token_id_to_minutes or {},
        max_time=max_time if use_time_stopping else None,
        time_check_interval=time_check_interval,
    )


def log_config(config: GenerationConfig, label: str):
    """Log key GenerationConfig fields."""
    use_time = config.max_time is not None and config.trunc_id is not None
    logger.info(
        f"CONFIG [{label}]: max_len={config.max_len}, n_samp={config.n_samp}, "
        f"trunc_id={config.trunc_id}, max_time={config.max_time}, "
        f"check_interval={config.time_check_interval}, "
        f"time_tokens={len(config.token_id_to_minutes)}, "
        f"use_time_stopping={use_time}"
    )


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


class PhaseRunner:
    """Encapsulates engine lifecycle and phase execution.

    Handles: create engine -> warmup -> run -> shutdown -> return results.
    """

    def __init__(self, model_path: str, max_len: int, use_time_horizon: bool):
        self.model_path = model_path
        self.max_len = max_len
        self.use_time_horizon = use_time_horizon
        self.engine = None

    async def __aenter__(self):
        self.engine = create_engine(
            model_path=self.model_path,
            max_len=self.max_len,
            use_time_horizon=self.use_time_horizon,
        )
        return self

    async def __aexit__(self, *exc):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            gc.collect()
            torch.cuda.empty_cache()
            # Brief yield to let CUDA finalize resource release before
            # a subsequent engine is created in the same process.
            await asyncio.sleep(1)

    async def warmup(self, config: GenerationConfig, tokens: Sequence[list[int]]):
        """Run a small warmup pass."""
        logger.info("Warmup pass...")
        await generate_and_score(
            self.engine, config, tokens[:2],
            target_token_id=config.target_event_id,
        )

    async def run_generate_and_score(
        self, config: GenerationConfig, patient_tokens: Sequence[list[int]],
    ):
        """Generate and score, returning (trajectories, results, wall_time)."""
        start = time.time()
        trajectories, results = await generate_and_score(
            self.engine, config, patient_tokens,
            target_token_id=config.target_event_id,
        )
        wall_time = time.time() - start
        return trajectories, results, wall_time

    async def run_generate_then_rescore(
        self,
        gen_config: GenerationConfig,
        score_config: GenerationConfig,
        patient_tokens: Sequence[list[int]],
    ):
        """Generate without time horizon, then score with post-hoc truncation."""
        start = time.time()
        trajectories = await generate_trajectories(
            self.engine, gen_config, patient_tokens,
        )
        scored = await score_trajectories(
            self.engine, score_config, trajectories, patient_tokens,
            truncate_again=True,
        )
        results = aggregate_results(
            scored, num_patients=len(patient_tokens),
            target_event_id=gen_config.target_event_id,
        )
        wall_time = time.time() - start
        return trajectories, results, wall_time


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_phase_metrics(
    trajectories, results, outcome, patient_tokens, config, phase_name,
    wall_time,
) -> dict:
    """Compute and log metrics for a phase. Returns a metrics dict."""
    M0 = np.array([np.mean(r.m0_samples) for r in results])
    M1 = np.array([np.mean(r.m1_samples) for r in results])
    M2 = np.array([np.mean(r.m2_samples) for r in results])

    len_by_patient: dict[int, list[int]] = defaultdict(list)
    for traj in trajectories:
        len_by_patient[traj.patient_idx].append(len(traj.output_ids))
    gen_lengths = np.array([
        np.mean(len_by_patient[i]) for i in range(len(patient_tokens))
    ])

    total_gen_tokens = sum(len(t.output_ids) for t in trajectories)

    logger.info(f"--- {phase_name} ---")
    logger.info(f"Wall time: {wall_time:.2f}s")
    logger.info(f"Total generated tokens: {total_gen_tokens:,}")
    logger.info(f"Gen throughput: {total_gen_tokens / wall_time:.0f} tok/s")

    log_trajectory_diagnostics(trajectories, config, phase_name, logger)

    for name, estm in {"M0": M0, "M1": M1, "M2": M2}.items():
        auc = safe_auc(outcome, estm)
        logger.info(f"{name}: mean={np.mean(estm):.4f}, AUC={auc}")

    return {
        "wall_time": wall_time,
        "gen_tokens": total_gen_tokens,
        "M0": M0, "M1": M1, "M2": M2,
        "gen_lengths": gen_lengths,
    }


def safe_auc(y_true, y_score) -> str:
    try:
        return f"{roc_auc_score(y_true, y_score):.4f}"
    except ValueError:
        return "NA"


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def log_comparison_table(
    phase_results: dict[str, dict],
    outcome: np.ndarray,
    title: str = "COMPARISON",
):
    """Print a side-by-side comparison table for multiple phases."""
    phase_names = list(phase_results.keys())

    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)

    header = f"{'Metric':<35}" + "".join(f"{n:>15}" for n in phase_names)
    logger.info(header)
    logger.info("-" * 70)

    metrics = [
        ("Wall time (s)", lambda pr: f"{pr['wall_time']:.2f}"),
        ("Gen tokens", lambda pr: f"{pr['gen_tokens']:,}"),
        ("Gen tok/s", lambda pr: f"{pr['gen_tokens'] / pr['wall_time']:.0f}"),
        ("AUC M0", lambda pr: safe_auc(outcome, pr["M0"])),
        ("AUC M1", lambda pr: safe_auc(outcome, pr["M1"])),
        ("AUC M2", lambda pr: safe_auc(outcome, pr["M2"])),
        ("Mean M0", lambda pr: f"{np.mean(pr['M0']):.4f}"),
        ("Mean M1", lambda pr: f"{np.mean(pr['M1']):.4f}"),
        ("Mean M2", lambda pr: f"{np.mean(pr['M2']):.4f}"),
    ]

    if any("gen_lengths" in pr for pr in phase_results.values()):
        metrics.insert(3, (
            "Mean gen length",
            lambda pr: f"{np.mean(pr.get('gen_lengths', [0])):.1f}",
        ))

    for name, fn in metrics:
        vals = "".join(f"{fn(phase_results[p]):>15}" for p in phase_names)
        logger.info(f"{name:<35}{vals}")

    logger.info("=" * 70)

    # Speedups relative to first phase
    if len(phase_names) > 1:
        base_time = phase_results[phase_names[0]]["wall_time"]
        for p in phase_names[1:]:
            speedup = base_time / phase_results[p]["wall_time"]
            logger.info(f"Speedup ({p} vs {phase_names[0]}): {speedup:.2f}x")


def log_score_correlations(
    phase_results: dict[str, dict],
    phase_a: str,
    phase_b: str,
):
    """Log per-patient score correlations between two phases."""
    for est in ["M0", "M1", "M2"]:
        a = phase_results[phase_a][est]
        b = phase_results[phase_b][est]
        corr = np.corrcoef(a, b)[0, 1]
        mae = np.mean(np.abs(a - b))
        logger.info(
            f"{est} correlation ({phase_a} vs {phase_b}): "
            f"r={corr:.6f}, MAE={mae:.6f}"
        )
