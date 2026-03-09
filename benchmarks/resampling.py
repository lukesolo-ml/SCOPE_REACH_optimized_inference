"""Stratified resampling AUC analysis for generation-length sensitivity.

Draws balanced resamples with length-biased sampling to ensure coverage
across the full range of generation lengths. Used by benchmarks to plot
AUC vs. generation length.
"""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score


def length_biased_sample(
    indices: np.ndarray,
    lengths: np.ndarray,
    target_percentile: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `n` indices with probability concentrated near a target percentile."""
    target_value = np.percentile(lengths, target_percentile)
    iqr = np.percentile(lengths, 75) - np.percentile(lengths, 25)
    bw = max(iqr / 4.0, 1.0)

    weights = np.exp(-0.5 * ((lengths - target_value) / bw) ** 2)
    weights = weights / weights.sum()

    chosen = rng.choice(len(indices), size=n, replace=False, p=weights)
    return indices[chosen]


def stratified_resample_auc(
    outcome: np.ndarray,
    estimators: dict[str, np.ndarray],
    gen_lengths: np.ndarray,
    half_n: int,
    n_resamples: int,
    logger: logging.Logger,
    rng: np.random.Generator,
):
    """Draw balanced resamples and log per-resample AUCs.

    Each resample targets a different percentile of the generation-length
    distribution, giving broad coverage for AUC vs. length analysis.

    Args:
        outcome: Binary outcome array (full cohort).
        estimators: Dict of estimator name -> score array.
        gen_lengths: Mean generated token count per patient.
        half_n: Patients per class per resample.
        n_resamples: Number of resamples.
        logger: Logger for output.
        rng: numpy Generator for reproducibility.
    """
    pos_idx = np.where(outcome == 1)[0]
    neg_idx = np.where(outcome == 0)[0]
    pos_lengths = gen_lengths[pos_idx]
    neg_lengths = gen_lengths[neg_idx]

    estm_names = sorted(estimators.keys())
    header = (
        "RESAMPLE|header|"
        "mean_gen_len,mean_gen_len_pos,mean_gen_len_neg,"
        + ",".join(f"auc_{n}" for n in estm_names)
    )
    logger.info(header)

    for i in range(n_resamples):
        target_pct = rng.uniform(0, 100)

        pos_sel = length_biased_sample(pos_idx, pos_lengths, target_pct, half_n, rng)
        neg_sel = length_biased_sample(neg_idx, neg_lengths, target_pct, half_n, rng)

        sel = np.concatenate([pos_sel, neg_sel])
        y = outcome[sel]
        gl = gen_lengths[sel]

        aucs = []
        for n in estm_names:
            try:
                aucs.append(f"{roc_auc_score(y, estimators[n][sel]):.4f}")
            except ValueError:
                aucs.append("NA")

        row = (
            f"RESAMPLE|{i}|"
            f"{np.mean(gl):.1f},{np.mean(gl[y == 1]):.1f},{np.mean(gl[y == 0]):.1f},"
            + ",".join(aucs)
        )
        logger.info(row)
