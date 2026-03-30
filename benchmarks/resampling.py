"""Bootstrap resampling AUC analysis.

Draws balanced bootstrap resamples to compute AUC confidence intervals
for each estimator.
"""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score


def stratified_resample_auc(
    outcome: np.ndarray,
    estimators: dict[str, np.ndarray],
    half_n: int,
    n_resamples: int,
    logger: logging.Logger,
    rng: np.random.Generator,
):
    """Draw balanced bootstrap resamples and log per-resample AUCs.

    Each resample draws half_n positive and half_n negative patients
    uniformly at random (with replacement), giving balanced classes
    for AUC estimation.

    Args:
        outcome: Binary outcome array (full cohort).
        estimators: Dict of estimator name -> score array.
        half_n: Patients per class per resample.
        n_resamples: Number of resamples.
        logger: Logger for output.
        rng: numpy Generator for reproducibility.
    """
    pos_idx = np.where(outcome == 1)[0]
    neg_idx = np.where(outcome == 0)[0]

    estm_names = sorted(estimators.keys())
    header = "RESAMPLE|header|" + ",".join(f"auc_{n}" for n in estm_names)
    logger.info(header)

    for i in range(n_resamples):
        pos_sel = rng.choice(pos_idx, size=half_n, replace=True)
        neg_sel = rng.choice(neg_idx, size=half_n, replace=True)

        sel = np.concatenate([pos_sel, neg_sel])
        y = outcome[sel]

        aucs = []
        for n in estm_names:
            try:
                aucs.append(f"{roc_auc_score(y, estimators[n][sel]):.4f}")
            except ValueError:
                aucs.append("NA")

        row = f"RESAMPLE|{i}|" + ",".join(aucs)
        logger.info(row)