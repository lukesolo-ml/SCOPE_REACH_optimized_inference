"""
Generate timeline completions and score them using the mc_timeline package.

This script handles the data/model-specific concerns:
  - Loading tokenized timelines from parquet
  - Vocabulary mapping
  - Outcome joining and classification metric reporting

The generic Monte Carlo generation and scoring is delegated to mc_timeline.
"""

import argparse
import asyncio
import math
import os
import pathlib

import numpy as np
import polars as pl
import sglang as sgl

from mc_timeline import (
    GenerationConfig,
    generate_and_score,
    save_scores,
    save_trajectories,
)

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.stats import bootstrap_ci
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=pathlib.Path,
    default="/gpfs/data/bbj-lab/users/burkh4rt/data-mimic"
    if os.uname().nodename.startswith("cri")
    else "/mnt/bbj-lab/users/burkh4rt/data-mimic",
)
parser.add_argument("--data_version", type=str, default="Y21_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument("--tto_version", type=str, default="Y21_first_24h")
parser.add_argument("--max_len", type=int, default=10_000)
parser.add_argument("--n_samp", type=int, default=20)
parser.add_argument("--test_size", type=int, default=1_000)
parser.add_argument(
    "--time_horizon_minutes",
    type=float,
    default=None,
    help="Stop generation after this many minutes of simulated clinical time. "
    "Requires the engine to run with disable_overlap_schedule and "
    "enable_custom_logit_processor. Omit to disable time-based stopping.",
)
args, unknowns = parser.parse_known_args()


# ---------------------------------------------------------------------------
# Time-token helpers (dataset-specific; must match vocabulary bin labels)
# ---------------------------------------------------------------------------

# Maps vocabulary time-bin token strings to their (lo, hi) minute bounds.
# The package receives the geometric mean of each bin as the token's time value.
_TIME_TOKEN_MINUTES: dict[str, tuple[float, float]] = {
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


def _build_token_id_to_minutes(vocab) -> dict[int, float]:
    """Return a mapping from token ID to elapsed minutes (geometric mean of bin bounds)."""
    mapping = {}
    for word, (lo, hi) in _TIME_TOKEN_MINUTES.items():
        if word in vocab:
            mapping[vocab(word)] = math.sqrt(lo * hi)
    return mapping


async def async_main():
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (args.data_dir, args.model_loc),
    )

    # Load tokenized test timelines
    df_test = (
        pl.read_parquet(
            data_dir
            / f"{args.data_version}-tokenized"
            / "test"
            / "tokens_timelines.parquet"
        )
        .sample(n=args.test_size)
        .lazy()
    )
    test_token_list = df_test.select("tokens").collect().to_series().to_list()

    # Load vocabulary and resolve special token IDs
    vocab = Vocabulary().load(
        data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
    )

    trunc_id = vocab("TRUNC")
    use_time_stopping = args.time_horizon_minutes is not None
    token_id_to_minutes = _build_token_id_to_minutes(vocab) if use_time_stopping else {}

    if use_time_stopping:
        logger.info(
            f"Time-based stopping enabled: horizon={args.time_horizon_minutes} min "
            f"({args.time_horizon_minutes / 60:.1f} h), "
            f"{len(token_id_to_minutes)} time tokens found in vocabulary"
        )
        if not token_id_to_minutes:
            logger.warning(
                "No time tokens found in vocabulary — time-based stopping will have no effect."
            )

    config = GenerationConfig(
        max_len=args.max_len,
        n_samp=args.n_samp,
        target_event_id=vocab("DSCG_Expired"),
        timeline_end_id=vocab("TL_END"),
        # TRUNC is kept in suppressed_ids so it stays a stop token when time
        # stopping is off.  When time stopping is on, generation.py automatically
        # exempts it from logit suppression so the logit processor can force it.
        suppressed_ids=[vocab("PAD"), trunc_id],
        trunc_id=trunc_id if use_time_stopping else None,
        token_id_to_minutes=token_id_to_minutes,
        max_time=args.time_horizon_minutes,
    )

    logger.info(f"Loaded {len(test_token_list)} patients, {args.n_samp} samples each")

    # Time-based stopping requires two additional engine flags:
    #   disable_overlap_schedule  – prevents race conditions on output_ids
    #   enable_custom_logit_processor – allows the logit processor to run
    engine_kwargs: dict = {}
    if use_time_stopping:
        engine_kwargs["disable_overlap_schedule"] = True
        engine_kwargs["enable_custom_logit_processor"] = True

    # Start engine and run generation + scoring
    engine = sgl.Engine(
        model_path=str(model_loc),
        skip_tokenizer_init=True,
        context_length=args.max_len,
        **engine_kwargs,
    )

    trajectories, results = await generate_and_score(
        engine, config, test_token_list
    )

    engine.shutdown()

    # Compute estimators
    M0 = np.array([np.mean(r.m0_samples) for r in results])
    M1 = np.array([np.mean(r.m1_samples) for r in results])
    M2 = np.array([np.mean(r.m2_samples) for r in results])

    # Load outcomes
    outcome = (
        df_test.join(
            pl.scan_parquet(
                data_dir
                / f"{args.tto_version}-tokenized"
                / "test"
                / "tokens_timelines_outcomes.parquet"
            ),
            how="left",
            on="hospitalization_id",
            validate="1:1",
        )
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )

    # Report results
    logger.info("=" * 50)
    logger.info(f"RESULTS (n={len(test_token_list)}, samples={args.n_samp})")
    if use_time_stopping:
        logger.info(
            f"Time horizon: {args.time_horizon_minutes} min "
            f"({args.time_horizon_minutes / 60:.1f} h)"
        )
    logger.info("=" * 50)

    for name, estm in {"M0": M0, "M1": M1, "M2": M2}.items():
        logger.info(f"{name}: mean={np.mean(estm):.4f}, max={np.max(estm):.4f}")
        log_classification_metrics(y_true=outcome, y_score=estm, logger=logger)
        logger.info(bootstrap_ci(y_true=outcome, y_score=estm))

    logger.info("---fin")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()