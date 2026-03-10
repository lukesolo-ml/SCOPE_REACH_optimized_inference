"""Entry point for mc_timeline benchmarks.

Replaces the monolithic run_tests.py with a clean dispatcher that delegates
to focused benchmark modules.

Usage:
    python -m benchmarks.run_benchmarks --generation_mode with_lp [options]
"""

import argparse
import asyncio
import logging
import os
import pathlib

import numpy as np
import polars as pl

from .common import build_token_id_to_minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="mc_timeline benchmark runner",
    )
    parser.add_argument(
        "--data_dir", type=pathlib.Path,
        default="/gpfs/data/bbj-lab/users/burkh4rt/data-mimic"
        if os.uname().nodename.startswith("cri")
        else "/mnt/bbj-lab/users/burkh4rt/data-mimic",
    )
    parser.add_argument("--data_version", type=str, default="Y21_first_24h")
    parser.add_argument("--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21")
    parser.add_argument("--tto_version", type=str, default="Y21_first_24h")
    parser.add_argument("--max_len", type=int, default=10_000)
    parser.add_argument("--n_samp", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=1_000)
    parser.add_argument("--resample_size", type=int, default=100)
    parser.add_argument("--n_resamples", type=int, default=200)
    parser.add_argument("--time_horizon_minutes", type=float, default=None)
    parser.add_argument("--time_check_interval", type=int, default=100)
    parser.add_argument(
        "--generation_mode", type=str, default="with_lp",
        choices=["baseline", "with_lp", "benchmark", "truncation_test", "interleave_test"],
    )
    return parser.parse_args()


def load_balanced_cohort(args):
    """Load data and build a 50/50 balanced cohort."""
    data_dir = pathlib.Path(args.data_dir).expanduser().resolve()

    df_all = pl.read_parquet(
        data_dir / f"{args.data_version}-tokenized" / "test" / "tokens_timelines.parquet"
    ).join(
        pl.read_parquet(
            data_dir / f"{args.tto_version}-tokenized" / "test" / "tokens_timelines_outcomes.parquet"
        ),
        how="left", on="hospitalization_id", validate="1:1",
    )

    n_per_class = args.test_size // 2
    df_pos = df_all.filter(pl.col("same_admission_death") == 1).sample(n=n_per_class, shuffle=True, seed=42)
    df_neg = df_all.filter(pl.col("same_admission_death") == 0).sample(n=n_per_class, shuffle=True, seed=42)
    df_test = pl.concat([df_pos, df_neg]).sample(fraction=1.0, shuffle=True, seed=42)

    logger.info(
        f"Balanced cohort: {df_test.height} patients "
        f"({df_pos.height} positive, {df_neg.height} negative)"
    )

    patient_tokens = df_test.select("tokens").to_series().to_list()
    outcome = df_test.select("same_admission_death").to_numpy().ravel()
    return patient_tokens, outcome


async def async_main():
    args = parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    model_path = str(pathlib.Path(args.model_loc).expanduser().resolve())
    data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
    patient_tokens, outcome = load_balanced_cohort(args)

    from fms_ehrs.framework.vocabulary import Vocabulary
    vocab = Vocabulary().load(
        data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
    )

    trunc_id = vocab("TRUNC")
    use_time_stopping = args.time_horizon_minutes is not None
    token_id_to_minutes = build_token_id_to_minutes(vocab) if use_time_stopping else {}

    if use_time_stopping:
        logger.info(
            f"Time-based stopping: horizon={args.time_horizon_minutes} min, "
            f"check_interval={args.time_check_interval}, "
            f"{len(token_id_to_minutes)} time tokens"
        )
    else:
        logger.info("Time-based stopping DISABLED")

    # Common kwargs for all benchmarks
    common = dict(
        model_path=model_path, vocab=vocab, trunc_id=trunc_id,
        max_len=args.max_len, n_samp=args.n_samp,
        time_check_interval=args.time_check_interval,
        patient_tokens=patient_tokens, outcome=outcome,
    )
    time_kwargs = dict(
        token_id_to_minutes=token_id_to_minutes,
        max_time=args.time_horizon_minutes,
    )

    mode = args.generation_mode

    if mode == "baseline":
        from .bench_baseline import run
        await run(**common, resample_size=args.resample_size, n_resamples=args.n_resamples)

    elif mode == "with_lp":
        from .bench_time_horizon import run
        await run(**common, **time_kwargs,
                  resample_size=args.resample_size, n_resamples=args.n_resamples)

    elif mode == "benchmark":
        from .bench_comparison import run
        await run(**common, **time_kwargs, use_time_stopping=use_time_stopping)

    elif mode == "truncation_test":
        if not use_time_stopping:
            logger.error("truncation_test requires --time_horizon_minutes")
            return
        from .bench_truncation import run
        await run(**common, **time_kwargs)

    elif mode == "interleave_test":
        from .bench_interleave import run
        await run(**common, **time_kwargs, use_time_stopping=use_time_stopping)

    logger.info("---fin")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
