#!/usr/bin/env python3
"""Merge sharded SCOPE/REACH outputs into one directory compatible with analysis.ipynb.

Each SLURM array task run by run_timelines_shard.py writes its results to a
subdirectory:

    {base_output_dir}/shard_000/
    {base_output_dir}/shard_001/
    ...

This script combines them into a single merged directory that looks exactly like
the output of a monolithic run_timelines.py run — so analysis.ipynb can be
pointed at it unchanged.

Fault tolerance
---------------
If a shard directory is absent or its run_summary.json is missing, that shard is
skipped with a warning.  The remaining shards are merged normally.  The patients
from the failed shard are simply absent from the merged output (their subject_ids
won't appear, their scores won't be included).  AUC / calibration computed from
the merged output are still valid — just over a slightly smaller cohort.

Usage
-----
    python merge_shards.py \\
        --base-output-dir ./scope_reach_output_ucmc \\
        --n-shards 8

    # Custom merged output dir:
    python merge_shards.py \\
        --base-output-dir ./scope_reach_output_ucmc \\
        --n-shards 8 \\
        --merged-dir ./scope_reach_output_ucmc_merged

    # Explicit shard dirs (e.g. after a partial run):
    python merge_shards.py \\
        --shard-dirs ./scope_reach_output_ucmc/shard_000 \\
                     ./scope_reach_output_ucmc/shard_002 \\
        --merged-dir ./scope_reach_output_ucmc_merged
"""

import argparse
import json
import logging
import pathlib
import shutil
import sys

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)
logger = logging.getLogger("merge_shards")


# ---------------------------------------------------------------------------
# Score file helpers (mirrors quick_sco_re/io.py format)
# ---------------------------------------------------------------------------

def _pad_to_2d(rows: list[list[float]], fill: float = float("nan")) -> np.ndarray:
    """Pad a ragged list of rows into a rectangular float32 array."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    max_len = max(len(r) for r in rows)
    out = np.full((len(rows), max_len), fill, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def load_scores_npz(path: pathlib.Path) -> dict:
    """Load a scores_*.npz file produced by save_scores() in quick_sco_re/io.py."""
    data = np.load(path)
    result = {k: data[k] for k in data.files}
    return result


def save_merged_scores(
    M0: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    M0_raw: np.ndarray,
    M1_raw: np.ndarray,
    M2_raw: np.ndarray,
    avg_m1_tokens: float | None,
    avg_m2_tokens: float | None,
    output_path: pathlib.Path,
) -> None:
    """Write merged scores in the same format as quick_sco_re save_scores()."""
    arrays: dict[str, np.ndarray] = {
        "M0": M0.astype(np.float64),
        "M1": M1.astype(np.float64),
        "M2": M2.astype(np.float64),
        "M0_raw": M0_raw.astype(np.float32),
        "M1_raw": M1_raw.astype(np.float32),
        "M2_raw": M2_raw.astype(np.float32),
    }
    if avg_m1_tokens is not None:
        arrays["avg_m1_tokens"] = np.array([avg_m1_tokens], dtype=np.float64)
    if avg_m2_tokens is not None:
        arrays["avg_m2_tokens"] = np.array([avg_m2_tokens], dtype=np.float64)
    np.savez_compressed(output_path, **arrays)


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def discover_shard_dirs(
    base_output_dir: pathlib.Path,
    n_shards: int,
) -> tuple[list[pathlib.Path], list[int]]:
    """Return (present_dirs, missing_indices) for shards 0..n_shards-1."""
    present = []
    missing = []
    for i in range(n_shards):
        d = base_output_dir / f"shard_{i:03d}"
        summary = d / "run_summary.json"
        if d.exists() and summary.exists():
            present.append(d)
        else:
            missing.append(i)
            logger.warning(f"Shard {i:03d} missing or incomplete at {d} — skipping")
    return present, missing


def merge(
    shard_dirs: list[pathlib.Path],
    merged_dir: pathlib.Path,
    n_shards_total: int | None = None,
) -> None:
    if not shard_dirs:
        raise RuntimeError("No shard directories to merge")

    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Merging {len(shard_dirs)} shard(s) → {merged_dir}")

    # ---- Load summaries ----
    summaries = []
    for d in shard_dirs:
        with open(d / "run_summary.json") as f:
            summaries.append(json.load(f))

    # Validate n_samp consistency
    n_samp_values = {s["n_samp"] for s in summaries}
    if len(n_samp_values) > 1:
        raise RuntimeError(f"Shards have different n_samp values: {n_samp_values}")
    n_samp = n_samp_values.pop()

    outcome_names = list(summaries[0]["outcomes"].keys())
    methods = summaries[0].get("methods", ["M1", "M2"])
    score_inline = summaries[0].get("score_inline", False)
    shard_indices = [s.get("shard_idx") for s in summaries]
    n_shards_from_summary = summaries[0].get("n_shards")

    # ---- Build global patient list ----
    all_subject_ids: list[str] = []
    shard_patient_counts: list[int] = []
    patient_index_dfs: list[pl.DataFrame] = []
    global_offset = 0

    for d, summary in zip(shard_dirs, summaries):
        sids = summary["subject_ids"]
        all_subject_ids.extend(sids)
        shard_patient_counts.append(len(sids))

        idx_df = pl.read_parquet(d / "patient_index.parquet")
        # Remap patient_idx to global space
        idx_df = idx_df.with_columns(
            (pl.col("patient_idx") + global_offset).alias("patient_idx")
        )
        patient_index_dfs.append(idx_df)
        global_offset += len(sids)

    n_total = len(all_subject_ids)
    logger.info(f"Total patients across all shards: {n_total}")

    # ---- Merged patient index ----
    merged_index = pl.concat(patient_index_dfs, how="diagonal")
    # Rewrite patient_idx as a clean 0..N-1 sequence (concat may have gaps if shards differ in columns)
    merged_index = merged_index.with_columns(
        pl.Series("patient_idx", list(range(n_total)))
    )
    merged_index.write_parquet(merged_dir / "patient_index.parquet")
    logger.info(f"Wrote patient_index.parquet ({n_total} rows)")
    # Sanity: do score row positions agree with merged_index row positions?
    score_order_subjects = all_subject_ids
    index_order_subjects = merged_index["subject_id"].to_list()
    mismatches = sum(1 for a, b in zip(score_order_subjects, index_order_subjects) if a != b)
    logger.info(f"Subject-ID order agreement: {len(score_order_subjects) - mismatches}/{len(score_order_subjects)}")
    # ---- AUC helper ----
    try:
        from sklearn.metrics import roc_auc_score
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    # ---- Per-outcome merge ----
    merged_outcomes: dict = {}

    for evt_name in outcome_names:
        safe_name = evt_name.replace("/", "_").replace(" ", "_")
        scores_filename = f"scores_{safe_name}.npz"

        M0_parts, M1_parts, M2_parts = [], [], []
        M0_raw_parts, M1_raw_parts, M2_raw_parts = [], [], []
        avg_m1_tokens_weighted = 0.0
        avg_m2_tokens_weighted = 0.0
        total_m2_tokens = 0
        total_m1_event_rate_weighted = 0.0
        total_true_events = 0
        total_eval_patients = 0

        for d, summary, n_pts in zip(shard_dirs, summaries, shard_patient_counts):
            scores_path = d / scores_filename
            if not scores_path.exists():
                logger.warning(f"  {d.name}: {scores_filename} missing — filling with NaN")
                M0_parts.append(np.full(n_pts, np.nan, dtype=np.float64))
                M1_parts.append(np.full(n_pts, np.nan, dtype=np.float64))
                M2_parts.append(np.full(n_pts, np.nan, dtype=np.float64))
                M0_raw_parts.append(np.full((n_pts, n_samp), np.nan, dtype=np.float32))
                M1_raw_parts.append(np.full((n_pts, n_samp), np.nan, dtype=np.float32))
                M2_raw_parts.append(np.full((n_pts, n_samp), np.nan, dtype=np.float32))
                continue

            sd = load_scores_npz(scores_path)
            M0_parts.append(sd["M0"])
            M1_parts.append(sd["M1"])
            M2_parts.append(sd["M2"])

            # Raw arrays may have different n_samp columns if shard had fewer samples
            # (e.g. due to past-flag exclusions).  Pad to uniform n_samp width.
            for raw_key, parts_list in [("M0_raw", M0_raw_parts), ("M1_raw", M1_raw_parts), ("M2_raw", M2_raw_parts)]:
                raw = sd.get(raw_key)
                if raw is None:
                    raw = np.full((n_pts, n_samp), np.nan, dtype=np.float32)
                elif raw.shape[1] < n_samp:
                    pad = np.full((raw.shape[0], n_samp - raw.shape[1]), np.nan, dtype=np.float32)
                    raw = np.hstack([raw, pad])
                elif raw.shape[1] > n_samp:
                    raw = raw[:, :n_samp]
                parts_list.append(raw.astype(np.float32))

            # Weighted averages for token counts
            if "avg_m1_tokens" in sd:
                avg_m1_tokens_weighted += float(sd["avg_m1_tokens"][0]) * n_pts
            if "avg_m2_tokens" in sd:
                avg_m2_tokens_weighted += float(sd["avg_m2_tokens"][0]) * n_pts

            # Accumulate per-outcome stats from shard summaries
            shard_outcome = summary["outcomes"].get(evt_name, {})
            total_m2_tokens += shard_outcome.get("m2_generated_tokens", 0) or 0
            if shard_outcome.get("true_n_events") is not None:
                total_true_events += shard_outcome["true_n_events"]
            m1_er = shard_outcome.get("m1_event_rate", 0.0)
            total_m1_event_rate_weighted += m1_er * n_pts
            if shard_outcome.get("true_prevalence") is not None:
                total_eval_patients += n_pts

        # Concatenate along patient axis
        M0 = np.concatenate(M0_parts)
        M1 = np.concatenate(M1_parts)
        M2 = np.concatenate(M2_parts)
        M0_raw = np.vstack(M0_raw_parts)
        M1_raw = np.vstack(M1_raw_parts)
        M2_raw = np.vstack(M2_raw_parts)

        avg_m1_tok = avg_m1_tokens_weighted / n_total if n_total > 0 else None
        avg_m2_tok = avg_m2_tokens_weighted / n_total if n_total > 0 else None

        save_merged_scores(M0, M1, M2, M0_raw, M1_raw, M2_raw,
                           avg_m1_tok, avg_m2_tok,
                           merged_dir / scores_filename)
        logger.info(f"  Wrote {scores_filename} ({n_total} patients)")

        # Recompute AUC from merged arrays
        auc_M0 = auc_M1 = auc_M2 = None
        true_prevalence = None
        future_col = f"{evt_name}_future"
        if future_col in merged_index.columns and _has_sklearn:
            outcome_arr = merged_index[future_col].to_numpy().astype(float)
            true_prevalence = float(np.nanmean(outcome_arr)) if len(outcome_arr) > 0 else None
            for est_name, est_arr in [("M0", M0), ("M1", M1), ("M2", M2)]:
                valid = ~np.isnan(est_arr)
                if valid.sum() > 0 and len(np.unique(outcome_arr[valid])) > 1:
                    auc_val = float(roc_auc_score(outcome_arr[valid], est_arr[valid]))
                    if est_name == "M0":
                        auc_M0 = auc_val
                    elif est_name == "M1":
                        auc_M1 = auc_val
                    else:
                        auc_M2 = auc_val

        # Aggregate first shard's event_id (same across shards)
        evt_id = summaries[0]["outcomes"].get(evt_name, {}).get("event_id")
        merged_outcomes[evt_name] = {
            "event_id": evt_id,
            "m1_event_rate": total_m1_event_rate_weighted / n_total if n_total > 0 else 0.0,
            "m2_generated_tokens": total_m2_tokens,
            "true_n_events": total_true_events if total_eval_patients > 0 else None,
            "true_prevalence": true_prevalence,
            "mean_M0": float(np.nanmean(M0)),
            "mean_M1": float(np.nanmean(M1)),
            "mean_M2": float(np.nanmean(M2)),
            "auc_M0": auc_M0,
            "auc_M1": auc_M1,
            "auc_M2": auc_M2,
        }

    # ---- Merged run_summary.json ----
    total_m1_tokens = sum(s.get("m1_generated_tokens", 0) for s in summaries)
    total_wall_time = sum(s.get("wall_time_seconds", 0.0) for s in summaries)

    # Determine which shard indices were included/missing
    included_shard_indices = [s.get("shard_idx") for s in summaries]
    if n_shards_total is not None:
        all_indices = set(range(n_shards_total))
        present_set = {i for i in included_shard_indices if i is not None}
        missing_indices = sorted(all_indices - present_set)
    else:
        missing_indices = []

    merged_summary = {
        "timestamp": summaries[0]["timestamp"],
        "n_patients": n_total,
        "n_samp": n_samp,
        "methods": methods,
        "score_inline": score_inline,
        "m1_generated_tokens": total_m1_tokens,
        "wall_time_seconds": total_wall_time,
        "subject_ids": all_subject_ids,
        "outcomes": merged_outcomes,
        # Merge provenance
        "merged_from_shards": included_shard_indices,
        "shards_missing": missing_indices,
        "n_shards_total": n_shards_total or n_shards_from_summary,
    }
    with open(merged_dir / "run_summary.json", "w") as f:
        json.dump(merged_summary, f, indent=2)
    logger.info("Wrote run_summary.json")

    # ---- Copy pipeline_config.yaml from first shard ----
    src_cfg = shard_dirs[0] / "pipeline_config.yaml"
    if src_cfg.exists():
        shutil.copy2(src_cfg, merged_dir / "pipeline_config.yaml")
        logger.info("Copied pipeline_config.yaml from first shard")

    # ---- Summary ----
    if missing_indices:
        logger.warning(
            f"MERGE COMPLETE with {len(missing_indices)} missing shard(s): {missing_indices}. "
            f"Results cover {n_total} of the expected patients."
        )
    else:
        logger.info(f"MERGE COMPLETE — {n_total} patients across {len(shard_dirs)} shards")
    logger.info(f"Merged output: {merged_dir}")
    logger.info("To analyze, open analysis.ipynb and set:")
    logger.info(f"  OUTPUT_DIR = pathlib.Path('{merged_dir}')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded run_timelines_shard.py outputs for analysis.ipynb.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--base-output-dir", type=str,
        help="Base output directory that contains shard_000/, shard_001/, ... subdirs. "
             "Use with --n-shards.",
    )
    group.add_argument(
        "--shard-dirs", nargs="+", type=str,
        help="Explicit list of shard directories to merge (alternative to --base-output-dir).",
    )
    parser.add_argument(
        "--n-shards", type=int, default=None,
        help="Total number of shards (used with --base-output-dir to discover shard dirs). "
             "Shards not found on disk are skipped with a warning.",
    )
    parser.add_argument(
        "--merged-dir", type=str, default=None,
        help="Output directory for merged results. "
             "Defaults to {base_output_dir}/merged.",
    )

    args = parser.parse_args()

    if args.base_output_dir is not None and args.n_shards is None:
        parser.error("--base-output-dir requires --n-shards")

    if args.base_output_dir is not None:
        base = pathlib.Path(args.base_output_dir).expanduser().resolve()
        shard_dirs, missing = discover_shard_dirs(base, args.n_shards)
        merged_dir = pathlib.Path(args.merged_dir).expanduser().resolve() if args.merged_dir else base / "merged"
    else:
        shard_dirs = [pathlib.Path(d).expanduser().resolve() for d in args.shard_dirs]
        # Filter to existing ones
        valid = []
        for d in shard_dirs:
            if d.exists() and (d / "run_summary.json").exists():
                valid.append(d)
            else:
                logger.warning(f"Shard dir missing or incomplete: {d} — skipping")
        shard_dirs = valid
        merged_dir = pathlib.Path(args.merged_dir).expanduser().resolve() if args.merged_dir else None
        if merged_dir is None:
            parser.error("--merged-dir is required when using --shard-dirs")

    if not shard_dirs:
        logger.error("No valid shard directories found — nothing to merge")
        sys.exit(1)

    merge(shard_dirs, merged_dir, n_shards_total=args.n_shards)


if __name__ == "__main__":
    main()
