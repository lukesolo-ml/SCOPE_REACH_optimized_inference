#!/usr/bin/env python3
"""SCOPE/REACH inference pipeline for cocoa-tokenized EHR timelines.

Reads the winnowed held-out data produced by cocoa's Winnower
(held_out_for_inference.parquet), which already contains:

  - tokens_past: the prompt (timeline up to the threshold)
  - tokens_future: ground truth (timeline after the threshold)
  - outcome flags: boolean columns like DSCG//expired_future

Builds a GenerationConfig from a YAML file, runs SCOPE/REACH trajectory
generation and scoring via quick_sco_re, and persists results to disk.

Scoring modes
-------------
  - two-pass (default): generate trajectories, then run a separate prefill-only
    forward pass to extract P(target_event) logprobs and compute SCOPE/REACH.
  - inline (score_inline=True): compute SCOPE/REACH during generation via
    InlineScopeReachProcessor. Requires the time-horizon processor's engine
    configuration (disable_overlap_schedule=True, enable_custom_logit_processor=True).
    Per-trajectory estimates are attached directly to GeneratedTrajectory.

Usage:
    python run_pipeline.py --config pipeline_config.yaml
    python run_pipeline.py --config pipeline_config.yaml --dry-run
"""

import argparse
import asyncio
import json
import logging
import math
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl
import yaml

from quick_sco_re import (
    GenerationConfig,
    PatientResults,
    TrajectoryType,
    create_engine,
    generate_and_score,
    generate_trajectories,
    score_trajectories,
    aggregate_results,
    save_trajectories,
    save_scores,
)
from quick_sco_re.diagnostics import log_trajectory_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)
logger = logging.getLogger("scope_reach_pipeline")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for s in ["cocoa_outputs", "model_path", "output_dir", "generation"]:
        if s not in cfg:
            raise ValueError(f"Config missing required section: '{s}'")

    for k in ["held_out_for_inference", "tokenizer_yaml"]:
        if k not in cfg["cocoa_outputs"]:
            raise ValueError(f"cocoa_outputs missing required key: '{k}'")

    for k in ["max_len", "n_samp", "target_event"]:
        if k not in cfg["generation"]:
            raise ValueError(f"generation missing required key: '{k}'")

    return cfg


# ---------------------------------------------------------------------------
# Cocoa vocabulary helper
# ---------------------------------------------------------------------------

class CocoaVocab:
    """Lightweight wrapper around a cocoa tokenizer.yaml for token ↔ id lookups."""

    def __init__(self, tokenizer_yaml_path: str | pathlib.Path):
        path = pathlib.Path(tokenizer_yaml_path).expanduser().resolve()
        with open(path) as f:
            data = yaml.safe_load(f)

        self.name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}
        for name, tid in data["lookup"].items():
            self.name_to_id[str(name)] = int(tid)
            self.id_to_name[int(tid)] = str(name)

        self.bins = data.get("bins", {})

    def __call__(self, name: str) -> int:
        return self.name_to_id.get(name, 0)

    def __contains__(self, name: str) -> bool:
        return name in self.name_to_id

    def __len__(self) -> int:
        return len(self.name_to_id)

    def ids_with_prefix(self, prefix: str) -> set[int]:
        return {
            tid for name, tid in self.name_to_id.items()
            if name.startswith(prefix)
        }

    def resolve_name(self, tid: int) -> str:
        return self.id_to_name.get(tid, "UNK")


# ---------------------------------------------------------------------------
# Data loading (from winnower output)
# ---------------------------------------------------------------------------

def load_winnowed_data(cfg: dict, vocab: CocoaVocab) -> tuple[list[list[int]], list[str], pl.DataFrame]:
    """Load prompts and metadata from the winnower's held_out_for_inference.parquet.

    The winnower has already:
      - filtered to held_out split
      - computed a threshold (duration-based or first-occurrence-based)
      - split tokens into tokens_past / tokens_future
      - added boolean outcome flags (e.g. DSCG//expired_future)

    Returns:
        patient_tokens: list of token-id lists (from tokens_past), one per patient.
        subject_ids: parallel list of subject_id strings.
        metadata_df: DataFrame with subject_id and all outcome flag columns.
    """
    cocoa = cfg["cocoa_outputs"]
    cohort = cfg.get("cohort", {})

    parquet_path = pathlib.Path(cocoa["held_out_for_inference"]).expanduser().resolve()
    logger.info(f"Loading winnowed data from {parquet_path}")
    df = pl.read_parquet(parquet_path)
    logger.info(f"Loaded {df.height} winnowed held-out timelines")

    # Identify outcome flag columns (booleans ending in _past or _future)
    flag_cols = [
        c for c in df.columns
        if (c.endswith("_past") or c.endswith("_future"))
        and df[c].dtype == pl.Boolean
    ]
    logger.info(f"Outcome flag columns: {flag_cols}")

    # Optional flag-based filters
    require_flag = cohort.get("require_flag")
    if require_flag:
        if require_flag not in df.columns:
            logger.warning(f"require_flag '{require_flag}' not in columns — skipping")
        else:
            df = df.filter(pl.col(require_flag))
            logger.info(f"After require_flag '{require_flag}': {df.height} patients")

    exclude_flag = cohort.get("exclude_flag")
    if exclude_flag:
        if exclude_flag not in df.columns:
            logger.warning(f"exclude_flag '{exclude_flag}' not in columns — skipping")
        else:
            df = df.filter(~pl.col(exclude_flag))
            logger.info(f"After exclude_flag '{exclude_flag}': {df.height} patients")

    # Optional subsample
    max_patients = cohort.get("max_patients")
    if max_patients is not None and df.height > max_patients:
        seed = cohort.get("shuffle_seed", 42)
        df = df.sample(n=max_patients, shuffle=True, seed=seed)
        logger.info(f"Subsampled to {df.height} patients (seed={seed})")

    # Extract prompts from tokens_past
    subject_ids = df.select("subject_id").to_series().to_list()
    patient_tokens_raw = df.select("tokens_past").to_series().to_list()

    # Handle prompt overflow
    max_len = cfg["generation"]["max_len"]
    overflow = cfg.get("prompt_overflow", "truncate_left")
    n_dropped = 0
    n_truncated = 0
    final_tokens = []
    final_ids = []
    keep_mask = []

    for i, (sid, toks) in enumerate(zip(subject_ids, patient_tokens_raw)):
        if len(toks) > max_len:
            if overflow == "drop":
                n_dropped += 1
                keep_mask.append(False)
                continue
            elif overflow == "truncate_left":
                toks = toks[-max_len:]
                n_truncated += 1
        final_tokens.append(toks)
        final_ids.append(sid)
        keep_mask.append(True)

    if n_dropped:
        logger.warning(f"Dropped {n_dropped} patients with prompts > {max_len} tokens")
    if n_truncated:
        logger.info(f"Left-truncated {n_truncated} prompts to {max_len} tokens")

    # Filter metadata to match
    metadata_df = df.filter(pl.Series(keep_mask)).select(
        "subject_id", *flag_cols
    )

    logger.info(f"Final cohort: {len(final_tokens)} patients")
    lengths = [len(t) for t in final_tokens]
    if lengths:
        logger.info(
            f"Prompt lengths — min: {min(lengths)}, "
            f"median: {sorted(lengths)[len(lengths)//2]}, "
            f"max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.0f}"
        )

    # Log outcome prevalence
    for col in flag_cols:
        rate = metadata_df[col].mean()
        logger.info(f"  {col}: {rate:.3f}")

    return final_tokens, final_ids, metadata_df


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

def build_generation_config(cfg: dict, vocab: CocoaVocab) -> tuple[GenerationConfig, bool]:
    """Build a GenerationConfig from YAML.

    Returns:
        (config, score_inline) — score_inline is a runtime flag (not stored on
        GenerationConfig) that drives the generate-vs-generate_and_score branch
        in run_pipeline.
    """
    gen = cfg["generation"]

    # Target event
    target_name = gen["target_event"]
    target_id = vocab(target_name)
    if target_id == 0 and target_name != "UNK":
        raise ValueError(f"target_event '{target_name}' not found in vocabulary")
    logger.info(f"Target event: '{target_name}' → token ID {target_id}")

    # End tokens
    end_ids: set[int] = set()
    end_cfg = gen.get("end_tokens", {})
    for prefix in end_cfg.get("prefixes", []):
        ids = vocab.ids_with_prefix(prefix)
        logger.info(f"End token prefix '{prefix}' matched {len(ids)} tokens")
        end_ids |= ids
    for name in end_cfg.get("names", []):
        tid = vocab(name)
        if tid != 0 or name == "UNK":
            end_ids.add(tid)
    logger.info(f"Total end token IDs: {len(end_ids)}")

    # Suppressed tokens
    suppressed_names = gen.get("suppressed_tokens", [])
    suppressed_ids = []
    for name in suppressed_names:
        tid = vocab(name)
        if tid != 0 or name == "UNK":
            suppressed_ids.append(tid)
        else:
            logger.warning(f"Suppressed token '{name}' not in vocab — skipping")
    logger.info(f"Suppressed token IDs: {suppressed_ids}")

    # Time stopping
    ts = gen.get("time_stopping") or {}
    trunc_id = None
    token_id_to_minutes: dict[int, float] = {}
    max_time = None
    time_check_interval = 100

    if ts.get("enabled", False):
        trunc_name = ts.get("trunc_token", "TRUNC")
        trunc_id = vocab(trunc_name)
        if trunc_id == 0 and trunc_name != "UNK":
            raise ValueError(f"trunc_token '{trunc_name}' not in vocabulary")

        max_time = ts.get("max_time_minutes")
        time_check_interval = ts.get("time_check_interval", 100)

        for tok_name, bounds in ts.get("time_token_bounds", {}).items():
            tid = vocab(tok_name)
            if tid != 0 or tok_name == "UNK":
                lo, hi = bounds
                token_id_to_minutes[tid] = math.sqrt(lo * hi)

        if trunc_id in suppressed_ids:
            suppressed_ids.remove(trunc_id)
            logger.info(f"Removed trunc_id {trunc_id} from suppressed_ids (handled internally)")

        logger.info(
            f"Time stopping: trunc='{trunc_name}'(id={trunc_id}), "
            f"max_time={max_time} min, check_interval={time_check_interval}, "
            f"{len(token_id_to_minutes)} time tokens mapped"
        )
    else:
        logger.info("Time-based stopping: DISABLED")

    # Scoring mode
    score_inline = gen.get("score_inline", False)

    # Inline tracked tokens. When score_inline is set, default to tracking just
    # the target event. Users can override via generation.tracked_tokens (list
    # of vocab names).
    tracked_ids: list[int] | None = None
    tracked_name: str | None = None
    if score_inline:
        tracked_token_names = gen.get("tracked_tokens")
        if tracked_token_names:
            tracked_ids = []
            for name in tracked_token_names:
                tid = vocab(name)
                if tid != 0 or name == "UNK":
                    tracked_ids.append(tid)
                else:
                    logger.warning(f"Tracked token '{name}' not in vocab — skipping")
            tracked_name = gen.get("tracked_name", "custom")
        else:
            tracked_ids = [target_id]
            tracked_name = target_name

        logger.info(
            f"Scoring mode: INLINE — tracking {len(tracked_ids)} token(s) "
            f"(name='{tracked_name}', ids={tracked_ids[:10]}{'...' if len(tracked_ids) > 10 else ''})"
        )
        logger.info(
            "Inline scoring requires engine with disable_overlap_schedule=True "
            "and enable_custom_logit_processor=True."
        )
    else:
        logger.info("Scoring mode: TWO-PASS (separate prefill scoring pass)")

    config = GenerationConfig(
        max_len=gen["max_len"],
        n_samp=gen["n_samp"],
        target_event_id=target_id,
        end_token_ids=end_ids,
        suppressed_ids=suppressed_ids,
        temperature=gen.get("temperature", 1.0),
        trunc_id=trunc_id,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
        time_check_interval=time_check_interval,
        tracked_ids=tracked_ids,
        tracked_name=tracked_name,
    )
    return config, score_inline


# ---------------------------------------------------------------------------
# Inline aggregation
# ---------------------------------------------------------------------------

def aggregate_inline_results(
    trajectories: list,
    num_patients: int,
    config: GenerationConfig,
) -> list[PatientResults]:
    """Build per-patient results from inline SCOPE/REACH estimates on trajectories.

    Looks up the target_event_id in each trajectory's inline_tracked_ids to
    locate the SCOPE/REACH index. M0 is derived from timeline_terminating_id.
    """
    target_id = config.target_event_id
    results = {i: PatientResults() for i in range(num_patients)}

    for traj in trajectories:
        if traj.inline_tracked_ids is None:
            logger.warning(
                f"Trajectory (patient={traj.patient_idx}, sample={traj.sample_idx}) "
                f"missing inline estimates — skipping"
            )
            continue
        try:
            k = traj.inline_tracked_ids.index(target_id)
        except ValueError:
            logger.warning(
                f"target_event_id {target_id} not in tracked_ids for traj "
                f"(patient={traj.patient_idx}) — skipping"
            )
            continue

        if traj.traj_type == TrajectoryType.M1:
            results[traj.patient_idx].m0_samples.append(
                traj.timeline_terminating_id == target_id
            )
            scope = float(traj.scope_estimates[k]) if traj.scope_estimates is not None else 0.0
            results[traj.patient_idx].m1_samples.append(scope)
        else:
            reach = float(traj.reach_estimates[k]) if traj.reach_estimates is not None else 0.0
            results[traj.patient_idx].m2_samples.append(reach)

    return [results[i] for i in range(num_patients)]


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

async def run_pipeline(cfg: dict):
    start_time = time.time()

    vocab = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    logger.info(f"Loaded vocabulary: {len(vocab)} tokens")

    patient_tokens, subject_ids, metadata_df = load_winnowed_data(cfg, vocab)
    if not patient_tokens:
        logger.error("No patients to process — exiting")
        return

    gen_config, score_inline = build_generation_config(cfg, vocab)
    methods = cfg["generation"].get("methods", ["M1", "M2"])

    output_dir = pathlib.Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    save_cfg = cfg.get("save", {})

    if save_cfg.get("config_snapshot", True):
        with open(output_dir / "pipeline_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Save patient index with outcome flags (for downstream AUC, etc.)
    if save_cfg.get("patient_index", True):
        index_df = pl.DataFrame({
            "patient_idx": list(range(len(subject_ids))),
            "subject_id": subject_ids,
            "prompt_length": [len(t) for t in patient_tokens],
        }).join(metadata_df, on="subject_id", how="left")
        index_df.write_parquet(output_dir / "patient_index.parquet")
        logger.info(f"Saved patient index ({len(subject_ids)} patients) with outcome flags")

    # Engine configuration. Both time-horizon and inline SCOPE/REACH require
    # the custom-logit-processor path, which needs disable_overlap_schedule=True.
    use_time = gen_config.max_time is not None and gen_config.trunc_id is not None
    needs_processor = use_time or score_inline
    mem_frac = cfg.get("engine", {}).get("mem_fraction", 0.85)

    logger.info(
        f"Creating SGLang engine (model={cfg['model_path']}, "
        f"custom_processor={needs_processor}, time_horizon={use_time}, score_inline={score_inline})"
    )
    engine = create_engine(
        model_path=str(pathlib.Path(cfg["model_path"]).expanduser().resolve()),
        max_len=gen_config.max_len,
        use_time_horizon=needs_processor,
        mem_fraction=mem_frac,
    )

    try:
        # Warmup
        logger.info("Running warmup pass...")
        warmup_tokens = patient_tokens[:min(2, len(patient_tokens))]
        if score_inline:
            await generate_trajectories(engine, gen_config, warmup_tokens, methods)
        else:
            await generate_and_score(
                engine, gen_config, warmup_tokens,
                target_token_id=gen_config.target_event_id,
                methods=methods,
            )
        logger.info("Warmup complete")

        # Main generation + scoring (chunked by patient)
        chunk_size = cfg.get("engine", {}).get("patient_chunk_size", 64)
        n_traj = len(patient_tokens) * gen_config.n_samp * len(methods)
        logger.info(
            f"Starting generation: {len(patient_tokens)} patients × "
            f"{gen_config.n_samp} samples × {len(methods)} methods = "
            f"{n_traj} trajectories (chunk_size={chunk_size})"
        )

        gen_start = time.time()
        trajectories = []
        results = []
        n_patients = len(patient_tokens)

        for chunk_start in range(0, n_patients, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_patients)
            chunk_tokens = patient_tokens[chunk_start:chunk_end]
            chunk_n_traj = len(chunk_tokens) * gen_config.n_samp * len(methods)
            logger.info(
                f"Chunk [{chunk_start}:{chunk_end}] — "
                f"{len(chunk_tokens)} patients, {chunk_n_traj} trajectories"
            )

            if score_inline:
                chunk_traj = await generate_trajectories(
                    engine, gen_config, chunk_tokens, methods
                )
                # Aggregate with chunk-local patient_idx (matches two-pass path)
                chunk_results = aggregate_inline_results(
                    chunk_traj, num_patients=len(chunk_tokens), config=gen_config
                )
            else:
                chunk_traj, chunk_results = await generate_and_score(
                    engine, gen_config, chunk_tokens,
                    target_token_id=gen_config.target_event_id,
                    methods=methods,
                )

            trajectories.extend(chunk_traj)
            results.extend(chunk_results)

            elapsed_so_far = time.time() - gen_start
            done = chunk_end
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (n_patients - done) / rate if rate > 0 else float("inf")
            logger.info(
                f"  {done}/{n_patients} patients done "
                f"({elapsed_so_far:.1f}s elapsed, ~{eta:.0f}s remaining)"
            )

        gen_elapsed = time.time() - gen_start
        logger.info(f"Generation + scoring completed in {gen_elapsed:.1f}s")

        total_gen_tokens = sum(len(t.output_ids) for t in trajectories)
        logger.info(f"Total generated tokens: {total_gen_tokens:,}")
        if gen_elapsed > 0:
            logger.info(f"Throughput: {total_gen_tokens / gen_elapsed:,.0f} tok/s")

        log_trajectory_diagnostics(trajectories, gen_config, "main", logger)

        # Summary statistics
        M0 = np.array([np.mean(r.m0_samples) if r.m0_samples else np.nan for r in results])
        M1 = np.array([np.mean(r.m1_samples) if r.m1_samples else np.nan for r in results])
        M2 = np.array([np.mean(r.m2_samples) if r.m2_samples else np.nan for r in results])

        for name, arr in [("M0 (MC)", M0), ("M1 (SCOPE)", M1), ("M2 (REACH)", M2)]:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                logger.info(
                    f"{name}: mean={np.nanmean(arr):.4f}, "
                    f"std={np.nanstd(arr):.4f}, "
                    f"median={np.nanmedian(arr):.4f}, "
                    f"[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]"
                )

        # Compute AUCs against winnower outcome flags
        target_event_name = cfg["generation"]["target_event"]
        future_col = f"{target_event_name}_future"
        if future_col in metadata_df.columns:
            outcome = metadata_df[future_col].to_numpy().astype(float)
            try:
                from sklearn.metrics import roc_auc_score
                for est_name, est_arr in [("M0", M0), ("M1", M1), ("M2", M2)]:
                    valid_mask = ~np.isnan(est_arr)
                    if valid_mask.sum() > 0 and len(np.unique(outcome[valid_mask])) > 1:
                        auc = roc_auc_score(outcome[valid_mask], est_arr[valid_mask])
                        logger.info(f"AUC {est_name} vs {future_col}: {auc:.4f}")
                    else:
                        logger.info(f"AUC {est_name} vs {future_col}: N/A (single class)")
            except ImportError:
                logger.info("sklearn not available — skipping AUC computation")
        else:
            logger.info(f"No '{future_col}' column found — skipping AUC computation")

        # Save results
        if save_cfg.get("trajectories", True):
            traj_dir = output_dir / "trajectories"
            save_trajectories(trajectories, traj_dir, config=gen_config)
            logger.info(f"Saved trajectories to {traj_dir}")

        if save_cfg.get("scores", True):
            scores_path = output_dir / "scores.npz"
            save_scores(results, scores_path)
            logger.info(f"Saved scores to {scores_path}")

        # Human-readable summary
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_patients": len(patient_tokens),
            "n_samp": gen_config.n_samp,
            "methods": methods,
            "score_inline": score_inline,
            "target_event": target_event_name,
            "target_event_id": gen_config.target_event_id,
            "tracked_name": gen_config.tracked_name,
            "tracked_ids_len": len(gen_config.tracked_ids) if gen_config.tracked_ids else 0,
            "wall_time_seconds": gen_elapsed,
            "total_generated_tokens": total_gen_tokens,
            "mean_M0": float(np.nanmean(M0)),
            "mean_M1": float(np.nanmean(M1)),
            "mean_M2": float(np.nanmean(M2)),
        }
        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    finally:
        engine.shutdown()
        logger.info("Engine shut down")

    total_elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def dry_run(cfg: dict):
    logger.info("=== DRY RUN ===")

    vocab = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    logger.info(f"Vocabulary: {len(vocab)} tokens")

    patient_tokens, subject_ids, metadata_df = load_winnowed_data(cfg, vocab)
    gen_config, score_inline = build_generation_config(cfg, vocab)
    methods = cfg["generation"].get("methods", ["M1", "M2"])

    n_traj = len(patient_tokens) * gen_config.n_samp * len(methods)
    logger.info(f"Would generate {n_traj} trajectories for {len(patient_tokens)} patients")

    logger.info("--- Resolved token mappings ---")
    logger.info(f"  target_event_id: {gen_config.target_event_id} "
                f"({vocab.resolve_name(gen_config.target_event_id)})")
    logger.info(f"  end_token_ids ({len(gen_config.end_token_ids)}): "
                f"{sorted(gen_config.end_token_ids)[:10]}...")
    logger.info(f"  suppressed_ids: {gen_config.suppressed_ids}")
    if gen_config.trunc_id is not None:
        logger.info(f"  trunc_id: {gen_config.trunc_id} ({vocab.resolve_name(gen_config.trunc_id)})")
    logger.info(f"  time tokens mapped: {len(gen_config.token_id_to_minutes)}")
    logger.info(f"  score_inline: {score_inline}")
    if gen_config.tracked_ids:
        logger.info(
            f"  tracked_ids ({len(gen_config.tracked_ids)}): "
            f"{gen_config.tracked_ids[:10]}{'...' if len(gen_config.tracked_ids) > 10 else ''}"
        )
        logger.info(f"  tracked_name: {gen_config.tracked_name}")

    # Show outcome flag prevalence
    flag_cols = [c for c in metadata_df.columns if c != "subject_id"]
    if flag_cols:
        logger.info("--- Outcome prevalence ---")
        for col in flag_cols:
            rate = metadata_df[col].mean()
            logger.info(f"  {col}: {rate:.3f}")

    logger.info("=== DRY RUN COMPLETE — config is valid ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SCOPE/REACH inference on cocoa winnowed held-out timelines.",
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to pipeline YAML config file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and data without running inference.")
    parser.add_argument("--n-samp", type=int, default=None,
                        help="Override generation.n_samp")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Override cohort.max_patients")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output_dir")
    parser.add_argument("--score-inline", action="store_true", default=None,
                        help="Enable single-pass inline scoring (override generation.score_inline)")
    parser.add_argument("--no-score-inline", action="store_true", default=None,
                        help="Force two-pass scoring (override generation.score_inline)")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.n_samp is not None:
        cfg["generation"]["n_samp"] = args.n_samp
    if args.max_patients is not None:
        cfg.setdefault("cohort", {})["max_patients"] = args.max_patients
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.score_inline:
        cfg["generation"]["score_inline"] = True
    elif args.no_score_inline:
        cfg["generation"]["score_inline"] = False

    if args.dry_run:
        dry_run(cfg)
    else:
        asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
