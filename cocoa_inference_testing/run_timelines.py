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
    token_ids_logprob. Per-trajectory estimates are attached directly to
    GeneratedTrajectory.

Usage:
    python run_pipeline.py --config pipeline_config.yaml
    python run_pipeline.py --config pipeline_config.yaml --dry-run
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import pathlib
import re
import sys
import threading
import time
import warnings
from datetime import datetime, timezone

# Python 3.12 resource_tracker race condition — harmless, from SGLang's loky subprocess pool
warnings.filterwarnings("ignore", message="resource_tracker: process died unexpectedly")


def _suppress_resource_tracker_stderr() -> None:
    """Redirect fd 2 through a filter thread to drop resource_tracker KeyError noise.

    Python 3.12 has a race in the multiprocessing resource_tracker daemon (triggered
    by loky/SGLang) that prints spurious KeyError tracebacks to stderr. These originate
    in a forked subprocess so warnings.filterwarnings cannot intercept them — they
    write directly to fd 2. We replace fd 2 with a pipe and drain it on a daemon
    thread, blocking any line that matches the known noise pattern.
    """
    _NOISE = re.compile(
        r"resource_tracker\.py|cache\[rtype\]\.remove\(name\)|KeyError: '/loky-"
    )

    r_fd, w_fd = os.pipe()
    real_fd = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)

    real_stderr = os.fdopen(real_fd, "w", buffering=1)
    sys.stderr = real_stderr

    def _run() -> None:
        buf: list[str] = []
        with os.fdopen(r_fd, "r", buffering=1, errors="replace") as pipe:
            for line in pipe:
                if line.rstrip() == "Traceback (most recent call last):":
                    buf = [line]
                elif buf:
                    buf.append(line)
                    if not (line.startswith("  ") or line.startswith("\t")):
                        if not any(_NOISE.search(l) for l in buf):
                            real_stderr.writelines(buf)
                            real_stderr.flush()
                        buf = []
                elif not _NOISE.search(line):
                    real_stderr.write(line)
                    real_stderr.flush()

    threading.Thread(target=_run, daemon=True, name="stderr-filter").start()

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from quick_sco_re import (
    GenerationConfig,
    PatientResults,
    TrajectoryType,
    create_engine,
    generate_and_score,
    generate_trajectories,
    generate_m2_from_m1_trajectories,
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
logging.getLogger("sglang").setLevel(logging.WARNING)
logger = logging.getLogger("scope_reach_pipeline")
_suppress_resource_tracker_stderr()


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

    for k in ["max_len", "n_samp"]:
        if k not in cfg["generation"]:
            raise ValueError(f"generation missing required key: '{k}'")
    if "tracked_events" not in cfg["generation"] and "target_event" not in cfg["generation"]:
        raise ValueError("generation config must specify 'tracked_events' (list) or 'target_event' (single)")

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

    # Drop discharged patients: any DSCG// prefix token in tokens_past
    dscg_ids = list(vocab.ids_with_prefix("DSCG"))
    if dscg_ids:
        before = df.height
        df = df.filter(
            ~pl.col("tokens_past").list.eval(pl.element().is_in(dscg_ids)).list.any()
        )
        n_discharged = before - df.height
        logger.info(
            f"Dropped {n_discharged} discharged patients (DSCG token in timeline); "
            f"{df.height} remaining"
        )
    else:
        logger.warning("No DSCG tokens found in vocabulary — discharge filter skipped")

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

    # Tracked events — support both new 'tracked_events' list and legacy 'target_event'
    if "tracked_events" in gen:
        event_names = list(gen["tracked_events"])
        if not event_names:
            raise ValueError("tracked_events cannot be empty")
    else:
        event_names = [gen["target_event"]]

    tracked_event_ids = []
    for name in event_names:
        tid = vocab(name)
        if tid == 0 and name != "UNK":
            raise ValueError(f"tracked_event '{name}' not found in vocabulary")
        tracked_event_ids.append(tid)
        logger.info(f"Tracked event: '{name}' → token ID {tid}")

    primary_id = tracked_event_ids[0]

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

    score_inline = gen.get("score_inline", False)

    # tracked_ids only set for inline scoring (controls inline logprob requests in generation)
    inline_tracked_ids = tracked_event_ids if score_inline else None

    if score_inline:
        logger.info(
            f"Scoring mode: INLINE — tracking {len(tracked_event_ids)} event(s): "
            f"{event_names}"
        )
    else:
        if len(event_names) > 1:
            logger.warning(
                "Two-pass scoring mode does not support multiple tracked events. "
                "Only the first event will be scored."
            )
        logger.info("Scoring mode: TWO-PASS (separate prefill scoring pass)")

    config = GenerationConfig(
        max_len=gen["max_len"],
        n_samp=gen["n_samp"],
        target_event_id=primary_id,
        end_token_ids=end_ids,
        suppressed_ids=suppressed_ids,
        temperature=gen.get("temperature", 1.0),
        trunc_id=trunc_id,
        token_id_to_minutes=token_id_to_minutes,
        max_time=max_time,
        time_check_interval=time_check_interval,
        tracked_ids=inline_tracked_ids,
        tracked_names=event_names,
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

    use_time = gen_config.max_time is not None and gen_config.trunc_id is not None
    needs_processor = use_time
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
            warmup_m1 = await generate_trajectories(engine, gen_config, warmup_tokens, ["M1"])
            if "M2" in methods and gen_config.tracked_ids:
                first_outcome_config = dataclasses.replace(gen_config, target_event_id=gen_config.tracked_ids[0])
                warmup_m2 = await generate_m2_from_m1_trajectories(engine, first_outcome_config, warmup_m1, warmup_tokens)
                await score_trajectories(engine, first_outcome_config, warmup_m2, warmup_tokens)
        else:
            await generate_and_score(
                engine, gen_config, warmup_tokens,
                target_token_id=gen_config.target_event_id,
                methods=methods,
            )
        logger.info("Warmup complete")

        # Main generation + scoring (chunked by patient)
        chunk_size = cfg.get("engine", {}).get("patient_chunk_size", 64)
        tracked_names = gen_config.tracked_names or []
        tracked_ids_list = gen_config.tracked_ids or []
        n_outcomes = len(tracked_names)

        logger.info(
            f"Starting generation: {len(patient_tokens)} patients × "
            f"{gen_config.n_samp} samples × {n_outcomes} outcome(s)"
        )

        n_patients = len(patient_tokens)

        # Per-outcome boolean mask: True means patient already had the event in the past
        # and should be excluded from that outcome's evaluation and M2 generation.
        outcome_past_masks: list[np.ndarray] = []
        for evt_name in tracked_names:
            past_col = f"{evt_name}_past"
            if past_col in metadata_df.columns:
                mask = metadata_df[past_col].to_numpy().astype(bool)
            else:
                mask = np.zeros(n_patients, dtype=bool)
            outcome_past_masks.append(mask)
            n_excl = int(mask.sum())
            if n_excl:
                logger.info(
                    f"  Outcome '{evt_name}': excluding {n_excl} / {n_patients} patients "
                    f"with past flag '{past_col}'"
                )

        gen_start = time.time()
        m1_trajectories: list = []
        all_results: list[list[PatientResults]] = [[] for _ in range(n_outcomes)]
        per_outcome_m2_tokens: list[int] = [0] * n_outcomes
        per_outcome_m2_count: list[int] = [0] * n_outcomes

        with logging_redirect_tqdm():
            with tqdm(total=n_patients, desc="Generating", unit="pt", dynamic_ncols=True) as pbar:
                for chunk_start in range(0, n_patients, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_patients)
                    chunk_tokens = patient_tokens[chunk_start:chunk_end]

                    if score_inline:
                        chunk_m1 = await generate_trajectories(engine, gen_config, chunk_tokens, ["M1"])
                        m1_trajectories.extend(chunk_m1)

                        outcome_configs = [
                            dataclasses.replace(gen_config, target_event_id=evt_id)
                            for evt_id in tracked_ids_list
                        ]
                        outcome_m1_results_list = [
                            aggregate_inline_results(chunk_m1, len(chunk_tokens), oc)
                            for oc in outcome_configs
                        ]

                        if "M2" in methods:
                            # For non-event M1 trajectories: inline REACH is already available —
                            # no scoring pass needed. Only regenerate where the event occurred.
                            regen_per_outcome: list[list] = [[] for _ in tracked_ids_list]
                            for k, (evt_id, om1r) in enumerate(
                                zip(tracked_ids_list, outcome_m1_results_list)
                            ):
                                past_mask_k = outcome_past_masks[k]
                                for traj in chunk_m1:
                                    if past_mask_k[chunk_start + traj.patient_idx]:
                                        continue
                                    if traj.timeline_terminating_id == evt_id:
                                        regen_per_outcome[k].append(traj)
                                    elif (
                                        traj.inline_tracked_ids is not None
                                        and traj.reach_estimates is not None
                                    ):
                                        try:
                                            ki = traj.inline_tracked_ids.index(evt_id)
                                            om1r[traj.patient_idx].m2_samples.append(
                                                float(traj.reach_estimates[ki])
                                            )
                                        except (ValueError, IndexError):
                                            pass

                            # Regenerate M2 for all outcomes in parallel
                            regen_m2_lists = await asyncio.gather(*[
                                generate_m2_from_m1_trajectories(engine, oc, regen, chunk_tokens)
                                for oc, regen in zip(outcome_configs, regen_per_outcome)
                            ])

                            # Score all regenerated M2s in parallel across outcomes
                            scored_regen_lists = await asyncio.gather(*[
                                score_trajectories(engine, oc, regen_m2, chunk_tokens)
                                for oc, regen_m2 in zip(outcome_configs, regen_m2_lists)
                            ])

                            for k, (regen_m2, scored_regen) in enumerate(
                                zip(regen_m2_lists, scored_regen_lists)
                            ):
                                per_outcome_m2_tokens[k] += sum(
                                    t.n_new_tokens or 0 for t in regen_m2
                                )
                                per_outcome_m2_count[k] += len(regen_m2)
                                om1r = outcome_m1_results_list[k]
                                for st in scored_regen:
                                    om1r[st.trajectory.patient_idx].m2_samples.append(st.score)

                        for k, om1r in enumerate(outcome_m1_results_list):
                            all_results[k].extend(om1r)

                    else:
                        chunk_traj, chunk_results = await generate_and_score(
                            engine, gen_config, chunk_tokens,
                            target_token_id=gen_config.target_event_id,
                            methods=methods,
                        )
                        chunk_m2 = [t for t in chunk_traj if t.traj_type == TrajectoryType.M2]
                        m1_trajectories.extend(t for t in chunk_traj if t.traj_type == TrajectoryType.M1)
                        per_outcome_m2_tokens[0] += sum(len(t.output_ids) for t in chunk_m2)
                        per_outcome_m2_count[0] += len(chunk_m2)
                        all_results[0].extend(chunk_results)

                    pbar.update(len(chunk_tokens))

        gen_elapsed = time.time() - gen_start
        logger.info(f"Generation + scoring completed in {gen_elapsed:.1f}s")

        # Zero out results for past-flagged patients so they produce NaN scores
        # and are excluded from AUC/Brier computation for that outcome.
        for k in range(n_outcomes):
            for i, is_past in enumerate(outcome_past_masks[k]):
                if is_past:
                    all_results[k][i] = PatientResults()

        m1_tokens = sum(len(t.output_ids) for t in m1_trajectories)
        avg_m1_tokens = m1_tokens / len(m1_trajectories) if m1_trajectories else 0.0
        total_m2_tokens = sum(per_outcome_m2_tokens)
        total_gen_tokens = m1_tokens + total_m2_tokens
        logger.info(f"Generated tokens — M1: {m1_tokens:,}  M2 total: {total_m2_tokens:,}  overall: {total_gen_tokens:,}")
        if gen_elapsed > 0:
            logger.info(f"Throughput: {total_gen_tokens / gen_elapsed:,.0f} tok/s")

        log_trajectory_diagnostics(m1_trajectories, gen_config, "main", logger)

        # Per-outcome statistics and AUC
        outcomes_summary: dict = {}
        try:
            from sklearn.metrics import roc_auc_score
            _has_sklearn = True
        except ImportError:
            _has_sklearn = False
            logger.info("sklearn not available — skipping AUC computation")

        for k, evt_name in enumerate(tracked_names):
            evt_id = tracked_ids_list[k] if tracked_ids_list else gen_config.target_event_id
            results_k = all_results[k]

            M0_k = np.array([np.mean(r.m0_samples) if r.m0_samples else np.nan for r in results_k])
            M1_k = np.array([np.mean(r.m1_samples) if r.m1_samples else np.nan for r in results_k])
            M2_k = np.array([np.mean(r.m2_samples) if r.m2_samples else np.nan for r in results_k])

            m1_with_event = sum(1 for t in m1_trajectories if t.timeline_terminating_id == evt_id)
            m1_total = len(m1_trajectories)

            logger.info(f"=== Outcome: {evt_name} ===")
            logger.info(f"  M1 event rate: {m1_with_event:,} / {m1_total:,} ({m1_with_event / m1_total:.1%})")
            for est_name, arr in [("M0", M0_k), ("M1 SCOPE", M1_k), ("M2 REACH", M2_k)]:
                if not np.all(np.isnan(arr)):
                    logger.info(
                        f"  {est_name}: mean={np.nanmean(arr):.4f}  "
                        f"std={np.nanstd(arr):.4f}  median={np.nanmedian(arr):.4f}"
                    )

            future_col = f"{evt_name}_future"
            true_n_events = None
            true_prevalence = None
            auc_M0 = auc_M1 = auc_M2 = None
            if future_col in metadata_df.columns:
                outcome_arr = metadata_df[future_col].to_numpy().astype(float)
                eval_mask = ~outcome_past_masks[k]
                eval_outcome_arr = outcome_arr[eval_mask]
                true_n_events = int(eval_outcome_arr.sum())
                true_prevalence = float(true_n_events / len(eval_outcome_arr)) if len(eval_outcome_arr) > 0 else 0.0
                logger.info(f"  True prevalence: {true_n_events} / {len(outcome_arr)} ({true_prevalence:.1%})")
                if _has_sklearn:
                    _aucs: dict = {}
                    for est_name, est_arr in [("M0", M0_k), ("M1", M1_k), ("M2", M2_k)]:
                        valid_mask = ~np.isnan(est_arr)
                        if valid_mask.sum() > 0 and len(np.unique(outcome_arr[valid_mask])) > 1:
                            _aucs[est_name] = float(roc_auc_score(outcome_arr[valid_mask], est_arr[valid_mask]))
                            logger.info(f"  AUC {est_name}: {_aucs[est_name]:.4f}")
                        else:
                            logger.info(f"  AUC {est_name}: N/A (single class)")
                    auc_M0 = _aucs.get("M0")
                    auc_M1 = _aucs.get("M1")
                    auc_M2 = _aucs.get("M2")
            else:
                logger.info(f"  No '{future_col}' column found — skipping AUC")

            outcomes_summary[evt_name] = {
                "event_id": evt_id,
                "m1_event_rate": m1_with_event / m1_total if m1_total > 0 else 0.0,
                "m2_generated_tokens": per_outcome_m2_tokens[k],
                "true_n_events": true_n_events,
                "true_prevalence": true_prevalence,
                "mean_M0": float(np.nanmean(M0_k)),
                "mean_M1": float(np.nanmean(M1_k)),
                "mean_M2": float(np.nanmean(M2_k)),
                "auc_M0": auc_M0,
                "auc_M1": auc_M1,
                "auc_M2": auc_M2,
            }

            if save_cfg.get("scores", True):
                safe_name = evt_name.replace("/", "_").replace(" ", "_")
                scores_path = output_dir / f"scores_{safe_name}.npz"
                avg_m2_toks = (
                    per_outcome_m2_tokens[k] / per_outcome_m2_count[k]
                    if per_outcome_m2_count[k] > 0 else 0.0
                )
                save_scores(
                    results_k, scores_path,
                    avg_m1_tokens=avg_m1_tokens,
                    avg_m2_tokens=avg_m2_toks,
                )
                logger.info(f"  Saved scores → {scores_path.name}")

        # Save M1 trajectories
        if save_cfg.get("trajectories", True):
            traj_dir = output_dir / "trajectories"
            save_trajectories(m1_trajectories, traj_dir, config=gen_config)
            logger.info(f"Saved M1 trajectories to {traj_dir}")

        # Human-readable summary
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_patients": len(patient_tokens),
            "n_samp": gen_config.n_samp,
            "methods": methods,
            "score_inline": score_inline,
            "m1_generated_tokens": m1_tokens,
            "wall_time_seconds": gen_elapsed,
            "subject_ids": subject_ids,
            "outcomes": outcomes_summary,
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
        logger.info(f"  tracked_names: {gen_config.tracked_names}")

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
