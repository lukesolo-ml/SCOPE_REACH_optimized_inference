#!/usr/bin/env python3
"""
Rescue script for early-terminated M1 trajectories.

The bug in generation.py adds all tracked_ids as stop tokens for M1 generation.
This causes M1 to terminate when any tracked outcome occurs rather than running
to the natural end of the timeline, underestimating outcome probabilities.

Usage:
    python rescue.py --output-dir ./scope_reach_output
                     --rescue-dir ./scope_reach_output_rescue
                     --config ./pipeline_config.yaml
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import pathlib
import shutil
import sys
import time
from collections import Counter

import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("rescue")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from quick_sco_re.generation import generate_trajectory, generate_m2_from_m1_trajectory
from quick_sco_re.io import save_scores
from quick_sco_re.scheduler import create_engine
from quick_sco_re.scoring import score_trajectory
from quick_sco_re.structures import GeneratedTrajectory, GenerationConfig, PatientResults, TrajectoryType

BATCH_SIZE = 1_000


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(traj_dir: pathlib.Path) -> GenerationConfig | None:
    config_path = traj_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        d = json.load(f)
    if "token_id_to_minutes" in d:
        d["token_id_to_minutes"] = {int(k): v for k, v in d["token_id_to_minutes"].items()}
    if "end_token_ids" in d:
        d["end_token_ids"] = set(d["end_token_ids"])
    return GenerationConfig(**d)


# ---------------------------------------------------------------------------
# Partial trajectory loading from npz flat arrays
# ---------------------------------------------------------------------------

def _extract_trajectory_at(data, i: int) -> GeneratedTrajectory:
    """Reconstruct one GeneratedTrajectory from npz flat arrays at position i."""
    start = int(data["output_ids_offsets"][i])
    end   = int(data["output_ids_offsets"][i + 1])
    output_ids = data["output_ids_flat"][start:end].tolist()

    scope_estimates = reach_estimates = occurred_flag = occurred_index = None
    inline_tracked_ids = inline_tracked_name = None

    if "scope_flat" in data:
        s0 = int(data["scope_offsets"][i]);  s1 = int(data["scope_offsets"][i + 1])
        if s1 > s0:
            scope_estimates = data["scope_flat"][s0:s1].astype(np.float64)
            reach_estimates = data["reach_flat"][
                int(data["reach_offsets"][i]):int(data["reach_offsets"][i + 1])
            ].astype(np.float64)
            occurred_flag = data["occurred_flag_flat"][
                int(data["occurred_flag_offsets"][i]):int(data["occurred_flag_offsets"][i + 1])
            ].astype(bool)
            occurred_index = data["occurred_index_flat"][
                int(data["occurred_index_offsets"][i]):int(data["occurred_index_offsets"][i + 1])
            ].astype(np.int64)

        t0 = int(data["tracked_ids_offsets"][i]);  t1 = int(data["tracked_ids_offsets"][i + 1])
        if t1 > t0:
            inline_tracked_ids = data["tracked_ids_flat"][t0:t1].tolist()

        name_val = str(data["tracked_name"][i])
        if name_val:
            inline_tracked_name = name_val.split("|")

    trunc_idx = int(data["truncation_idx"][i])
    term_id   = int(data["timeline_terminating_id"][i])

    return GeneratedTrajectory(
        patient_idx=int(data["patient_idx"][i]),
        sample_idx=int(data["sample_idx"][i]),
        traj_type=TrajectoryType(str(data["traj_type"][i])),
        prompt_len=int(data["prompt_len"][i]),
        output_ids=output_ids,
        timeline_terminating_id=term_id if term_id >= 0 else None,
        was_time_truncated=bool(data["was_time_truncated"][i]),
        truncation_idx=trunc_idx if trunc_idx >= 0 else None,
        scope_estimates=scope_estimates,
        reach_estimates=reach_estimates,
        occurred_flag=occurred_flag,
        occurred_index=occurred_index,
        inline_tracked_ids=inline_tracked_ids,
        inline_tracked_name=inline_tracked_name,
    )


def load_batch(data, indices) -> list[GeneratedTrajectory]:
    """Load a set of trajectories by npz-array position."""
    return [_extract_trajectory_at(data, int(i)) for i in indices]


def _save_shard(trajectories: list[GeneratedTrajectory], path: pathlib.Path) -> None:
    """Write trajectories directly to a single .npz shard file (no subdirectory wrapper)."""
    patient_idxs, sample_idxs, traj_types, prompt_lens = [], [], [], []
    was_time_truncateds, truncation_idxs, timeline_terminating_ids = [], [], []
    output_ids_flat, output_ids_offsets = [], [0]
    scope_flat, scope_offsets = [], [0]
    reach_flat, reach_offsets = [], [0]
    occurred_flag_flat, occurred_flag_offsets = [], [0]
    occurred_index_flat, occurred_index_offsets = [], [0]
    tracked_ids_flat, tracked_ids_offsets = [], [0]
    tracked_names: list[str] = []
    has_inline_sr = False

    for traj in trajectories:
        patient_idxs.append(traj.patient_idx)
        sample_idxs.append(traj.sample_idx)
        traj_types.append(traj.traj_type.value)
        prompt_lens.append(traj.prompt_len)
        was_time_truncateds.append(traj.was_time_truncated)
        truncation_idxs.append(traj.truncation_idx if traj.truncation_idx is not None else -1)
        timeline_terminating_ids.append(
            traj.timeline_terminating_id if traj.timeline_terminating_id is not None else -1
        )
        output_ids_flat.extend(traj.output_ids)
        output_ids_offsets.append(len(output_ids_flat))

        if traj.scope_estimates is not None:
            has_inline_sr = True
            scope_flat.extend(traj.scope_estimates.tolist())
            reach_flat.extend(traj.reach_estimates.tolist())
            occurred_flag_flat.extend(traj.occurred_flag.tolist())
            occurred_index_flat.extend(traj.occurred_index.tolist())
        scope_offsets.append(len(scope_flat))
        reach_offsets.append(len(reach_flat))
        occurred_flag_offsets.append(len(occurred_flag_flat))
        occurred_index_offsets.append(len(occurred_index_flat))

        if traj.inline_tracked_ids is not None:
            has_inline_sr = True
            tracked_ids_flat.extend(traj.inline_tracked_ids)
        tracked_ids_offsets.append(len(tracked_ids_flat))

        name = traj.inline_tracked_name
        tracked_names.append("|".join(name) if isinstance(name, list) else (str(name) if name else ""))

    save_dict: dict = dict(
        schema_version=np.array([2], dtype=np.int32),
        patient_idx=np.array(patient_idxs, dtype=np.int32),
        sample_idx=np.array(sample_idxs, dtype=np.int32),
        traj_type=np.array(traj_types, dtype="U2"),
        prompt_len=np.array(prompt_lens, dtype=np.int32),
        was_time_truncated=np.array(was_time_truncateds, dtype=bool),
        truncation_idx=np.array(truncation_idxs, dtype=np.int32),
        timeline_terminating_id=np.array(timeline_terminating_ids, dtype=np.int32),
        output_ids_flat=np.array(output_ids_flat, dtype=np.int32),
        output_ids_offsets=np.array(output_ids_offsets, dtype=np.int64),
    )
    if has_inline_sr:
        save_dict.update(
            scope_flat=np.array(scope_flat, dtype=np.float64),
            scope_offsets=np.array(scope_offsets, dtype=np.int64),
            reach_flat=np.array(reach_flat, dtype=np.float64),
            reach_offsets=np.array(reach_offsets, dtype=np.int64),
            occurred_flag_flat=np.array(occurred_flag_flat, dtype=bool),
            occurred_flag_offsets=np.array(occurred_flag_offsets, dtype=np.int64),
            occurred_index_flat=np.array(occurred_index_flat, dtype=np.int64),
            occurred_index_offsets=np.array(occurred_index_offsets, dtype=np.int64),
            tracked_ids_flat=np.array(tracked_ids_flat, dtype=np.int32),
            tracked_ids_offsets=np.array(tracked_ids_offsets, dtype=np.int64),
            tracked_name=np.array(tracked_names, dtype="U256"),
        )
    np.savez_compressed(path, **save_dict)


# ---------------------------------------------------------------------------
# Fast metadata scan — no Python trajectory objects, no output_ids_flat access
# ---------------------------------------------------------------------------

def scan_trajectories(
    traj_dir: pathlib.Path,
) -> tuple[object, np.ndarray, np.ndarray, set[int], GenerationConfig | None]:
    """Load the full trajectories npz into RAM and classify trajectories.

    Returns:
        data            — dict of all npz arrays (fully decompressed in RAM)
        early_indices   — npz positions of early-terminated M1 trajectories
        correct_indices — npz positions of correct M1 trajectories
        tracked_ids_set — set of tracked token IDs
        config          — GenerationConfig loaded from config.json, or None
    """
    logger.info("Loading trajectories.npz into RAM (one-time decompression)…")
    with np.load(traj_dir / "trajectories.npz", allow_pickle=False) as npz:
        data = dict(npz)
    logger.info("  Loaded %d arrays.", len(data))
    config = _load_config(traj_dir)

    tracked_ids_set: set[int] = set()
    if config is not None and config.tracked_ids:
        tracked_ids_set = set(config.tracked_ids)
    elif "tracked_ids_flat" in data:
        # Fall back: read from the first trajectory that has inline tracked ids
        t0 = int(data["tracked_ids_offsets"][0])
        t1 = int(data["tracked_ids_offsets"][1])
        if t1 > t0:
            tracked_ids_set = set(data["tracked_ids_flat"][t0:t1].tolist())

    traj_types = data["traj_type"]           # shape (N,), dtype U2
    term_ids   = data["timeline_terminating_id"]  # shape (N,), dtype int32

    m1_mask      = traj_types == "m1"
    early_mask   = m1_mask & np.isin(term_ids, sorted(tracked_ids_set))
    correct_mask = m1_mask & ~early_mask

    early_indices   = np.where(early_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    return data, early_indices, correct_indices, tracked_ids_set, config


# ---------------------------------------------------------------------------
# Summary (operates on numpy metadata arrays — no Python trajectory objects)
# ---------------------------------------------------------------------------

def print_summary(
    data,
    early_indices: np.ndarray,
    correct_indices: np.ndarray,
    tracked_ids_set: set[int],
    config: GenerationConfig | None,
) -> None:
    n_early   = len(early_indices)
    n_correct = len(correct_indices)
    n_total   = n_early + n_correct

    if n_total == 0:
        logger.info("No M1 trajectories found.")
        return

    logger.info("=" * 55)
    logger.info("  TRAJECTORY CLASSIFICATION SUMMARY")
    logger.info("=" * 55)
    logger.info("Tracked outcome token IDs (%d): %s", len(tracked_ids_set), sorted(tracked_ids_set))
    logger.info("Total M1 trajectories : %8d", n_total)
    logger.info("  Correct (natural end): %8d  (%.1f%%)", n_correct, 100 * n_correct / n_total)
    logger.info("  Early terminated     : %8d  (%.1f%%)", n_early,   100 * n_early   / n_total)

    if n_early == 0:
        logger.info("No early-terminated trajectories — nothing to rescue.")
        return

    logger.info("")
    logger.info("*** BUGGED (EARLY-TERMINATED) TIMELINES: %d / %d ***", n_early, n_total)
    logger.info("")

    term_ids_early = data["timeline_terminating_id"][early_indices]
    term_counts = Counter(int(x) for x in term_ids_early)
    logger.info("Early terminations by token ID:")
    for token_id, count in term_counts.most_common():
        name = "(unknown)"
        if config is not None and config.tracked_names and config.tracked_ids:
            try:
                name = config.tracked_names[list(config.tracked_ids).index(token_id)]
            except (ValueError, IndexError):
                pass
        logger.info("  token %6d  (%-40s): %6d trajectories", token_id, name, count)

    patient_idxs_early = data["patient_idx"][early_indices]
    n_patients_affected = len(np.unique(patient_idxs_early))
    all_m1_patients = len(np.unique(data["patient_idx"][np.concatenate([early_indices, correct_indices])]))
    logger.info(
        "Patients affected: %d / %d (%.1f%%)",
        n_patients_affected, all_m1_patients, 100 * n_patients_affected / all_m1_patients,
    )

    _, counts = np.unique(patient_idxs_early, return_counts=True)
    cs = np.sort(counts)
    logger.info(
        "Early-terminated samples per affected patient:  min=%d  median=%d  max=%d",
        cs[0], cs[len(cs) // 2], cs[-1],
    )

    # Output_ids lengths from offset arithmetic — no decompression of flat array needed
    offsets = data["output_ids_offsets"]
    le = np.sort(offsets[early_indices + 1]   - offsets[early_indices])
    lc = np.sort(offsets[correct_indices + 1] - offsets[correct_indices])
    logger.info(
        "Early-terminated output_ids lengths:  min=%d  median=%d  max=%d",
        le[0], le[len(le) // 2], le[-1],
    )
    if len(lc):
        logger.info(
            "Correct output_ids lengths:           min=%d  median=%d  max=%d",
            lc[0], lc[len(lc) // 2], lc[-1],
        )


# ---------------------------------------------------------------------------
# Patient token loading
# ---------------------------------------------------------------------------

def load_patient_tokens(output_dir: pathlib.Path, cfg: dict) -> list[list[int]]:
    """Load patient prompt tokens in the original run's patient_idx order."""
    index_df = pl.read_parquet(output_dir / "patient_index.parquet")
    parquet_path = (
        pathlib.Path(cfg["cocoa_outputs"]["held_out_for_inference"])
        .expanduser().resolve()
    )
    raw_df = pl.read_parquet(parquet_path, columns=["subject_id", "tokens_past"])

    joined = (
        index_df.select("patient_idx", "subject_id")
        .join(raw_df, on="subject_id", how="left")
        .sort("patient_idx")
    )

    missing = joined["tokens_past"].is_null().sum()
    if missing > 0:
        raise ValueError(
            f"{missing} patients in patient_index.parquet had no match in the parquet — "
            "ensure the same held_out_for_inference.parquet is used."
        )

    return joined["tokens_past"].to_list()


# ---------------------------------------------------------------------------
# Continuation generation — batched, streaming tqdm with ETA
# ---------------------------------------------------------------------------

async def _generate_continuation_batch(
    engine,
    config: GenerationConfig,
    batch: list[GeneratedTrajectory],
    patient_tokens: list[list[int]],
    pbar: tqdm,
    tok_counter: list[int],
    t0: float,
) -> list[GeneratedTrajectory]:
    """Generate continuations for one batch, updating a shared tqdm bar in-place."""
    results: list[GeneratedTrajectory | None] = [None] * len(batch)

    async def _run(i: int, traj: GeneratedTrajectory):
        result = await generate_trajectory(
            engine=engine,
            config=config,
            prompt_tokens=patient_tokens[traj.patient_idx] + traj.output_ids,
            patient_idx=traj.patient_idx,
            sample_idx=traj.sample_idx,
            traj_type=TrajectoryType.M1,
            stop_at_tracked_events=False,
        )
        return i, result

    tasks = [asyncio.create_task(_run(i, traj)) for i, traj in enumerate(batch)]
    for fut in asyncio.as_completed(tasks):
        i, result = await fut
        results[i] = result
        tok_counter[0] += result.n_new_tokens or 0
        elapsed = time.perf_counter() - t0
        pbar.set_postfix(
            new_tok=f"{tok_counter[0]:,}",
            tok_s=f"{tok_counter[0] / elapsed:.0f}" if elapsed > 0 else "—",
        )
        pbar.update(1)

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Score merging
# ---------------------------------------------------------------------------

def merge_trajectory(
    original: GeneratedTrajectory,
    continuation: GeneratedTrajectory,
) -> GeneratedTrajectory:
    """Merge an early-terminated M1 trajectory with its continuation.

    SCOPE merge rule per tracked event k:
      - If the original terminated AT tracked_ids[k] (i.e. the event occurred
        as the last token), SCOPE is already complete — do not add continuation.
      - Otherwise: SCOPE_corrected[k] = SCOPE_original[k] + SCOPE_continuation[k]

    REACH is merged analogously:
      - If the event already occurred: REACH is complete.
      - Otherwise: REACH_corrected[k] = 1 - (1 - R_orig[k]) * (1 - R_cont[k])

    M0 and timeline_terminating_id come from the continuation (the natural end).
    """
    tracked_ids = original.inline_tracked_ids or []
    orig_len = len(original.output_ids)

    if (
        original.scope_estimates is not None
        and continuation.scope_estimates is not None
    ):
        scope        = original.scope_estimates.copy()
        reach        = original.reach_estimates.copy()
        occurred_flag  = original.occurred_flag.copy()
        occurred_index = original.occurred_index.copy()

        for k, tid in enumerate(tracked_ids):
            if original.timeline_terminating_id == tid:
                # Event caused the early stop — SCOPE/REACH already accumulated to occurrence
                pass
            else:
                scope[k] = original.scope_estimates[k] + continuation.scope_estimates[k]
                reach[k] = 1.0 - (1.0 - original.reach_estimates[k]) * (1.0 - continuation.reach_estimates[k])
                if (
                    not occurred_flag[k]
                    and continuation.occurred_flag is not None
                    and continuation.occurred_flag[k]
                ):
                    occurred_flag[k]  = True
                    occurred_index[k] = orig_len + int(continuation.occurred_index[k])
    else:
        scope        = original.scope_estimates
        reach        = original.reach_estimates
        occurred_flag  = original.occurred_flag
        occurred_index = original.occurred_index

    truncation_idx = (
        None if continuation.truncation_idx is None
        else orig_len + continuation.truncation_idx
    )

    return GeneratedTrajectory(
        patient_idx=original.patient_idx,
        sample_idx=original.sample_idx,
        traj_type=original.traj_type,
        prompt_len=original.prompt_len,
        output_ids=original.output_ids + continuation.output_ids,
        timeline_terminating_id=continuation.timeline_terminating_id,
        was_time_truncated=continuation.was_time_truncated,
        truncation_idx=truncation_idx,
        scope_estimates=scope,
        reach_estimates=reach,
        occurred_flag=occurred_flag,
        occurred_index=occurred_index,
        inline_tracked_ids=original.inline_tracked_ids,
        inline_tracked_name=original.inline_tracked_name,
        n_new_tokens=(original.n_new_tokens or 0) + (continuation.n_new_tokens or 0),
    )


# ---------------------------------------------------------------------------
# Score accumulation
# ---------------------------------------------------------------------------

def accumulate_correct_scores(
    data,
    correct_indices: np.ndarray,
    tracked_ids: list[int],
    results: list[list[PatientResults]],
) -> None:
    """Stream correct M1 trajectories from npz in batches, accumulating inline scores.

    Correct trajectories were never stopped early so their inline SCOPE/REACH
    estimates are valid as-is. We read directly from the flat numpy arrays to
    avoid creating GeneratedTrajectory Python objects for the entire correct set.
    """
    # Pre-load all flat arrays once — NpzFile has no internal cache and would
    # re-decompress the full array on every dict access otherwise.
    pat_idx_arr        = data["patient_idx"]
    tid_offsets        = data["tracked_ids_offsets"]
    tid_flat           = data["tracked_ids_flat"]
    scope_off          = data["scope_offsets"]
    scope_fl           = data["scope_flat"]
    reach_off          = data["reach_offsets"]
    reach_fl           = data["reach_flat"]
    occ_flag_off       = data["occurred_flag_offsets"]
    occ_flag_fl        = data["occurred_flag_flat"]

    n = len(correct_indices)
    with tqdm(total=n, desc="Accumulating correct scores", unit="traj", dynamic_ncols=True) as pbar:
        for batch_start in range(0, n, BATCH_SIZE):
            batch = correct_indices[batch_start:batch_start + BATCH_SIZE]
            for raw_i in batch:
                i = int(raw_i)
                patient_idx = int(pat_idx_arr[i])

                t0 = int(tid_offsets[i])
                t1 = int(tid_offsets[i + 1])
                if t1 <= t0:
                    continue
                traj_tracked = tid_flat[t0:t1].tolist()

                s0 = int(scope_off[i]);  s1 = int(scope_off[i + 1])
                if s1 <= s0:
                    continue
                scope = scope_fl[s0:s1]
                reach = reach_fl[int(reach_off[i]):int(reach_off[i + 1])]
                occ   = occ_flag_fl[int(occ_flag_off[i]):int(occ_flag_off[i + 1])]

                for k, tid in enumerate(tracked_ids):
                    try:
                        ki = traj_tracked.index(tid)
                    except ValueError:
                        continue
                    results[k][patient_idx].m0_samples.append(bool(occ[ki]))
                    results[k][patient_idx].m1_samples.append(float(scope[ki]))
                    results[k][patient_idx].m2_samples.append(float(reach[ki]))

            pbar.update(len(batch))


async def accumulate_corrected_scores(
    engine,
    config: GenerationConfig,
    corrected: list[GeneratedTrajectory],
    continuations: list[GeneratedTrajectory],
    patient_tokens: list[list[int]],
    results: list[list[PatientResults]],
) -> None:
    """Accumulate scores for corrected (merged) M1 trajectories.

    Case B  — tracked event k does NOT appear in the continuation:
        Inline REACH from merge_trajectory is valid; use directly.
    Case B2 — tracked event k DOES appear in the continuation:
        Regenerate M2 by truncating before the event and re-sampling with
        event k suppressed, then score with a prefill pass.
    """
    tracked_ids = config.tracked_ids
    b2_tasks: list[tuple] = []

    for merged, cont in zip(corrected, continuations):
        cont_tracked_ks = {
            k for k, tid in enumerate(tracked_ids) if tid in cont.output_ids
        }
        for k, tid in enumerate(tracked_ids):
            ki = merged.inline_tracked_ids.index(tid)
            results[k][merged.patient_idx].m0_samples.append(bool(merged.occurred_flag[ki]))
            results[k][merged.patient_idx].m1_samples.append(float(merged.scope_estimates[ki]))

            if k in cont_tracked_ks:
                # Check that patient_tokens + prefix (up to the event) fits in context.
                # If not, fall back to the merged REACH rather than crashing SGLang.
                try:
                    cut = merged.output_ids.index(tid)
                except ValueError:
                    cut = len(merged.output_ids)
                if len(patient_tokens[merged.patient_idx]) + cut +10 >= config.max_len - 1:
                    results[k][merged.patient_idx].m2_samples.append(float(merged.reach_estimates[ki]))
                else:
                    b2_tasks.append((merged, k, dataclasses.replace(config, target_event_id=tid)))
            else:
                results[k][merged.patient_idx].m2_samples.append(float(merged.reach_estimates[ki]))

    if not b2_tasks:
        return

    logger.info("Regenerating M2 for %d Case B2 (outcome × trajectory) pairs...", len(b2_tasks))
    m2_trajectories = list(await asyncio.gather(*[
        generate_m2_from_m1_trajectory(
            engine, oc,
            dataclasses.replace(merged, timeline_terminating_id=oc.target_event_id),
            patient_tokens[merged.patient_idx],
        )
        for merged, _, oc in b2_tasks
    ]))
    scored = list(await asyncio.gather(*[
        score_trajectory(engine, oc, m2_traj, patient_tokens[merged.patient_idx])
        for (merged, _, oc), m2_traj in zip(b2_tasks, m2_trajectories)
    ]))
    for (merged, k, _), st in zip(b2_tasks, scored):
        results[k][merged.patient_idx].m2_samples.append(st.score)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_rescue(args):
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    traj_dir   = output_dir / "trajectories"

    if not (traj_dir / "trajectories.npz").exists():
        logger.error("trajectories.npz not found in %s", traj_dir)
        sys.exit(1)

    t_start = time.perf_counter()

    # ── Phase 1: fast metadata scan (no output_ids loaded) ──────────────────
    logger.info("Scanning trajectory metadata from: %s", traj_dir)
    data, early_indices, correct_indices, tracked_ids_set, config = scan_trajectories(traj_dir)
    print_summary(data, early_indices, correct_indices, tracked_ids_set, config)

    if len(early_indices) == 0:
        return

    cfg_path = pathlib.Path(args.config).expanduser().resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading patient tokens from parquet...")
    patient_tokens = load_patient_tokens(output_dir, cfg)
    logger.info("Loaded %d patient prompts.", len(patient_tokens))

    rescue_dir = pathlib.Path(args.rescue_dir).expanduser().resolve()
    rescue_dir.mkdir(parents=True, exist_ok=True)
    traj_rescue_dir = rescue_dir / "trajectories"
    traj_rescue_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(traj_dir / "config.json", traj_rescue_dir / "config.json")
    shard_paths: list[str] = []
    shard_idx = 0

    mem_frac = cfg.get("engine", {}).get("mem_fraction", 0.85)
    logger.info("Starting SGLang engine (model=%s)...", cfg["model_path"])
    engine = create_engine(
        model_path=str(pathlib.Path(cfg["model_path"]).expanduser().resolve()),
        max_len=config.max_len,
        use_time_horizon=False,
        mem_fraction=mem_frac,
    )

    try:
        n_early     = len(early_indices)
        n_patients  = len(patient_tokens)
        tracked_ids = config.tracked_ids
        n_outcomes  = len(tracked_ids)
        results = [[PatientResults() for _ in range(n_patients)] for _ in range(n_outcomes)]

        # ── Phase 3 + 4: load → generate → merge → score, BATCH_SIZE at a time
        logger.info(
            "Processing %d early-terminated trajectories in batches of %d...",
            n_early, BATCH_SIZE,
        )
        t0          = time.perf_counter()
        tok_counter = [0]  # mutable so _generate_continuation_batch can update it

        with tqdm(total=n_early, desc="Completing bugged timelines", unit="traj", dynamic_ncols=True) as pbar:
            for batch_start in range(0, n_early, BATCH_SIZE):
                batch_indices = early_indices[batch_start:batch_start + BATCH_SIZE]

                # Load only this batch — previous batch is now out of scope and GC-eligible
                early_batch = load_batch(data, batch_indices)

                continuations_batch = await _generate_continuation_batch(
                    engine, config, early_batch, patient_tokens, pbar, tok_counter, t0,
                )

                corrected_batch = [
                    merge_trajectory(orig, cont)
                    for orig, cont in zip(early_batch, continuations_batch)
                ]

                shard_path = traj_rescue_dir / f"corrected_shard_{shard_idx:04d}.npz"
                _save_shard(corrected_batch, shard_path)
                logger.info(
                    "  Shard %d: %d trajectories saved → %s",
                    shard_idx, len(corrected_batch), shard_path.name,
                )
                shard_paths.append(shard_path.name)
                shard_idx += 1

                await accumulate_corrected_scores(
                    engine, config, corrected_batch, continuations_batch, patient_tokens, results,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Continuations complete: %d  |  %s new tokens  |  %.1f tok/s  |  %.1fs total",
            n_early, f"{tok_counter[0]:,}",
            tok_counter[0] / elapsed if elapsed > 0 else 0.0, elapsed,
        )

        # ── Phase 4b: accumulate scores for correct (unchanged) trajectories ─
        logger.info(
            "Accumulating scores for %d correct trajectories, %d outcomes...",
            len(correct_indices), n_outcomes,
        )
        accumulate_correct_scores(data, correct_indices, tracked_ids, results)

    finally:
        engine.shutdown()
        logger.info("Engine shut down.")

    # ── Phase 5: save outputs ────────────────────────────────────────────────
    logger.info("Saving outputs to %s ...", rescue_dir)

    tracked_names = config.tracked_names or []
    for k, evt_name in enumerate(tracked_names):
        safe_name   = evt_name.replace("/", "_").replace(" ", "_")
        scores_path = rescue_dir / f"scores_{safe_name}.npz"
        save_scores(results[k], scores_path)
        logger.info("  Saved %s", scores_path.name)

    manifest = {
        "corrected_shards": [f"trajectories/{p}" for p in shard_paths],
        "original_trajectories": str(traj_dir / "trajectories.npz"),
        "n_corrected": n_early,
        "n_correct_original": len(correct_indices),
    }
    manifest_path = rescue_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  Saved manifest.json  (%d corrected shards)", len(shard_paths))

    total_elapsed = time.perf_counter() - t_start
    logger.info("=" * 55)
    logger.info("  Rescue complete in %.1fs  →  %s", total_elapsed, rescue_dir)
    logger.info("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Rescue early-terminated M1 trajectories.")
    parser.add_argument("--output-dir", "-o", default="./scope_reach_output")
    parser.add_argument("--rescue-dir", "-r", default="./scope_reach_output_rescue")
    parser.add_argument("--config",     "-c", default="./pipeline_config.yaml")
    args = parser.parse_args()

    asyncio.run(run_rescue(args))


if __name__ == "__main__":
    main()
