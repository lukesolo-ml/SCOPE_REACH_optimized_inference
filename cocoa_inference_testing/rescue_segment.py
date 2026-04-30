#!/usr/bin/env python3
"""
Segmented rescue: one SLURM array task processes 1/N of the early-terminated trajectories.

Run rescue_merge.py after all segments complete to produce final score files.

Usage:
    python rescue_segment.py \
        --output-dir  ./scope_reach_output \
        --rescue-dir  ./scope_reach_output_rescue \
        --config      ./pipeline_config.yaml \
        --segment-idx 0 \
        --n-segments  32
"""

import argparse
import asyncio
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import yaml
from tqdm.auto import tqdm

# Pull all helpers from rescue.py — no duplication.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rescue import (
    BATCH_SIZE,
    _save_shard,
    scan_trajectories,
    print_summary,
    load_patient_tokens,
    load_batch,
    _generate_continuation_batch,
    merge_trajectory,
    accumulate_corrected_scores,
    accumulate_correct_scores,
    logger,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from quick_sco_re.scheduler import create_engine
from quick_sco_re.structures import PatientResults


# ---------------------------------------------------------------------------
# Partial score serialisation
# ---------------------------------------------------------------------------

def _save_partial_scores(
    results: list[list[PatientResults]],
    tracked_ids: list[int],
    tracked_names: list[str],
    n_patients: int,
    path: pathlib.Path,
) -> None:
    """Write per-sample contributions to a flat .npz for later merging.

    One row per (patient_idx, tracked-event, MC-sample).  The merge script
    groups by patient and event to compute final per-patient means.
    """
    patient_idx_flat: list[int] = []
    event_k_flat:     list[int] = []
    m0_flat: list[float] = []
    m1_flat: list[float] = []
    m2_flat: list[float] = []

    for k, patient_list in enumerate(results):
        for patient_idx, pr in enumerate(patient_list):
            n = len(pr.m0_samples)
            if n == 0:
                continue
            patient_idx_flat.extend([patient_idx] * n)
            event_k_flat.extend([k] * n)
            m0_flat.extend(pr.m0_samples)
            m1_flat.extend(pr.m1_samples)
            m2_flat.extend(pr.m2_samples)

    np.savez_compressed(
        path,
        n_patients=np.array([n_patients], dtype=np.int64),
        tracked_ids=np.array(tracked_ids, dtype=np.int32),
        tracked_names=np.array(tracked_names, dtype="U256"),
        patient_idx_flat=np.array(patient_idx_flat, dtype=np.int32),
        event_k_flat=np.array(event_k_flat, dtype=np.int32),
        m0_flat=np.array(m0_flat, dtype=np.float64),
        m1_flat=np.array(m1_flat, dtype=np.float64),
        m2_flat=np.array(m2_flat, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_rescue_segment(args):
    seg_idx    = args.segment_idx
    n_segments = args.n_segments

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    traj_dir   = output_dir / "trajectories"

    if not (traj_dir / "trajectories.npz").exists():
        logger.error("trajectories.npz not found in %s", traj_dir)
        sys.exit(1)

    t_start = time.perf_counter()
    logger.info("=" * 55)
    logger.info("  Segment %d / %d", seg_idx, n_segments)
    logger.info("=" * 55)

    # ── Phase 1: scan ────────────────────────────────────────────────────────
    logger.info("Scanning trajectory metadata...")
    data, early_indices, correct_indices, tracked_ids_set, config = scan_trajectories(traj_dir)

    if seg_idx == 0:
        print_summary(data, early_indices, correct_indices, tracked_ids_set, config)

    if len(early_indices) == 0:
        logger.info("No early-terminated trajectories — nothing to rescue.")
        return

    # ── Phase 2: compute this segment's slice ────────────────────────────────
    # Early trajectories: contiguous block (GPU work, balanced by count).
    early_per_seg = (len(early_indices)   + n_segments - 1) // n_segments
    early_start   = seg_idx * early_per_seg
    early_end     = min(early_start + early_per_seg, len(early_indices))
    early_seg     = early_indices[early_start:early_end]

    # Correct trajectories: contiguous block (CPU work, balanced by count).
    correct_per_seg = (len(correct_indices) + n_segments - 1) // n_segments
    correct_start   = seg_idx * correct_per_seg
    correct_end     = min(correct_start + correct_per_seg, len(correct_indices))
    correct_seg     = correct_indices[correct_start:correct_end]

    # Optional cap for testing (--max-early limits both slices proportionally).
    if args.max_early is not None:
        cap = args.max_early
        early_seg   = early_seg[:cap]
        correct_seg = correct_seg[:cap]

    logger.info(
        "Segment %d: %d early (positions %d–%d), %d correct (positions %d–%d)",
        seg_idx,
        len(early_seg),   early_start,   early_end   - 1,
        len(correct_seg), correct_start, correct_end - 1,
    )

    cfg_path = pathlib.Path(args.config).expanduser().resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Loading patient tokens from parquet...")
    patient_tokens = load_patient_tokens(output_dir, cfg)
    logger.info("  Loaded %d patient prompts.", len(patient_tokens))

    # ── Segment output directory ──────────────────────────────────────────────
    rescue_dir   = pathlib.Path(args.rescue_dir).expanduser().resolve()
    seg_dir      = rescue_dir / f"seg_{seg_idx:02d}"
    traj_seg_dir = seg_dir / "trajectories"

    if seg_dir.exists():
        logger.error("Segment directory already exists: %s", seg_dir)
        logger.error("Move or delete it before re-running this segment.")
        sys.exit(1)

    seg_dir.mkdir(parents=True)
    traj_seg_dir.mkdir()
    shutil.copy2(traj_dir / "config.json", traj_seg_dir / "config.json")

    shard_paths: list[str] = []
    shard_idx = 0

    # ── Phase 3: engine ───────────────────────────────────────────────────────
    mem_frac = cfg.get("engine", {}).get("mem_fraction", 0.85)
    logger.info("Starting SGLang engine (model=%s)...", cfg["model_path"])
    engine = create_engine(
        model_path=str(pathlib.Path(cfg["model_path"]).expanduser().resolve()),
        max_len=config.max_len,
        use_time_horizon=False,
        mem_fraction=mem_frac,
    )

    try:
        n_early    = len(early_seg)
        n_patients = len(patient_tokens)
        tracked_ids = config.tracked_ids
        n_outcomes  = len(tracked_ids)
        results = [[PatientResults() for _ in range(n_patients)] for _ in range(n_outcomes)]

        # ── Phase 4: generate continuations ──────────────────────────────────
        logger.info(
            "Completing %d early-terminated trajectories in batches of %d...",
            n_early, BATCH_SIZE,
        )
        t0          = time.perf_counter()
        tok_counter = [0]

        with tqdm(
            total=n_early,
            desc=f"Seg {seg_idx:02d} continuations",
            unit="traj",
            dynamic_ncols=True,
        ) as pbar:
            for batch_start in range(0, n_early, BATCH_SIZE):
                batch_indices = early_seg[batch_start:batch_start + BATCH_SIZE]
                early_batch   = load_batch(data, batch_indices)

                continuations_batch = await _generate_continuation_batch(
                    engine, config, early_batch, patient_tokens, pbar, tok_counter, t0,
                )
                corrected_batch = [
                    merge_trajectory(orig, cont)
                    for orig, cont in zip(early_batch, continuations_batch)
                ]

                shard_path = traj_seg_dir / f"corrected_shard_{shard_idx:04d}.npz"
                _save_shard(corrected_batch, shard_path)
                logger.info(
                    "  Shard %d: %d trajectories → %s",
                    shard_idx, len(corrected_batch), shard_path.name,
                )
                shard_paths.append(shard_path.name)
                shard_idx += 1

                await accumulate_corrected_scores(
                    engine, config, corrected_batch, continuations_batch, patient_tokens, results,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Continuations done: %d  |  %s tokens  |  %.1f tok/s  |  %.1fs",
            n_early, f"{tok_counter[0]:,}",
            tok_counter[0] / max(elapsed, 1e-9), elapsed,
        )

        # ── Phase 4b: correct trajectories (CPU only) ─────────────────────────
        logger.info(
            "Accumulating scores for %d correct trajectories (this segment's share)...",
            len(correct_seg),
        )
        accumulate_correct_scores(data, correct_seg, tracked_ids, results)

    finally:
        engine.shutdown()
        logger.info("Engine shut down.")

    # ── Phase 5: save ────────────────────────────────────────────────────────
    partial_path = seg_dir / "partial_scores.npz"
    _save_partial_scores(
        results,
        list(tracked_ids),
        list(config.tracked_names or []),
        n_patients,
        partial_path,
    )
    logger.info("Partial scores saved → %s", partial_path)

    manifest = {
        "segment_idx":          seg_idx,
        "n_segments":           n_segments,
        "early_start":          int(early_start),
        "early_end":            int(early_end),
        "correct_start":        int(correct_start),
        "correct_end":          int(correct_end),
        "n_corrected_this_seg": int(n_early),
        "corrected_shards":     [f"trajectories/{p}" for p in shard_paths],
    }
    with open(seg_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_elapsed = time.perf_counter() - t_start
    logger.info("Segment %d complete in %.1fs  →  %s", seg_idx, total_elapsed, seg_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Rescue one segment of early-terminated M1 trajectories (SLURM array task)."
    )
    parser.add_argument("--output-dir",  "-o", required=True, help="Original pipeline output dir")
    parser.add_argument("--rescue-dir",  "-r", required=True, help="Root rescue output dir (shared across segments)")
    parser.add_argument("--config",      "-c", required=True, help="pipeline_config.yaml")
    parser.add_argument("--segment-idx", "-i", type=int, required=True, help="0-based segment index")
    parser.add_argument("--n-segments",  "-n", type=int, default=32,    help="Total number of segments")
    parser.add_argument("--max-early",        type=int, default=None,   help="Cap early trajectories per segment (for testing)")
    args = parser.parse_args()

    if args.segment_idx < 0 or args.segment_idx >= args.n_segments:
        parser.error(f"--segment-idx must be in [0, {args.n_segments - 1}]")

    asyncio.run(run_rescue_segment(args))


if __name__ == "__main__":
    main()
