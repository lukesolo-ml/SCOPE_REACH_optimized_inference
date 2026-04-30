#!/usr/bin/env python3
"""
Recover global patient_idx for each segment's partial_scores.npz.

Self-contained: does NOT import from rescue.py. Reads trajectories.npz
directly and reproduces the chunk-local → global translation inline.

The original rescue_segment.py runs were chunk-local-indexed: trajectories
in partial_scores.npz carry patient_idx in [0, CHUNK_SIZE), and only slots
0..9 of every per-outcome results array were ever populated.

This script reconstructs the GLOBAL patient_idx (in [0, n_subjects)) for
every contribution row in partial_scores.npz by replaying the deterministic
emission order of accumulate_corrected_scores + accumulate_correct_scores.

Per (merged_traj, k), accumulate_corrected_scores appends EXACTLY one m0/m1/m2
sample regardless of B2: the B2 path replaces the inline reach value, it does
not add an extra sample. So contribution counts per (k, local_pidx) are
fully determined by:

  count = #{i in early_seg : data["patient_idx"][i] % CHUNK_SIZE == local}
        + #{i in correct_seg with non-empty inline data, tid_k in traj_tids,
                   data["patient_idx"][i] % CHUNK_SIZE == local}

We do NOT need to know whether B2 fired — we carry m0/m1/m2 values from
the partial unchanged, and only translate patient_idx_flat from local to
global. The order of contributions within each (k, local) group in the
partial matches the early_seg-then-correct_seg traversal order.

Output: seg_NN/partial_scores_global.npz with the same schema as
partial_scores.npz, but patient_idx_flat is global and n_patients is the
canonical n_subjects from run_summary.json.

Usage:
    python rescue_recover.py \\
        --output-dir ./scope_reach_output \\
        --rescue-dir ./scope_reach_output_rescue \\
        --n-segments 32
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("recover")

# Inference concurrency used by the original pipeline run.
CHUNK_SIZE = 10


# ---------------------------------------------------------------------------
# Inlined helpers (would otherwise come from a fixed rescue.py)
# ---------------------------------------------------------------------------

def _load_subject_ids(output_dir: pathlib.Path) -> list:
    """Load the canonical global subject_id list from run_summary.json."""
    rs_path = output_dir / "run_summary.json"
    with open(rs_path) as f:
        rs = json.load(f)
    if "subject_ids" not in rs:
        raise KeyError(f"{rs_path} has no 'subject_ids' key")
    return rs["subject_ids"]


def _load_n_samp(traj_dir: pathlib.Path) -> int:
    """Read n_samp from config.json — required for chunk_idx arithmetic."""
    cfg_path = traj_dir / "config.json"
    with open(cfg_path) as f:
        d = json.load(f)
    if "n_samp" not in d:
        raise KeyError(f"{cfg_path} has no 'n_samp' key")
    return int(d["n_samp"])


def _load_tracked_ids(traj_dir: pathlib.Path) -> list[int]:
    cfg_path = traj_dir / "config.json"
    with open(cfg_path) as f:
        d = json.load(f)
    if "tracked_ids" not in d:
        raise KeyError(f"{cfg_path} has no 'tracked_ids' key")
    return [int(t) for t in d["tracked_ids"]]


def _compute_global_patient_idx(
    traj_types: np.ndarray,
    local_patient_idx: np.ndarray,
    n_samp: int,
    n_subjects: int,
) -> np.ndarray:
    """Translate chunk-local patient_idx (0..CHUNK_SIZE-1) → global (0..n_subjects-1).

    For each traj_type independently, position-within-type determines chunk_idx:
        chunk_idx = position_within_type // (CHUNK_SIZE * n_samp)
        global    = chunk_idx * CHUNK_SIZE + local_patient_idx
    """
    n = len(traj_types)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    global_pidx = np.empty(n, dtype=np.int32)
    chunk_traj_size = CHUNK_SIZE * n_samp

    for ttype in np.unique(traj_types):
        mask = traj_types == ttype
        type_count = int(mask.sum())
        positions_within_type = np.arange(type_count, dtype=np.int64)
        chunk_idx_for_type = positions_within_type // chunk_traj_size
        global_pidx[mask] = (
            chunk_idx_for_type * CHUNK_SIZE + local_patient_idx[mask]
        ).astype(np.int32, copy=False)

    # Validation
    if int(local_patient_idx.max()) >= CHUNK_SIZE:
        raise ValueError(
            f"local patient_idx max {int(local_patient_idx.max())} "
            f">= CHUNK_SIZE {CHUNK_SIZE} — file may already be globally indexed."
        )
    g_min = int(global_pidx.min())
    g_max = int(global_pidx.max())
    if g_min < 0 or g_max >= n_subjects:
        raise ValueError(
            f"Translated global patient_idx out of range: [{g_min}, {g_max}] "
            f"vs n_subjects={n_subjects}. Check CHUNK_SIZE ({CHUNK_SIZE}) and "
            f"n_samp ({n_samp}) match the original pipeline run."
        )

    # Spot-check chunk-major ordering on the largest traj_type
    largest_ttype = max(np.unique(traj_types), key=lambda t: int((traj_types == t).sum()))
    largest_local = local_patient_idx[traj_types == largest_ttype]
    n_full_chunks_largest = len(largest_local) // chunk_traj_size
    n_check = min(3, n_full_chunks_largest)
    for c in range(n_check):
        block = largest_local[c * chunk_traj_size : (c + 1) * chunk_traj_size]
        unique_in_block = set(int(x) for x in block.tolist())
        expected = set(range(CHUNK_SIZE))
        if unique_in_block != expected:
            raise ValueError(
                f"Chunk {c} of traj_type {largest_ttype!r} has unexpected "
                f"local patient_idx: {sorted(unique_in_block)} "
                f"(expected {sorted(expected)}). Chunk-major layout violated."
            )

    return global_pidx


def scan_and_classify(traj_dir: pathlib.Path, n_subjects: int):
    """Load trajectories.npz, rewrite patient_idx to global, classify M1.

    Returns: (data, early_indices, correct_indices, tracked_ids)
    """
    logger.info("Loading trajectories.npz into RAM (one-time decompression)...")
    t0 = time.perf_counter()
    with np.load(traj_dir / "trajectories.npz", allow_pickle=False) as npz:
        data = dict(npz)
    logger.info("  Loaded %d arrays in %.1fs.", len(data), time.perf_counter() - t0)

    n_samp = _load_n_samp(traj_dir)
    tracked_ids = _load_tracked_ids(traj_dir)

    local_max = int(data["patient_idx"].max()) if len(data["patient_idx"]) else 0
    if local_max >= CHUNK_SIZE:
        if local_max >= n_subjects:
            raise ValueError(
                f"patient_idx max {local_max} >= n_subjects {n_subjects} and "
                f">= CHUNK_SIZE {CHUNK_SIZE}; ambiguous."
            )
        logger.info(
            "patient_idx max=%d >= CHUNK_SIZE=%d — assuming already global, skipping rewrite.",
            local_max, CHUNK_SIZE,
        )
    else:
        logger.info(
            "Translating chunk-local patient_idx (CHUNK_SIZE=%d, n_samp=%d) → global...",
            CHUNK_SIZE, n_samp,
        )
        global_pidx = _compute_global_patient_idx(
            traj_types=data["traj_type"],
            local_patient_idx=data["patient_idx"],
            n_samp=n_samp,
            n_subjects=n_subjects,
        )
        data["patient_idx"] = global_pidx
        logger.info(
            "  Global patient_idx: range [%d, %d], unique=%d.",
            int(global_pidx.min()), int(global_pidx.max()),
            len(np.unique(global_pidx)),
        )

    traj_types = data["traj_type"]
    term_ids   = data["timeline_terminating_id"]
    tracked_ids_set = set(tracked_ids)

    m1_mask      = traj_types == "m1"
    early_mask   = m1_mask & np.isin(term_ids, sorted(tracked_ids_set))
    correct_mask = m1_mask & ~early_mask

    early_indices   = np.where(early_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    return data, early_indices, correct_indices, tracked_ids


# ---------------------------------------------------------------------------
# Per-segment replay + remap
# ---------------------------------------------------------------------------

def replay_segment_order(
    data: dict,
    early_indices: np.ndarray,
    correct_indices: np.ndarray,
    tracked_ids: list[int],
    seg_dir: pathlib.Path,
    n_outcomes: int,
) -> dict[tuple[int, int], list[int]]:
    """Replay the deterministic emission order for one segment.

    Returns: {(k, local_pidx): [global_pidx, ...]} in append order.
    """
    with open(seg_dir / "manifest.json") as f:
        m = json.load(f)
    early_start   = int(m["early_start"])
    early_end     = int(m["early_end"])
    correct_start = int(m["correct_start"])
    correct_end   = int(m["correct_end"])

    early_seg   = early_indices[early_start:early_end]
    correct_seg = correct_indices[correct_start:correct_end]

    pat_idx_arr   = data["patient_idx"]    # GLOBAL after scan_and_classify
    tid_offsets   = data["tracked_ids_offsets"]
    tid_flat      = data["tracked_ids_flat"]
    scope_offsets = data["scope_offsets"]

    out: dict[tuple[int, int], list[int]] = {}

    # ── Phase A: corrected (early-derived) trajectories ──────────────────────
    for raw_i in early_seg:
        i = int(raw_i)
        global_pidx = int(pat_idx_arr[i])
        local_pidx  = global_pidx % CHUNK_SIZE
        for k in range(n_outcomes):
            key = (k, local_pidx)
            if key not in out:
                out[key] = []
            out[key].append(global_pidx)

    # ── Phase B: correct trajectories ────────────────────────────────────────
    n_skipped = 0
    for raw_i in correct_seg:
        i = int(raw_i)
        t0 = int(tid_offsets[i])
        t1 = int(tid_offsets[i + 1])
        if t1 <= t0:
            n_skipped += 1
            continue
        s0 = int(scope_offsets[i])
        s1 = int(scope_offsets[i + 1])
        if s1 <= s0:
            n_skipped += 1
            continue
        traj_tids_set = set(int(t) for t in tid_flat[t0:t1].tolist())

        global_pidx = int(pat_idx_arr[i])
        local_pidx  = global_pidx % CHUNK_SIZE

        for k, tid in enumerate(tracked_ids):
            if int(tid) in traj_tids_set:
                key = (k, local_pidx)
                if key not in out:
                    out[key] = []
                out[key].append(global_pidx)

    if n_skipped:
        logger.info("  Phase B: skipped %d correct trajectories (no inline data)", n_skipped)
    return out


def remap_segment(
    seg_dir: pathlib.Path,
    ordered: dict[tuple[int, int], list[int]],
    n_subjects: int,
    overwrite: bool = False,
) -> pathlib.Path:
    """Validate counts and write partial_scores_global.npz."""
    partial_path = seg_dir / "partial_scores.npz"
    out_path     = seg_dir / "partial_scores_global.npz"

    if out_path.exists() and not overwrite:
        logger.info("  %s already exists, skipping (use --overwrite to redo)", out_path.name)
        return out_path

    with np.load(partial_path, allow_pickle=False) as npz:
        local_pidx_flat = npz["patient_idx_flat"][:]
        event_k_flat    = npz["event_k_flat"][:]
        m0_flat         = npz["m0_flat"][:]
        m1_flat         = npz["m1_flat"][:]
        m2_flat         = npz["m2_flat"][:]
        tracked_ids     = npz["tracked_ids"][:]
        tracked_names   = npz["tracked_names"][:]

    n_rows = len(local_pidx_flat)
    expected_total = sum(len(v) for v in ordered.values())
    logger.info("  Partial: %d rows  |  Replay: %d expected rows", n_rows, expected_total)

    if expected_total != n_rows:
        raise ValueError(
            f"TOTAL ROW COUNT MISMATCH for {seg_dir.name}: "
            f"partial has {n_rows}, replay says {expected_total}."
        )

    global_pidx_flat = np.empty(n_rows, dtype=np.int32)

    cursor = 0
    n_groups = 0
    while cursor < n_rows:
        k     = int(event_k_flat[cursor])
        local = int(local_pidx_flat[cursor])
        end = cursor
        while (end < n_rows
               and int(event_k_flat[end])    == k
               and int(local_pidx_flat[end]) == local):
            end += 1
        n_in_group = end - cursor

        expected = ordered.get((k, local), [])
        if n_in_group != len(expected):
            raise ValueError(
                f"GROUP COUNT MISMATCH in {seg_dir.name} (k={k}, local={local}): "
                f"partial has {n_in_group} rows, replay says {len(expected)}. "
                f"Diff = {n_in_group - len(expected)}."
            )
        global_pidx_flat[cursor:end] = expected
        cursor = end
        n_groups += 1

    assert cursor == n_rows
    logger.info("  Validated %d (k, local_pidx) groups", n_groups)

    g_min = int(global_pidx_flat.min())
    g_max = int(global_pidx_flat.max())
    g_unique = len(np.unique(global_pidx_flat))
    logger.info(
        "  Global pidx: range [%d, %d], unique=%d (n_subjects=%d)",
        g_min, g_max, g_unique, n_subjects,
    )
    if g_max >= n_subjects or g_min < 0:
        raise ValueError(
            f"Global pidx out of range: [{g_min}, {g_max}] vs n_subjects={n_subjects}"
        )

    np.savez_compressed(
        out_path,
        n_patients=np.array([n_subjects], dtype=np.int64),
        tracked_ids=tracked_ids,
        tracked_names=tracked_names,
        patient_idx_flat=global_pidx_flat,
        event_k_flat=event_k_flat,
        m0_flat=m0_flat,
        m1_flat=m1_flat,
        m2_flat=m2_flat,
    )
    logger.info("  ✓ Wrote %s", out_path.name)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Translate chunk-local partial_scores.npz to global indexing."
    )
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--rescue-dir", "-r", required=True)
    parser.add_argument("--n-segments", "-n", type=int, default=32)
    parser.add_argument("--segments",   type=int, nargs="+", default=None,
                        help="Optional: only process these segment indices")
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    rescue_dir = pathlib.Path(args.rescue_dir).expanduser().resolve()
    traj_dir   = output_dir / "trajectories"

    subject_ids = _load_subject_ids(output_dir)
    n_subjects  = len(subject_ids)
    logger.info("Loaded %d subject_ids from run_summary.json", n_subjects)

    data, early_indices, correct_indices, tracked_ids = scan_and_classify(traj_dir, n_subjects)
    n_outcomes = len(tracked_ids)
    logger.info(
        "%d M1 early, %d M1 correct, %d outcomes",
        len(early_indices), len(correct_indices), n_outcomes,
    )

    seg_indices = args.segments if args.segments else list(range(args.n_segments))
    successes, failures = [], []

    for seg_idx in seg_indices:
        seg_dir = rescue_dir / f"seg_{seg_idx:02d}"
        if not (seg_dir / "partial_scores.npz").exists():
            logger.warning("Skipping seg_%02d: no partial_scores.npz", seg_idx)
            continue
        if not (seg_dir / "manifest.json").exists():
            logger.warning("Skipping seg_%02d: no manifest.json", seg_idx)
            continue

        logger.info("─" * 55)
        logger.info("Segment %d:", seg_idx)
        t_seg = time.perf_counter()
        try:
            ordered = replay_segment_order(
                data, early_indices, correct_indices,
                tracked_ids, seg_dir, n_outcomes,
            )
            remap_segment(seg_dir, ordered, n_subjects, overwrite=args.overwrite)
            successes.append(seg_idx)
            logger.info("  Done in %.1fs", time.perf_counter() - t_seg)
        except Exception as e:
            logger.error("  FAILED: %s", e)
            failures.append((seg_idx, str(e)))

    logger.info("─" * 55)
    logger.info("Recovery complete: %d succeeded, %d failed", len(successes), len(failures))
    if failures:
        for s, msg in failures:
            logger.error("  seg_%02d: %s", s, msg)
        sys.exit(1)

    logger.info("")
    logger.info("Next: run rescue_merge.py over partial_scores_global.npz files.")
    logger.info("Quick swap-and-merge:")
    logger.info("  for d in %s/seg_*/; do "
                "mv \"$d/partial_scores.npz\" \"$d/partial_scores_local_BROKEN.npz\" && "
                "mv \"$d/partial_scores_global.npz\" \"$d/partial_scores.npz\"; done",
                rescue_dir)


if __name__ == "__main__":
    main()