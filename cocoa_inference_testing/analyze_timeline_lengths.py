#!/usr/bin/env python3
"""
Analyze token distribution around tracked events in full patient timelines.

For each patient in tokens_times.parquet:
  1. Skip the first 24 hours of the admission.
  2. In the remaining post-24h timeline, find the first occurrence of any tracked_id.
  3. Record:
       pre_event  = tokens BEFORE that first occurrence
                    (or the full post-24h length if no tracked event present)
       post_event = tokens AFTER that first occurrence (0 if no tracked event)

This validates whether the rescue run's ~1400-token continuation average
is consistent with actual clinical timeline structure.
"""

import datetime
import pathlib
import sys

import numpy as np
import polars as pl
from tqdm.auto import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKENS_TIMES = pathlib.Path(
    "/gpfs/data/bbj-lab/users/lsolo/standpipe/lib/cocoa/cocoa/processed/tokens_times.parquet"
)

TRACKED_IDS = np.array(
    [1370, 1213, 121, 611, 620, 548, 557, 388, 377, 1300, 1267, 1262, 1257, 1252, 61],
    dtype=np.int64,
)
TRACKED_NAMES = {
    1370: "XFR-IN//icu",
    1213: "RESP//imv",
    121:  "DSCG//expired",
    611:  "LAB-RES//sodium_Q0",
    620:  "LAB-RES//sodium_Q9",
    548:  "LAB-RES//potassium_Q0",
    557:  "LAB-RES//potassium_Q9",
    388:  "LAB-RES//hemoglobin_Q0",
    377:  "LAB-RES//glucose_serum_Q0",
    1300: "VTL//heart_rate_Q9",
    1267: "SOFA//resp-4",
    1262: "SOFA//renal-4",
    1257: "SOFA//liver-4",
    1252: "SOFA//cv-4",
    61:   "ASMT//cam_total_positive",
}

HOURS_24 = datetime.timedelta(hours=24)

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {TOKENS_TIMES} ...")
    df = pl.read_parquet(TOKENS_TIMES)
    print(f"  {len(df):,} patients loaded.")

    tracked_set = set(TRACKED_IDS.tolist())

    pre_event_lengths  = []   # tokens before first tracked event (post-24h)
    post_event_lengths = []   # tokens after first tracked event (0 if no event)
    n_no_event         = 0    # patients with no tracked event post-24h
    n_too_short        = 0    # patients whose full timeline is ≤24h
    per_event_counts   = {tid: 0 for tid in TRACKED_IDS}  # which event triggered

    rows = df.iter_rows(named=True)
    for row in tqdm(rows, total=len(df), desc="Analyzing", unit="patient"):
        tokens = row["tokens"]
        times  = row["times"]

        if not tokens:
            continue

        # 24h cutoff from admission (first token time)
        cutoff = times[0] + HOURS_24

        # Find first index past the 24h mark
        post24_start = next(
            (i for i, t in enumerate(times) if t > cutoff),
            None,
        )
        if post24_start is None:
            n_too_short += 1
            continue

        post_tokens = tokens[post24_start:post24_start + 10_000]

        if not post_tokens:
            n_too_short += 1
            continue

        # Find first tracked event in post-24h tokens
        first_event_idx = None
        first_event_id  = None
        for i, tok in enumerate(post_tokens):
            if tok in tracked_set:
                first_event_idx = i
                first_event_id  = tok
                break

        if first_event_idx is None:
            pre_event_lengths.append(len(post_tokens))
            post_event_lengths.append(0)
            n_no_event += 1
        else:
            pre_event_lengths.append(first_event_idx)
            post_event_lengths.append(len(post_tokens) - first_event_idx - 1)
            per_event_counts[first_event_id] += 1

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    pre  = np.array(pre_event_lengths,  dtype=np.int64)
    post = np.array(post_event_lengths, dtype=np.int64)
    n    = len(pre)

    print()
    print("=" * 62)
    print("  TIMELINE LENGTH ANALYSIS  (post-24h window)")
    print("=" * 62)
    print(f"  Total patients analysed : {n:,}")
    print(f"  Patients ≤24h (skipped) : {n_too_short:,}")
    print(f"  No tracked event post-24h: {n_no_event:,}  ({100*n_no_event/n:.1f}%)")
    print()

    print("--- Tokens BEFORE first tracked event (= rescue continuation length) ---")
    print(f"  mean   = {pre.mean():.0f}")
    print(f"  median = {np.median(pre):.0f}")
    print(f"  p25    = {np.percentile(pre, 25):.0f}")
    print(f"  p75    = {np.percentile(pre, 75):.0f}")
    print(f"  p90    = {np.percentile(pre, 90):.0f}")
    print(f"  p99    = {np.percentile(pre, 99):.0f}")
    print(f"  max    = {pre.max()}")
    print()

    print("--- Tokens AFTER first tracked event (post-event tail) ---")
    has_event = post[post > 0]
    print(f"  (among {len(has_event):,} patients with a tracked event)")
    if len(has_event):
        print(f"  mean   = {has_event.mean():.0f}")
        print(f"  median = {np.median(has_event):.0f}")
        print(f"  p75    = {np.percentile(has_event, 75):.0f}")
        print(f"  p90    = {np.percentile(has_event, 90):.0f}")
        print(f"  max    = {has_event.max()}")
    print()

    print("--- First tracked event breakdown ---")
    for tid, count in sorted(per_event_counts.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {TRACKED_NAMES.get(tid, tid):<40s} : {count:,}")
    print()

    print("--- Summary for rescue validation ---")
    print(f"  Original M1 mean output length (early-terminated) : ~109 tokens")
    print(f"  Rescue continuation mean (observed)               : ~1,400 tokens")
    print(f"  Actual pre-event length in full timelines         : {pre.mean():.0f} tokens (mean)")
    ratio = pre.mean() / 109
    print(f"  Ratio (actual pre-event / original M1 output)     : {ratio:.1f}x")


if __name__ == "__main__":
    main()
