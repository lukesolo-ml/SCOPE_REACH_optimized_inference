#!/usr/bin/env python3
"""
Analyze generated output_ids lengths for early-stopped vs correct M1 trajectories
in scope_reach_output/trajectories/trajectories.npz.

Length is computed from offset arithmetic — the large output_ids_flat array is
never decompressed.
"""

import pathlib
import numpy as np

TRAJ_NPZ = pathlib.Path(__file__).parent / "scope_reach_output/trajectories/trajectories.npz"

TRACKED_IDS = {1370, 1213, 121, 611, 620, 548, 557, 388, 377, 1300, 1267, 1262, 1257, 1252, 61}
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

def pct(arr, p):
    return int(np.percentile(arr, p))

def stats(label, arr):
    print(f"\n--- {label} (n={len(arr):,}) ---")
    print(f"  mean   = {arr.mean():.1f}")
    print(f"  median = {pct(arr, 50)}")
    print(f"  p25    = {pct(arr, 25)}")
    print(f"  p75    = {pct(arr, 75)}")
    print(f"  p90    = {pct(arr, 90)}")
    print(f"  p99    = {pct(arr, 99)}")
    print(f"  max    = {arr.max()}")

def main():
    print(f"Loading metadata from {TRAJ_NPZ} ...")
    with np.load(TRAJ_NPZ, allow_pickle=False) as npz:
        traj_type  = npz["traj_type"]
        term_ids   = npz["timeline_terminating_id"]
        offsets    = npz["output_ids_offsets"]
        prompt_len = npz["prompt_len"]

    lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)

    m1_mask      = traj_type == "m1"
    early_mask   = m1_mask & np.isin(term_ids, sorted(TRACKED_IDS))
    correct_mask = m1_mask & ~early_mask

    print(f"\nTotal trajectories   : {len(traj_type):,}")
    print(f"M1 trajectories      : {m1_mask.sum():,}")
    print(f"  Early-stopped (bug): {early_mask.sum():,}")
    print(f"  Correct            : {correct_mask.sum():,}")

    stats("Early-stopped M1 generated length (output_ids)", lengths[early_mask])
    stats("Correct M1 generated length (output_ids)",       lengths[correct_mask])
    stats("All M1 generated length",                        lengths[m1_mask])

    print("\n--- Early-stopped: breakdown by terminating token ---")
    from collections import Counter
    counts = Counter(int(t) for t in term_ids[early_mask])
    for tid, n in counts.most_common():
        print(f"  {TRACKED_NAMES.get(tid, tid):<40s}: {n:,}")

    print("\n--- Prompt lengths (all M1) ---")
    pl = prompt_len[m1_mask].astype(np.int64)
    print(f"  mean={pl.mean():.0f}  median={pct(pl,50)}  p75={pct(pl,75)}  max={pl.max()}")

    print("\n--- Effective rescue prompt (orig_prompt + early_output) ---")
    ep = (prompt_len[early_mask] + lengths[early_mask]).astype(np.int64)
    print(f"  mean={ep.mean():.0f}  median={pct(ep,50)}  p75={pct(ep,75)}  max={ep.max()}")

if __name__ == "__main__":
    main()
