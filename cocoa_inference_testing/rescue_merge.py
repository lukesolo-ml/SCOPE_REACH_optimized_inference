#!/usr/bin/env python3
"""
Merge partial score files per-outcome to bound peak memory by the largest
single outcome (not the global total).
"""
import argparse, gc, json, logging, pathlib, sys
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("rescue_merge")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from quick_sco_re.io import save_scores
from quick_sco_re.structures import PatientResults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-dir", "-r", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    ap.add_argument("--n-segments", "-n", type=int, default=32)
    ap.add_argument("--keep-ndarray", action="store_true",
                    help="Skip .tolist() — pass numpy arrays straight to PatientResults.")
    args = ap.parse_args()

    rescue_dir = pathlib.Path(args.rescue_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_dirs = [rescue_dir / f"seg_{i:02d}" for i in range(args.n_segments)]
    missing = [str(d) for d in seg_dirs if not (d / "partial_scores.npz").exists()]
    if missing:
        for m in missing:
            logger.error("missing: %s", m)
        sys.exit(1)

    # Metadata + consistency check (no payload reads)
    with np.load(seg_dirs[0] / "partial_scores.npz", allow_pickle=False) as npz:
        n_patients    = int(npz["n_patients"][0])
        tracked_ids   = npz["tracked_ids"].tolist()
        tracked_names = npz["tracked_names"].tolist()
    for sd in seg_dirs[1:]:
        with np.load(sd / "partial_scores.npz", allow_pickle=False) as npz:
            assert int(npz["n_patients"][0]) == n_patients,    f"n_patients mismatch: {sd}"
            assert npz["tracked_ids"].tolist() == tracked_ids, f"tracked_ids mismatch: {sd}"

    n_outcomes = len(tracked_ids)
    logger.info("Streaming %d outcomes × %d segments, %d patients.",
                n_outcomes, args.n_segments, n_patients)

    total = 0
    for k in range(n_outcomes):
        evt_name  = tracked_names[k] if tracked_names else str(tracked_ids[k])
        safe_name = evt_name.replace("/", "_").replace(" ", "_")
        logger.info("[%2d/%d] %s", k + 1, n_outcomes, evt_name)

        chunks_p, chunks_m0, chunks_m1, chunks_m2 = [], [], [], []
        for sd in seg_dirs:
            with np.load(sd / "partial_scores.npz", allow_pickle=False) as npz:
                mask = npz["event_k_flat"] == k
                if not mask.any():
                    continue
                chunks_p.append(npz["patient_idx_flat"][mask])
                chunks_m0.append(npz["m0_flat"][mask])
                chunks_m1.append(npz["m1_flat"][mask])
                chunks_m2.append(npz["m2_flat"][mask])

        if chunks_p:
            k_patient = np.concatenate(chunks_p); chunks_p.clear()
            k_m0      = np.concatenate(chunks_m0); chunks_m0.clear()
            k_m1      = np.concatenate(chunks_m1); chunks_m1.clear()
            k_m2      = np.concatenate(chunks_m2); chunks_m2.clear()
        else:
            k_patient = np.empty(0, dtype=np.int64)
            k_m0 = k_m1 = k_m2 = np.empty(0, dtype=np.float64)

        n_k = len(k_patient)
        total += n_k
        logger.info("    %s contributions; sort+group", f"{n_k:,}")

        if n_k:
            order     = np.argsort(k_patient, kind="stable")
            k_patient = k_patient[order]
            k_m0      = k_m0[order]
            k_m1      = k_m1[order]
            k_m2      = k_m2[order]
            del order

        results_k = [PatientResults() for _ in range(n_patients)]
        n_with_data = 0
        if n_k:
            splits = np.flatnonzero(np.diff(k_patient)) + 1
            p_groups  = np.split(k_patient, splits)
            m0_groups = np.split(k_m0, splits)
            m1_groups = np.split(k_m1, splits)
            m2_groups = np.split(k_m2, splits)
            n_with_data = len(p_groups)
            for pg, m0g, m1g, m2g in zip(p_groups, m0_groups, m1_groups, m2_groups):
                p = int(pg[0])
                if args.keep_ndarray:
                    results_k[p].m0_samples = m0g
                    results_k[p].m1_samples = m1g
                    results_k[p].m2_samples = m2g
                else:
                    results_k[p].m0_samples = m0g.tolist()
                    results_k[p].m1_samples = m1g.tolist()
                    results_k[p].m2_samples = m2g.tolist()

        del k_patient, k_m0, k_m1, k_m2
        # Quick instrumentation to add temporarily:
        import resource, os
        def rss_gb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # Linux: KB → GB
        logger.info("RSS now %.2f GB", rss_gb())
        save_scores(results_k, output_dir / f"scores_{safe_name}.npz")
        del results_k
        gc.collect()
        logger.info("    saved (%d patients with data)", n_with_data)

    with open(output_dir / "merge_manifest.json", "w") as f:
        json.dump({
            "n_segments":    args.n_segments,
            "n_patients":    n_patients,
            "tracked_ids":   tracked_ids,
            "tracked_names": tracked_names,
            "total_samples": int(total),
        }, f, indent=2)
    logger.info("done → %s", output_dir)


if __name__ == "__main__":
    main()