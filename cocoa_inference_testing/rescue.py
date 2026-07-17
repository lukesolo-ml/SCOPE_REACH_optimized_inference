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
import pathlib
import sys
from collections import Counter, defaultdict

import polars as pl
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from quick_sco_re.generation import generate_trajectory, generate_m2_from_m1_trajectory
from quick_sco_re.io import load_trajectories, save_trajectories, save_scores
from quick_sco_re.scheduler import create_engine
from quick_sco_re.scoring import score_trajectory
from quick_sco_re.structures import GeneratedTrajectory, GenerationConfig, PatientResults, TrajectoryType


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def identify_early_terminated(
    traj_dir: pathlib.Path,
) -> tuple[list, list, list, set, object]:
    trajectories, config = load_trajectories(traj_dir)

    tracked_ids_set: set[int] = set()
    if config is not None and config.tracked_ids:
        tracked_ids_set = set(config.tracked_ids)
    else:
        for traj in trajectories:
            if traj.inline_tracked_ids:
                tracked_ids_set = set(traj.inline_tracked_ids)
                break

    all_m1 = [t for t in trajectories if t.traj_type == TrajectoryType.M1]
    early_terminated, correct = [], []
    for traj in all_m1:
        term_id = traj.timeline_terminating_id
        if term_id is not None and term_id in tracked_ids_set:
            early_terminated.append(traj)
        else:
            correct.append(traj)

    return all_m1, early_terminated, correct, tracked_ids_set, config


def print_summary(all_m1, early_terminated, correct, tracked_ids_set, config):
    n_total = len(all_m1)
    if n_total == 0:
        print("No M1 trajectories found.")
        return

    n_early = len(early_terminated)
    n_correct = len(correct)

    print(f"\n=== Trajectory Classification Summary ===")
    print(f"Tracked outcome token IDs ({len(tracked_ids_set)}): {sorted(tracked_ids_set)}")
    print(f"Total M1 trajectories : {n_total:>8,}")
    print(f"  Correct (natural end): {n_correct:>8,}  ({n_correct/n_total:.1%})")
    print(f"  Early terminated     : {n_early:>8,}  ({n_early/n_total:.1%})")

    if n_early == 0:
        print("\nNo early-terminated trajectories found — nothing to rescue.")
        return

    term_counts = Counter(t.timeline_terminating_id for t in early_terminated)
    print(f"\nEarly terminations by token ID:")
    for token_id, count in term_counts.most_common():
        name = "(unknown)"
        if config is not None and config.tracked_names and config.tracked_ids:
            try:
                idx = list(config.tracked_ids).index(token_id)
                name = config.tracked_names[idx]
            except (ValueError, IndexError):
                pass
        print(f"  token {token_id:6d}  ({name}): {count:,} trajectories")

    by_patient: dict[int, list] = defaultdict(list)
    for t in early_terminated:
        by_patient[t.patient_idx].append(t)
    n_patients_affected = len(by_patient)
    total_patients = len(set(t.patient_idx for t in all_m1))
    print(f"\nPatients affected: {n_patients_affected} / {total_patients} "
          f"({n_patients_affected/total_patients:.1%})")

    samples_per_patient = sorted(len(v) for v in by_patient.values())
    print(f"Early-terminated samples per affected patient: "
          f"min={samples_per_patient[0]}, "
          f"median={samples_per_patient[len(samples_per_patient)//2]}, "
          f"max={samples_per_patient[-1]}")

    lengths = [len(t.output_ids) for t in early_terminated]
    print(f"\nEarly-terminated output_ids lengths: "
          f"min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
    lengths_correct = [len(t.output_ids) for t in correct]
    if lengths_correct:
        print(f"Correct output_ids lengths:          "
              f"min={min(lengths_correct)}, median={sorted(lengths_correct)[len(lengths_correct)//2]}, max={max(lengths_correct)}")


# ---------------------------------------------------------------------------
# Patient token loading
# ---------------------------------------------------------------------------

def load_patient_tokens(output_dir: pathlib.Path, cfg: dict) -> list[list[int]]:
    """Load patient prompt tokens in the original run's patient_idx order.

    Uses patient_index.parquet (saved by the original run) for ordering,
    then retrieves tokens_past from the original parquet by subject_id join.
    """
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
# Continuation generation
# ---------------------------------------------------------------------------

async def generate_continuations(
    engine,
    config,
    early_terminated: list[GeneratedTrajectory],
    patient_tokens: list[list[int]],
) -> list[GeneratedTrajectory]:
    """Generate continuations for each early-terminated M1 trajectory.

    The prefix fed to the model is prompt_tokens + original output_ids,
    which includes the tracked event token that caused early stopping.
    Generation then continues to the natural end of the timeline.
    Inline SCOPE/REACH is computed over the continuation portion only.
    """
    tasks = [
        generate_trajectory(
            engine=engine,
            config=config,
            prompt_tokens=patient_tokens[traj.patient_idx] + traj.output_ids,
            patient_idx=traj.patient_idx,
            sample_idx=traj.sample_idx,
            traj_type=TrajectoryType.M1,
            stop_at_tracked_events=False,
        )
        for traj in early_terminated
    ]
    print(f"Generating {len(tasks)} continuations...")
    return list(await asyncio.gather(*tasks))


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
        scope = original.scope_estimates.copy()
        reach = original.reach_estimates.copy()
        occurred_flag = original.occurred_flag.copy()
        occurred_index = original.occurred_index.copy()

        for k, tid in enumerate(tracked_ids):
            if original.timeline_terminating_id == tid:
                # Event caused the early stop — SCOPE/REACH already accumulated to occurrence
                pass
            else:
                scope[k] = original.scope_estimates[k] + continuation.scope_estimates[k]
                reach[k] = 1.0 - (1.0 - original.reach_estimates[k]) * (1.0 - continuation.reach_estimates[k])
                # If event now occurs in the continuation, update occurrence tracking
                if (
                    not occurred_flag[k]
                    and continuation.occurred_flag is not None
                    and continuation.occurred_flag[k]
                ):
                    occurred_flag[k] = True
                    occurred_index[k] = orig_len + int(continuation.occurred_index[k])
    else:
        scope = original.scope_estimates
        reach = original.reach_estimates
        occurred_flag = original.occurred_flag
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
# M2 score derivation
# ---------------------------------------------------------------------------

async def derive_m2_scores(
    engine,
    config: GenerationConfig,
    correct: list[GeneratedTrajectory],
    corrected: list[GeneratedTrajectory],
    continuations: list[GeneratedTrajectory],
    patient_tokens: list[list[int]],
    n_patients: int,
) -> list[list[PatientResults]]:
    """Compute per-outcome M0/M1/M2 for all corrected M1 trajectories.

    Case A (correct): No tracked events in output — inline REACH is valid.
    Case B (corrected, no tracked events in continuation): Inline REACH from
        merge_trajectory is valid.
    Case B2 (corrected, tracked event k appears in continuation): Regenerate
        M2 for outcome k by truncating the merged M1 before the first occurrence
        and regenerating with event k suppressed, then score.
    """
    tracked_ids = config.tracked_ids
    n_outcomes = len(tracked_ids)
    results = [[PatientResults() for _ in range(n_patients)] for _ in range(n_outcomes)]

    # Case A: correct trajectories — inline estimates are valid as-is
    for traj in correct:
        for k, tid in enumerate(tracked_ids):
            ki = traj.inline_tracked_ids.index(tid)
            results[k][traj.patient_idx].m0_samples.append(bool(traj.occurred_flag[ki]))
            results[k][traj.patient_idx].m1_samples.append(float(traj.scope_estimates[ki]))
            results[k][traj.patient_idx].m2_samples.append(float(traj.reach_estimates[ki]))

    # Case B / B2: corrected (merged) trajectories
    b2_tasks = []  # (merged, k, outcome_config) requiring M2 regen

    for merged, cont in zip(corrected, continuations):
        cont_tracked_ks = {
            k for k, tid in enumerate(tracked_ids) if tid in cont.output_ids
        }

        for k, tid in enumerate(tracked_ids):
            ki = merged.inline_tracked_ids.index(tid)
            results[k][merged.patient_idx].m0_samples.append(bool(merged.occurred_flag[ki]))
            results[k][merged.patient_idx].m1_samples.append(float(merged.scope_estimates[ki]))

            if k in cont_tracked_ks:
                outcome_config = dataclasses.replace(config, target_event_id=tid)
                b2_tasks.append((merged, k, outcome_config))
            else:
                results[k][merged.patient_idx].m2_samples.append(float(merged.reach_estimates[ki]))

    # Case B2: regenerate M2 for each (trajectory, outcome) pair
    if b2_tasks:
        print(f"Regenerating M2 for {len(b2_tasks)} Case B2 (outcome × trajectory) pairs...")
        # Fake the terminal ID so generate_m2_from_m1_trajectory takes the regen path
        m2_coros = [
            generate_m2_from_m1_trajectory(
                engine,
                oc,
                dataclasses.replace(merged, timeline_terminating_id=oc.target_event_id),
                patient_tokens[merged.patient_idx],
            )
            for merged, _, oc in b2_tasks
        ]
        m2_trajectories = list(await asyncio.gather(*m2_coros))

        score_coros = [
            score_trajectory(engine, oc, m2_traj, patient_tokens[merged.patient_idx])
            for (merged, _, oc), m2_traj in zip(b2_tasks, m2_trajectories)
        ]
        scored = list(await asyncio.gather(*score_coros))

        for (merged, k, _), st in zip(b2_tasks, scored):
            results[k][merged.patient_idx].m2_samples.append(st.score)

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_rescue(args):
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    traj_dir = output_dir / "trajectories"

    if not (traj_dir / "trajectories.npz").exists():
        print(f"ERROR: trajectories.npz not found in {traj_dir}")
        sys.exit(1)

    print(f"Loading trajectories from: {traj_dir}")
    all_m1, early_terminated, correct, tracked_ids_set, config = identify_early_terminated(traj_dir)
    print_summary(all_m1, early_terminated, correct, tracked_ids_set, config)

    if not early_terminated:
        return

    cfg_path = pathlib.Path(args.config).expanduser().resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\nLoading patient tokens from parquet...")
    patient_tokens = load_patient_tokens(output_dir, cfg)
    print(f"Loaded {len(patient_tokens)} patient prompts.")

    mem_frac = cfg.get("engine", {}).get("mem_fraction", 0.85)
    engine = create_engine(
        model_path=str(pathlib.Path(cfg["model_path"]).expanduser().resolve()),
        max_len=config.max_len,
        use_time_horizon=False,
        mem_fraction=mem_frac,
    )

    try:
        continuations = await generate_continuations(engine, config, early_terminated, patient_tokens)

        corrected = [
            merge_trajectory(orig, cont)
            for orig, cont in zip(early_terminated, continuations)
        ]

        all_corrected_m1 = correct + corrected
        print(f"\nRescue complete: {len(corrected)} trajectories corrected.")
        print(f"Total M1 trajectories ready: {len(all_corrected_m1)}")

        n_patients = len(patient_tokens)
        print(f"\nDeriving M2 scores...")
        results_per_outcome = await derive_m2_scores(
            engine, config, correct, corrected, continuations, patient_tokens, n_patients
        )
    finally:
        engine.shutdown()

    rescue_dir = pathlib.Path(args.rescue_dir).expanduser().resolve()
    rescue_dir.mkdir(parents=True, exist_ok=True)

    traj_dir = rescue_dir / "trajectories"
    save_trajectories(all_corrected_m1, traj_dir, config=config)
    print(f"Saved {len(all_corrected_m1)} M1 trajectories to {traj_dir}")

    tracked_names = config.tracked_names or []
    for k, evt_name in enumerate(tracked_names):
        safe_name = evt_name.replace("/", "_").replace(" ", "_")
        scores_path = rescue_dir / f"scores_{safe_name}.npz"
        save_scores(results_per_outcome[k], scores_path)
        print(f"  Saved {scores_path.name}")

    print(f"\nRescue output written to {rescue_dir}")


def main():
    parser = argparse.ArgumentParser(description="Rescue early-terminated M1 trajectories.")
    parser.add_argument("--output-dir", "-o", default="./scope_reach_output")
    parser.add_argument("--rescue-dir", "-r", default="./scope_reach_output_rescue")
    parser.add_argument("--config", "-c", default="./pipeline_config.yaml")
    args = parser.parse_args()

    asyncio.run(run_rescue(args))


if __name__ == "__main__":
    main()
