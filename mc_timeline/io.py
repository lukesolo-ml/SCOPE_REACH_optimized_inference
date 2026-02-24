"""
Persistence for generated trajectories and scores.

Trajectories are stored as compressed NumPy archives (.npz) grouped by patient.
Scores are stored as a single NumPy archive with M0, M1, M2 arrays.
"""

import json
import pathlib
from typing import Sequence

import numpy as np

from .types import GeneratedTrajectory, GenerationConfig, PatientResults, TrajectoryType


def save_trajectories(
    trajectories: Sequence[GeneratedTrajectory],
    output_dir: pathlib.Path | str,
    config: GenerationConfig | None = None,
) -> pathlib.Path:
    """Save generated trajectories to disk.

    Creates a directory structure:
        output_dir/
            config.json          (generation config, if provided)
            trajectories.npz     (all trajectory data)

    Trajectory data is stored as parallel arrays for efficient loading:
        - patient_idx, sample_idx: int arrays
        - traj_type: string array ("m1" / "m2")
        - prompt_len: int array
        - has_timeline_end, has_target_event: bool arrays
        - output_ids_flat: concatenated token IDs
        - output_ids_offsets: offsets into output_ids_flat for each trajectory

    Args:
        trajectories: List of generated trajectories.
        output_dir: Directory to save to (created if needed).
        config: Optional generation config to save alongside trajectories.

    Returns:
        Path to the output directory.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parallel arrays
    patient_idxs = []
    sample_idxs = []
    traj_types = []
    prompt_lens = []
    has_timeline_ends = []
    has_target_events = []
    output_ids_flat = []
    output_ids_offsets = [0]

    was_time_truncateds = []

    for traj in trajectories:
        patient_idxs.append(traj.patient_idx)
        sample_idxs.append(traj.sample_idx)
        traj_types.append(traj.traj_type.value)
        prompt_lens.append(traj.prompt_len)
        has_timeline_ends.append(traj.has_timeline_end)
        has_target_events.append(traj.has_target_event)
        was_time_truncateds.append(traj.was_time_truncated)
        output_ids_flat.extend(traj.output_ids)
        output_ids_offsets.append(len(output_ids_flat))

    np.savez_compressed(
        output_dir / "trajectories.npz",
        patient_idx=np.array(patient_idxs, dtype=np.int32),
        sample_idx=np.array(sample_idxs, dtype=np.int32),
        traj_type=np.array(traj_types, dtype="U2"),
        prompt_len=np.array(prompt_lens, dtype=np.int32),
        has_timeline_end=np.array(has_timeline_ends, dtype=bool),
        has_target_event=np.array(has_target_events, dtype=bool),
        was_time_truncated=np.array(was_time_truncateds, dtype=bool),
        output_ids_flat=np.array(output_ids_flat, dtype=np.int32),
        output_ids_offsets=np.array(output_ids_offsets, dtype=np.int64),
    )

    if config is not None:
        config_dict = {
            "max_len": config.max_len,
            "n_samp": config.n_samp,
            "target_event_id": config.target_event_id,
            "timeline_end_id": config.timeline_end_id,
            "suppressed_ids": config.suppressed_ids,
            "temperature": config.temperature,
            "trunc_id": config.trunc_id,
            # JSON requires string keys; load_trajectories converts them back to int.
            "token_id_to_minutes": {
                str(k): v for k, v in config.token_id_to_minutes.items()
            },
            "max_time": config.max_time,
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    return output_dir


def load_trajectories(
    input_dir: pathlib.Path | str,
) -> tuple[list[GeneratedTrajectory], GenerationConfig | None]:
    """Load trajectories from disk.

    Args:
        input_dir: Directory containing trajectories.npz and optionally config.json.

    Returns:
        Tuple of (list of trajectories, config or None).
    """
    input_dir = pathlib.Path(input_dir)

    data = np.load(input_dir / "trajectories.npz")
    patient_idxs = data["patient_idx"]
    sample_idxs = data["sample_idx"]
    traj_types = data["traj_type"]
    prompt_lens = data["prompt_len"]
    has_timeline_ends = data["has_timeline_end"]
    has_target_events = data["has_target_event"]
    # was_time_truncated was added later; default to False for files saved before.
    was_time_truncateds = (
        data["was_time_truncated"]
        if "was_time_truncated" in data.files
        else [False] * len(patient_idxs)
    )
    output_ids_flat = data["output_ids_flat"]
    output_ids_offsets = data["output_ids_offsets"]

    trajectories = []
    for i in range(len(patient_idxs)):
        start = output_ids_offsets[i]
        end = output_ids_offsets[i + 1]
        output_ids = output_ids_flat[start:end].tolist()

        trajectories.append(
            GeneratedTrajectory(
                patient_idx=int(patient_idxs[i]),
                sample_idx=int(sample_idxs[i]),
                traj_type=TrajectoryType(str(traj_types[i])),
                prompt_len=int(prompt_lens[i]),
                output_ids=output_ids,
                has_timeline_end=bool(has_timeline_ends[i]),
                has_target_event=bool(has_target_events[i]),
                was_time_truncated=bool(was_time_truncateds[i]),
            )
        )

    config = None
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            d = json.load(f)
        # token_id_to_minutes is saved with string keys (JSON limitation);
        # convert back to int.
        if "token_id_to_minutes" in d:
            d["token_id_to_minutes"] = {
                int(k): v for k, v in d["token_id_to_minutes"].items()
            }
        config = GenerationConfig(**d)

    return trajectories, config


def save_scores(
    results: Sequence[PatientResults],
    output_path: pathlib.Path | str,
) -> pathlib.Path:
    """Save aggregated patient scores to disk.

    Saves three arrays (M0, M1, M2) where each element is the mean
    estimator value for a patient.

    Args:
        results: Per-patient results from the scheduler.
        output_path: Path for the output .npz file.

    Returns:
        Path to the saved file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    M0 = np.array([np.mean(r.m0_samples) if r.m0_samples else np.nan for r in results])
    M1 = np.array([np.mean(r.m1_samples) if r.m1_samples else np.nan for r in results])
    M2 = np.array([np.mean(r.m2_samples) if r.m2_samples else np.nan for r in results])

    np.savez_compressed(output_path, M0=M0, M1=M1, M2=M2)
    return output_path


def load_scores(
    input_path: pathlib.Path | str,
) -> dict[str, np.ndarray]:
    """Load saved scores.

    Args:
        input_path: Path to the .npz scores file.

    Returns:
        Dict with keys "M0", "M1", "M2" mapping to numpy arrays.
    """
    data = np.load(input_path)
    return {"M0": data["M0"], "M1": data["M1"], "M2": data["M2"]}
