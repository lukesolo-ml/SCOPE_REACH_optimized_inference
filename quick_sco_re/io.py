"""
Persistence for generated trajectories and scores.

Trajectories are stored as compressed NumPy archives (.npz) grouped by patient.
Scores are stored as a single NumPy archive with M0, M1, M2 arrays.

Roundtrip correctness is validated by bench_io_roundtrip.
"""

import json
import logging
import pathlib
import warnings
from typing import Sequence

import numpy as np

from .structures import GeneratedTrajectory, GenerationConfig, PatientResults, TrajectoryType

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # Bumped from implicit v1 to v2 for inline SCOPE/REACH fields


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
        - truncation_idx: int array (-1 for None)
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

    patient_idxs = []
    sample_idxs = []
    traj_types = []
    prompt_lens = []
    was_time_truncateds = []
    truncation_idxs = []
    output_ids_flat = []
    output_ids_offsets = [0]

    timeline_terminating_ids = []

    # Inline SCOPE/REACH fields (variable-length per row, stored as flat + offsets)
    scope_flat = []
    scope_offsets = [0]
    reach_flat = []
    reach_offsets = [0]
    occurred_flag_flat = []
    occurred_flag_offsets = [0]
    occurred_index_flat = []
    occurred_index_offsets = [0]
    tracked_ids_flat = []
    tracked_ids_offsets = [0]
    tracked_names = []
    has_inline_sr = False

    for traj in trajectories:
        patient_idxs.append(traj.patient_idx)
        sample_idxs.append(traj.sample_idx)
        traj_types.append(traj.traj_type.value)
        prompt_lens.append(traj.prompt_len)
        was_time_truncateds.append(traj.was_time_truncated)
        truncation_idxs.append(
            traj.truncation_idx if traj.truncation_idx is not None else -1
        )
        timeline_terminating_ids.append(
            traj.timeline_terminating_id if traj.timeline_terminating_id is not None else -1
        )
        output_ids_flat.extend(traj.output_ids)
        output_ids_offsets.append(len(output_ids_flat))

        # Inline SCOPE/REACH arrays
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

        tracked_names.append(traj.inline_tracked_name if traj.inline_tracked_name is not None else "")

    save_dict = dict(
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int32),
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

    np.savez_compressed(output_dir / "trajectories.npz", **save_dict)

    if config is not None:
        config_dict = {
            "max_len": config.max_len,
            "n_samp": config.n_samp,
            "target_event_id": config.target_event_id,
            "end_token_ids": sorted(config.end_token_ids),
            "suppressed_ids": config.suppressed_ids,
            "temperature": config.temperature,
            "trunc_id": config.trunc_id,
            "token_id_to_minutes": {
                str(k): v for k, v in config.token_id_to_minutes.items()
            },
            "max_time": config.max_time,
            "time_check_interval": config.time_check_interval,
            "tracked_ids": config.tracked_ids,
            "tracked_name": config.tracked_name,
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

    data = np.load(input_dir / "trajectories.npz", allow_pickle=False)
    patient_idxs = data["patient_idx"]
    sample_idxs = data["sample_idx"]
    traj_types = data["traj_type"]
    prompt_lens = data["prompt_len"]
    was_time_truncateds = (
        data["was_time_truncated"]
        if "was_time_truncated" in data.files
        else [False] * len(patient_idxs)
    )
    truncation_idxs = (
        data["truncation_idx"]
        if "truncation_idx" in data.files
        else [-1] * len(patient_idxs)
    )
    timeline_terminating_ids = (
        data["timeline_terminating_id"]
        if "timeline_terminating_id" in data.files
        else [-1] * len(patient_idxs)
    )
    output_ids_flat = data["output_ids_flat"]
    output_ids_offsets = data["output_ids_offsets"]

    # Inline SCOPE/REACH fields (optional — absent in v1 files)
    has_inline_sr = "scope_flat" in data.files
    if not has_inline_sr and len(patient_idxs) > 0:
        # Check if this is an old file missing the new fields
        if "schema_version" not in data.files:
            warnings.warn(
                "Loading trajectories without inline SCOPE/REACH fields "
                "(pre-v2 schema). Inline estimates will be None.",
                stacklevel=2,
            )

    trajectories = []
    for i in range(len(patient_idxs)):
        start = output_ids_offsets[i]
        end = output_ids_offsets[i + 1]
        output_ids = output_ids_flat[start:end].tolist()

        trunc_idx_val = int(truncation_idxs[i])
        term_id_val = int(timeline_terminating_ids[i])

        # Inline SCOPE/REACH fields
        scope_estimates = None
        reach_estimates = None
        occurred_flag = None
        occurred_index = None
        inline_tracked_ids = None
        inline_tracked_name = None

        if has_inline_sr:
            s_start = int(data["scope_offsets"][i])
            s_end = int(data["scope_offsets"][i + 1])
            if s_end > s_start:
                scope_estimates = data["scope_flat"][s_start:s_end].astype(np.float64)
                reach_estimates = data["reach_flat"][
                    int(data["reach_offsets"][i]):int(data["reach_offsets"][i + 1])
                ].astype(np.float64)
                occurred_flag = data["occurred_flag_flat"][
                    int(data["occurred_flag_offsets"][i]):int(data["occurred_flag_offsets"][i + 1])
                ].astype(bool)
                occurred_index = data["occurred_index_flat"][
                    int(data["occurred_index_offsets"][i]):int(data["occurred_index_offsets"][i + 1])
                ].astype(np.int64)

            t_start = int(data["tracked_ids_offsets"][i])
            t_end = int(data["tracked_ids_offsets"][i + 1])
            if t_end > t_start:
                inline_tracked_ids = data["tracked_ids_flat"][t_start:t_end].tolist()

            name_val = str(data["tracked_name"][i])
            if name_val:
                inline_tracked_name = name_val

        trajectories.append(
            GeneratedTrajectory(
                patient_idx=int(patient_idxs[i]),
                sample_idx=int(sample_idxs[i]),
                traj_type=TrajectoryType(str(traj_types[i])),
                prompt_len=int(prompt_lens[i]),
                output_ids=output_ids,
                timeline_terminating_id=term_id_val if term_id_val >= 0 else None,
                was_time_truncated=bool(was_time_truncateds[i]),
                truncation_idx=trunc_idx_val if trunc_idx_val >= 0 else None,
                scope_estimates=scope_estimates,
                reach_estimates=reach_estimates,
                occurred_flag=occurred_flag,
                occurred_index=occurred_index,
                inline_tracked_ids=inline_tracked_ids,
                inline_tracked_name=inline_tracked_name,
            )
        )

    config = None
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            d = json.load(f)
        if "token_id_to_minutes" in d:
            d["token_id_to_minutes"] = {
                int(k): v for k, v in d["token_id_to_minutes"].items()
            }
        if "end_token_ids" in d:
            d["end_token_ids"] = set(d["end_token_ids"])
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