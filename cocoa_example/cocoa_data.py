"""cocoa adapter: read cocoa's winnower outputs and turn a YAML config into quick_sco_re inputs.

cocoa (github.com/bbj-lab/cocoa) tokenizes EHR timelines and its winnower writes, per split,
``<split>_for_inference.parquet``: one row per subject with

  tokens_past        prompt token ids (the timeline up to the prediction cut, e.g. 24 h)
  s_elapsed_past     seconds since the timeline's first event, per prompt token
  times, start_time  every event's timestamp, and the subject's start time
  <EVENT>_past       the event already happened in the prompt   (bool)
  <EVENT>_future     the event happens after the cut: GROUND TRUTH (bool)

and ``tokenizer.yaml`` with a ``lookup:`` map from token name to id and, under ``cfg``, the
tokenizer settings (``insert_spacers``, ``spacers``) the time-gap bins come from.
Written against cocoa-tokenizer 26.11. Nothing here needs a GPU.
"""

from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import yaml

from quick_sco_re import GenerationConfig, silence_restriction_bias

logger = logging.getLogger(__name__)

# cocoa's default spacers (tokenization.yaml): label -> lower edge in minutes. A gap of g minutes
# (floored) gets the last label whose edge g reaches, gaps under the first edge get none, and
# the last bin is open-ended; so the bins are [5, 15], (15, 60], ..., (262980, inf).
COCOA_DEFAULT_SPACERS = {
    "5m-15m": 5, "15m-1h": 15, "1h-2h": 60, "2h-6h": 120, "6h-12h": 360, "12h-1d": 720, "1d-3d": 1440,
    "3d-1w": 4320, "1w-2w": 10080, "2w-1mt": 20160, "1mt-3mt": 43200, "3mt-6mt": 131490, "6mt+": 262980,
}


def spacer_bins(spacers: dict) -> dict[str, tuple[float, float]]:
    """``TIME//<label>`` -> (lo, hi] minutes from a cocoa ``spacers`` map (label -> lower edge).

    cocoa labels a gap by the position of its bin in the map, not by the edges' values, so its
    labels are right only when the edges increase in the order given. Anything else is refused:
    the timelines were labelled inconsistently and no bin table describes them.
    """
    items = [(str(k), float(v)) for k, v in spacers.items()]
    edges = [v for _, v in items]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"tokenizer spacers must increase in the order listed, got {dict(items)}: cocoa assigns "
                         "TIME// labels by position, so these timelines were labelled with the wrong bins")
    return {f"TIME//{k}": (lo, items[i + 1][1] if i + 1 < len(items) else math.inf)
            for i, (k, lo) in enumerate(items)}


COCOA_TIME_BINS = spacer_bins(COCOA_DEFAULT_SPACERS)


def load_config(path: str | pathlib.Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("cocoa_outputs", "model_path", "output_dir", "generation"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    return cfg


class CocoaVocab:
    """Token name <-> id from cocoa's tokenizer.yaml."""

    def __init__(self, tokenizer_yaml: str | pathlib.Path):
        with open(tokenizer_yaml) as f:
            data = yaml.safe_load(f)
        self.name_to_id: dict[str, int] = {str(k): int(v) for k, v in data["lookup"].items()}
        self.id_to_name: dict[int, str] = {v: k for k, v in self.name_to_id.items()}
        self.size = max(self.id_to_name) + 1
        self.tokenizer_cfg: dict = data.get("cfg") or {}
        self.cocoa_version: str | None = data.get("cocoa_version")

    def time_bins(self) -> dict[str, tuple[float, float]]:
        """The ``TIME//`` bins this vocabulary was built with, (lo, hi] minutes, from the saved
        tokenizer config (cocoa's defaults for a tokenizer.yaml without one); only bins in the vocab."""
        spacers = self.tokenizer_cfg.get("spacers") or COCOA_DEFAULT_SPACERS
        return {n: b for n, b in spacer_bins(spacers).items() if n in self}

    def __call__(self, name: str) -> int:
        if name not in self.name_to_id:
            raise KeyError(f"token '{name}' is not in the vocabulary")
        return self.name_to_id[name]

    def __contains__(self, name: str) -> bool:
        return name in self.name_to_id

    def ids_with_prefix(self, prefix: str) -> set[int]:
        return {i for n, i in self.name_to_id.items() if n.startswith(prefix + "//") or n == prefix}


@dataclass
class Cohort:
    subject_ids: list[str]
    prompts: list[list[int]]            # prompt token ids (prefix tokens included)
    labels: pl.DataFrame                # subject_id + every <EVENT>_past / <EVENT>_future flag
    silence_minutes: np.ndarray | None  # minutes from the last prompt event to the cut
    n_truncated: int = 0
    n_dropped: dict = field(default_factory=dict)

    def past_mask(self, event: str) -> np.ndarray:
        col = f"{event}_past"
        return self.labels[col].to_numpy().astype(bool) if col in self.labels.columns else np.zeros(len(self.prompts), bool)

    def future(self, event: str) -> np.ndarray | None:
        col = f"{event}_future"
        return self.labels[col].to_numpy().astype(bool) if col in self.labels.columns else None


def end_token_ids(cfg: dict, vocab: CocoaVocab) -> set[int]:
    """Tokens that end a trajectory: ``generation.end_tokens`` prefixes and names.

    The default is every ``DSCG//`` token plus ``EOS``, which cocoa puts at the end of every
    timeline. Without EOS as an end token, a trajectory that reaches it runs on into what the
    model learned follows EOS in packed training blocks: another stay.
    """
    end = cfg["generation"].get("end_tokens")
    if end is None:
        end = {"prefixes": ["DSCG"], "names": ["EOS"] if "EOS" in vocab else []}
    ids: set[int] = set()
    for p in end.get("prefixes") or []:
        ids |= vocab.ids_with_prefix(p)
    for n in end.get("names") or []:
        ids.add(vocab(n))
    if not ids:
        raise ValueError("generation.end_tokens matches no token: trajectories would never end")
    if "EOS" in vocab and vocab("EOS") not in ids:
        logger.warning("EOS is not an end token: trajectories that reach it continue into a new timeline")
    return ids


def load_cohort(cfg: dict, vocab: CocoaVocab, split_file: str | None = None) -> Cohort:
    """The patients to score, in a fixed order, with their prompts and labels.

    Drops (counts in Cohort.n_dropped): rows failing cohort.require_flag / exclude_flag, prompts that
    already hold an end token (the stay ended before the cut), and, with prompt_overflow: drop, prompts
    longer than generation.max_len (truncate_left keeps their most recent tokens instead).
    """
    path = split_file or cfg["cocoa_outputs"]["held_out_for_inference"]
    df = pl.read_parquet(path)
    cohort_cfg = cfg.get("cohort") or {}
    gen = cfg["generation"]
    dropped = {}
    flag_cols = [c for c in df.columns if c.endswith(("_past", "_future")) and df[c].dtype == pl.Boolean]

    if cohort_cfg.get("require_flag"):
        n = df.height; df = df.filter(pl.col(cohort_cfg["require_flag"])); dropped["require_flag"] = n - df.height
    if cohort_cfg.get("exclude_flag"):
        n = df.height; df = df.filter(~pl.col(cohort_cfg["exclude_flag"])); dropped["exclude_flag"] = n - df.height
    ends = sorted(end_token_ids(cfg, vocab))
    n = df.height
    df = df.filter(~pl.col("tokens_past").list.eval(pl.element().is_in(ends)).list.any())
    dropped["ended_in_prompt"] = n - df.height
    if cohort_cfg.get("max_patients") is not None and df.height > cohort_cfg["max_patients"]:
        df = df.sample(n=int(cohort_cfg["max_patients"]), shuffle=True, seed=int(cohort_cfg.get("shuffle_seed", 42)))

    if "valid_future_count" in df.columns:
        logger.warning("the winnower ran with horizon_after_threshold_s: <EVENT>_future flags count only events "
                       "inside that window, while trajectories run to discharge or max_len")

    silence = cut_silence_minutes(df, gen.get("silence_restriction") or {})

    prefix = [vocab(n) for n in gen.get("prompt_prefix_tokens") or []]
    max_len = int(gen["max_len"])
    overflow = cfg.get("prompt_overflow", "truncate_left")
    keep, prompts, n_trunc = [], [], 0
    for toks in df["tokens_past"].to_list():
        toks = prefix + [int(t) for t in toks]
        if len(toks) > max_len:
            if overflow == "drop":
                keep.append(False); continue
            toks = toks[-max_len:]; n_trunc += 1
        keep.append(True); prompts.append(toks)
    km = np.array(keep, bool)
    dropped["too_long"] = int((~km).sum())
    df = df.filter(pl.Series(keep))
    labels = df.select(["subject_id", *flag_cols])
    logger.info(f"cohort: {len(prompts)} patients; dropped {dropped}; {n_trunc} prompts left-truncated")
    return Cohort(subject_ids=[str(s) for s in df["subject_id"].to_list()], prompts=prompts, labels=labels,
                  silence_minutes=None if silence is None else silence[km], n_truncated=n_trunc, n_dropped=dropped)


def cut_silence_minutes(df: pl.DataFrame, sr: dict) -> np.ndarray | None:
    """Minutes from each prompt's last token to the prediction cut, or None when unknown.

    ``silence_restriction.reference`` says where the winnower measured its cut from:
      first_event  (cocoa's ``threshold.duration_s`` / ``horizon_s``) the cut is
                   ``horizon_minutes`` after the timeline's first event; uses ``s_elapsed_past``.
      start_time   (``threshold.after_start_time_s``) the cut is ``horizon_minutes`` after the
                   subject's ``start_time``; uses ``times`` and ``start_time``.
    A ``first_occurrence`` cut ends the prompt on its trigger token, with no silence: leave the
    restriction off.
    """
    ref = sr.get("reference", "first_event")
    horizon = float(sr.get("horizon_minutes", 1440))
    if ref == "first_event":
        tcol = sr.get("time_column", "s_elapsed_past")
        if tcol not in df.columns:
            if sr.get("enabled"):
                raise ValueError(f"silence_restriction needs the column '{tcol}'")
            return None
        last = (df[tcol].list.last().fill_null(0) / 60.0).to_numpy().astype(float)
    elif ref == "start_time":
        missing = [c for c in ("times", "start_time", "tokens_past") if c not in df.columns]
        if missing:
            if sr.get("enabled"):
                raise ValueError(f"silence_restriction with reference: start_time needs the columns {missing}")
            return None
        last = df.select((pl.col("times").list.get(pl.col("tokens_past").list.len() - 1) - pl.col("start_time"))
                         .dt.total_milliseconds() / 60000.0).to_series().to_numpy().astype(float)
    else:
        raise ValueError(f"silence_restriction.reference must be first_event or start_time, got {ref!r}")
    if sr.get("enabled") and (last > horizon + 1e-6).any():
        raise ValueError(f"{int((last > horizon + 1e-6).sum())} prompts have events after horizon_minutes={horizon} "
                         f"(reference: {ref}); both must match the winnower's prediction cut")
    return horizon - last


def time_token_bounds(cfg: dict, vocab: CocoaVocab) -> dict[int, tuple[float, float]]:
    """Time-gap token id -> (lo, hi] minutes: the config's bounds, else the bins the vocab was built with."""
    gen = cfg["generation"]
    named = ((gen.get("silence_restriction") or {}).get("time_token_bounds")
             or (gen.get("time_stopping") or {}).get("time_token_bounds") or vocab.time_bins())
    return {vocab(n): (float(lo), float(hi)) for n, (lo, hi) in named.items() if n in vocab}


def first_token_biases(cfg: dict, vocab: CocoaVocab, cohort: Cohort) -> list[dict | None] | None:
    """Per-patient silence-restriction bias (None for everyone when the restriction is off)."""
    sr = cfg["generation"].get("silence_restriction") or {}
    if not sr.get("enabled"):
        return None
    bounds = time_token_bounds(cfg, vocab)
    if not bounds:
        raise ValueError("silence_restriction needs TIME// tokens, and this vocabulary has none (cocoa tokenizes "
                         "with insert_spacers: false by default); turn the restriction off or re-tokenize with spacers")
    ids = range(vocab.size)
    return [silence_restriction_bias(float(x), bounds, ids, mode=sr.get("mode", "hard"), min_minutes=sr.get("min_minutes"))
            for x in cohort.silence_minutes]


def build_generation_config(cfg: dict, vocab: CocoaVocab) -> GenerationConfig:
    """GenerationConfig for multi-outcome inline SCOPE/REACH: every tracked event gets inline
    estimates from one M1 pass; the first is the nominal target."""
    gen = cfg["generation"]
    names = list(gen["tracked_events"])
    tracked = [vocab(n) for n in names]
    suppressed = [vocab(n) for n in gen.get("suppressed_tokens", ["BOS"]) if n in vocab]
    rw = gen.get("rolling_window") or {}
    window = keep = None
    if rw.get("enabled"):
        window = int(rw["context_window"])
        keep = int(rw.get("keep") or window - window // 4)
    ts = gen.get("time_stopping") or {}
    trunc_id, minutes, max_time = None, {}, None
    if ts.get("enabled"):
        trunc_id = vocab(ts.get("trunc_token", "TRUNC"))
        max_time = float(ts["max_time_minutes"])
        minutes = {vocab(n): math.sqrt(lo * hi) if math.isfinite(hi) else float(lo)
                   for n, (lo, hi) in (ts.get("time_token_bounds") or vocab.time_bins()).items() if n in vocab}
        suppressed = [s for s in suppressed if s != trunc_id]
    return GenerationConfig(
        max_len=int(gen["max_len"]), n_samp=int(gen["n_samp"]), target_event_id=tracked[0],
        end_token_ids=end_token_ids(cfg, vocab), suppressed_ids=suppressed, temperature=float(gen.get("temperature", 1.0)),
        trunc_id=trunc_id, token_id_to_minutes=minutes, max_time=max_time,
        time_check_interval=int(ts.get("time_check_interval", 100)),
        tracked_ids=tracked, tracked_names=names, context_window=window, window_keep=keep,
    )
