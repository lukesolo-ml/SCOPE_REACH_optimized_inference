"""Multi-outcome SCOPE / REACH for a cocoa cohort, and the optional full-vocabulary feature pass.

One M1 pass per patient x sample tracks every outcome inline (generation runs to discharge, not to the
first tracked event). Per outcome:
  M0     the outcome appeared in the M1 trajectory
  SCOPE  inline sum of P(outcome) up to its first occurrence
  REACH  M1 trajectories without the outcome are already outcome-free paths: their inline
         1 - prod(1 - p) is used directly. Trajectories with it are cut before it and the rest is
         regenerated with it suppressed (generate_m2_from_m1_trajectory), then scored.
Patients whose prompt already holds an outcome (<EVENT>_past) get no estimate for it (NaN).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl

from quick_sco_re import (
    GenerationConfig,
    PatientResults,
    generate_m2_from_m1_trajectory,
    generate_trajectories,
    save_scores,
    save_trajectories,
    score_trajectory,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OutcomeRun:
    names: list[str]
    results: list[list[PatientResults]]          # [outcome][patient]
    m1_trajectories: list
    m2_regenerated: list[int]                    # per outcome
    m2_new_tokens: list[int]                     # per outcome
    seconds: float


async def run_outcomes(
    engine,
    config: GenerationConfig,
    prompts: list[list[int]],
    past_masks: list[np.ndarray],
    first_token_biases: list[dict | None] | None = None,
    methods: tuple[str, ...] = ("M1", "M2"),
    chunk_size: int = 64,
) -> OutcomeRun:
    """M0 / SCOPE / REACH samples for every tracked outcome. ``past_masks[k][i]`` marks patient i as
    already having outcome k: no samples are kept for it."""
    names = list(config.tracked_names)
    ids = list(config.tracked_ids)
    n = len(prompts)
    outcome_cfgs = [dataclasses.replace(config, target_event_id=t) for t in ids]
    results = [[PatientResults() for _ in range(n)] for _ in ids]
    m1_all, regen_n, regen_tok = [], [0] * len(ids), [0] * len(ids)
    t0 = time.time()
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        ftb = None if first_token_biases is None else first_token_biases[lo:hi]
        m1 = await generate_trajectories(engine, config, prompts[lo:hi], ["M1"], stop_at_tracked_events=False,
                                         first_token_biases=ftb)
        regen = []
        for tr in m1:
            i = lo + tr.patient_idx
            tr.patient_idx = i                                   # global index from here on
            for k, tid in enumerate(ids):
                if past_masks[k][i]:
                    continue
                kk = tr.inline_tracked_ids.index(tid)
                occurred = bool(tr.occurred_flag[kk]) if tr.occurred_flag is not None else tid in tr.output_ids
                results[k][i].m0_samples.append(occurred)
                results[k][i].m1_samples.append(float(tr.scope_estimates[kk]))
                if "M2" not in methods:
                    continue
                if occurred:
                    regen.append((tr, i, k))
                else:
                    results[k][i].m2_samples.append(float(tr.reach_estimates[kk]))
        m1_all.extend(m1)
        if regen:
            # all regenerations first, then all scoring: SGLang's scheduler fails on batches that mix
            # requests with and without logprobs
            m2 = await asyncio.gather(*[generate_m2_from_m1_trajectory(engine, outcome_cfgs[k], tr, prompts[i])
                                        for tr, i, k in regen])
            scored = await asyncio.gather(*[score_trajectory(engine, outcome_cfgs[k], t2, prompts[i])
                                            for (tr, i, k), t2 in zip(regen, m2)])
            for (tr, i, k), st in zip(regen, scored):
                results[k][i].m2_samples.append(st.score)
                regen_n[k] += 1
                regen_tok[k] += st.trajectory.n_new_tokens or 0
        logger.info(f"patients {hi}/{n}: {len(regen)} REACH regenerations in this chunk")
    return OutcomeRun(names, results, m1_all, regen_n, regen_tok, time.time() - t0)


def estimates(run: OutcomeRun, k: int) -> dict[str, np.ndarray]:
    """Per-patient means of the samples (NaN where none were kept)."""
    m = lambda xs: float(np.mean(xs)) if xs else np.nan
    r = run.results[k]
    return {"M0": np.array([m(p.m0_samples) for p in r]), "M1": np.array([m(p.m1_samples) for p in r]),
            "M2": np.array([m(p.m2_samples) for p in r])}


def summarize(run: OutcomeRun, cohort) -> dict:
    """Per outcome: mean estimates, prevalence among eligible patients, and AUCs against the winnower's
    <EVENT>_future flags (sklearn needed for AUCs)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None
    out = {}
    for k, name in enumerate(run.names):
        est = estimates(run, k)
        y = cohort.future(name)
        elig = ~cohort.past_mask(name)
        s = {"eligible": int(elig.sum()), "m2_regenerated": run.m2_regenerated[k],
             "m2_new_tokens_mean": run.m2_new_tokens[k] / max(run.m2_regenerated[k], 1)}
        for e, v in est.items():
            s[f"mean_{e}"] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        if y is not None:
            s["prevalence"] = float(y[elig].mean()) if elig.any() else None
            for e, v in est.items():
                ok = elig & np.isfinite(v)
                s[f"auc_{e}"] = (float(roc_auc_score(y[ok], v[ok]))
                                 if roc_auc_score and ok.any() and len(np.unique(y[ok])) > 1 else None)
        out[name] = s
    return out


def save_outputs(out_dir: str | pathlib.Path, run: OutcomeRun, cohort, config: GenerationConfig, cfg: dict,
                 summary: dict, save_trajs: bool = True) -> pathlib.Path:
    """patient_index.parquet (ids, prompt lengths, labels), scores_<outcome>.npz per outcome,
    trajectories/ (M1), run_summary.json. These hold patient-level data: keep them local."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"patient_idx": list(range(len(cohort.prompts))), "subject_id": cohort.subject_ids,
                  "prompt_length": [len(p) for p in cohort.prompts]}).join(
        cohort.labels, on="subject_id", how="left").write_parquet(out / "patient_index.parquet")
    m1_tok = sum(len(t.output_ids) for t in run.m1_trajectories)
    for k, name in enumerate(run.names):
        save_scores(run.results[k], out / f"scores_{name.replace('/', '_')}.npz",
                    avg_m1_tokens=m1_tok / max(len(run.m1_trajectories), 1),
                    avg_m2_tokens=run.m2_new_tokens[k] / max(run.m2_regenerated[k], 1))
    if save_trajs:
        save_trajectories(run.m1_trajectories, out / "trajectories", config=config)
    with open(out / "run_summary.json", "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "n_patients": len(cohort.prompts),
                   "n_samp": config.n_samp, "context_window": config.context_window, "window_keep": config.window_keep,
                   "silence_restriction": (cfg["generation"].get("silence_restriction") or {}).get("mode")
                   if (cfg["generation"].get("silence_restriction") or {}).get("enabled") else None,
                   "prompt_prefix_tokens": cfg["generation"].get("prompt_prefix_tokens") or [],
                   "m1_generated_tokens": m1_tok, "seconds": run.seconds, "dropped": cohort.n_dropped,
                   "outcomes": summary}, f, indent=2)
    return out


def full_vocab_features(hf_model, run: OutcomeRun, cohort, config: GenerationConfig,
                        first_token_biases=None, token_budget: int = 60000, autocast_dtype=None) -> dict:
    """Per-patient full-vocabulary features of the M1 trajectories (expected counts, SCOPE, truncated
    REACH for every token; mean generated hidden state), via quick_sco_re.score_full_vocab."""
    from quick_sco_re import score_full_vocab

    trs = run.m1_trajectories
    return score_full_vocab(
        hf_model, cohort.prompts, [t.output_ids for t in trs],
        prompt_index=[t.patient_idx for t in trs], group_ids=[t.patient_idx for t in trs], n_groups=len(cohort.prompts),
        context_window=config.context_window, window_keep=config.window_keep,
        n_conditioned=[t.n_conditioned_steps for t in trs], first_token_biases=first_token_biases,
        suppressed_ids=config.suppressed_ids, return_hidden=True, token_budget=token_budget, autocast_dtype=autocast_dtype)
