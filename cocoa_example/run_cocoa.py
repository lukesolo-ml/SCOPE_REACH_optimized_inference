"""Run multi-outcome SCOPE / REACH (and optionally full-vocabulary features) on a cocoa cohort.

    python -m cocoa_example.run_cocoa --config cocoa_example/config_example.yaml
    python -m cocoa_example.run_cocoa --config my.yaml --shard-idx 0 --n-shards 8     # patients i % 8 == 0
    python -m cocoa_example.run_cocoa --config my.yaml --dry-run                      # cohort + settings, no GPU

Writes to output_dir (or output_dir/shard_NNN): patient_index.parquet, scores_<outcome>.npz,
trajectories/, run_summary.json, and features.npz with features.enabled. All of these are
patient-level: keep them off any shared or hosted location.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib

import numpy as np

from cocoa_example.cocoa_data import (
    CocoaVocab,
    build_generation_config,
    first_token_biases,
    load_cohort,
    load_config,
)
from cocoa_example.pipeline import full_vocab_features, run_outcomes, save_outputs, summarize

logger = logging.getLogger("cocoa_example")


def shard(cohort, idx: int, n: int):
    keep = list(range(idx, len(cohort.prompts), n))
    cohort.prompts = [cohort.prompts[i] for i in keep]
    cohort.subject_ids = [cohort.subject_ids[i] for i in keep]
    cohort.labels = cohort.labels[keep]
    if cohort.silence_minutes is not None:
        cohort.silence_minutes = cohort.silence_minutes[keep]
    return cohort


async def run(cfg: dict, shard_idx: int | None, n_shards: int | None, dry: bool) -> None:
    from quick_sco_re import check_rope_config, create_engine

    vocab = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    logger.info(f"vocabulary: {vocab.size} tokens (cocoa-tokenizer {vocab.cocoa_version or 'version not recorded'}), "
                f"{len(vocab.time_bins())} TIME// bins")
    cohort = load_cohort(cfg, vocab)
    if shard_idx is not None:
        cohort = shard(cohort, shard_idx, n_shards)
    config = build_generation_config(cfg, vocab)
    biases = first_token_biases(cfg, vocab, cohort)
    past = [cohort.past_mask(n) for n in config.tracked_names]
    logger.info(f"{len(cohort.prompts)} patients x {config.n_samp} samples; outcomes {config.tracked_names}; "
                f"window {config.context_window}/{config.window_keep}; "
                f"silence restriction {'on' if biases is not None else 'off'} "
                f"({0 if biases is None else sum(b is not None for b in biases)} patients restricted)")
    if dry or not cohort.prompts:
        return
    out = pathlib.Path(cfg["output_dir"]).expanduser()
    out = out / f"shard_{shard_idx:03d}" if shard_idx is not None else out
    model = str(pathlib.Path(cfg["model_path"]).expanduser().resolve())
    check_rope_config(model)
    eng_cfg = cfg.get("engine") or {}
    feats_cfg = cfg.get("features") or {}
    engine = create_engine(model_path=model, max_len=config.max_len, mem_fraction=eng_cfg.get("mem_fraction", 0.5),
                           context_length=config.context_window, attention_backend=eng_cfg.get("attention_backend", "triton"))
    hf = None
    try:
        if (cfg.get("checks") or {}).get("parity", True):
            import torch
            from transformers import AutoModelForCausalLM

            from quick_sco_re import check_engine_parity

            hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).cuda().eval()
            d = await check_engine_parity(engine, hf, cohort.prompts[:16], context_window=config.context_window)
            logger.info(f"SGLang vs HF next-token distributions: max L1 {d:.4f}")
            hf.cpu()
        result = await run_outcomes(engine, config, cohort.prompts, past, biases,
                                    methods=tuple(cfg["generation"].get("methods", ["M1", "M2"])),
                                    chunk_size=int(eng_cfg.get("patient_chunk_size", 64)))
    finally:
        engine.shutdown()
    summary = summarize(result, cohort)
    save_outputs(out, result, cohort, config, cfg, summary, save_trajs=(cfg.get("save") or {}).get("trajectories", True))
    logger.info(json.dumps(summary, indent=1))
    if feats_cfg.get("enabled"):
        import torch
        from transformers import AutoModelForCausalLM

        hf = hf or AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        hf = hf.cuda().eval()
        f = full_vocab_features(hf, result, cohort, config, biases, token_budget=int(feats_cfg.get("token_budget", 60000)),
                                autocast_dtype=torch.bfloat16)
        np.savez(out / "features.npz", subject_id=np.array(cohort.subject_ids), **{k: v.astype(np.float32) for k, v in f.items()})
        logger.info(f"wrote {out / 'features.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard-idx", type=int)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if (a.shard_idx is None) != (a.n_shards is None):
        ap.error("--shard-idx and --n-shards go together")
    # own handler: SGLang reconfigures the root logger when its engine starts, which would silence ours
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    for name in ("cocoa_example", "quick_sco_re"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
        lg.propagate = False
    asyncio.run(run(load_config(a.config), a.shard_idx, a.n_shards, a.dry_run))


if __name__ == "__main__":
    main()
