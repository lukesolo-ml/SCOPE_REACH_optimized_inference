"""CLI pieces that need no GPU: config loading, dry run, sharding."""

import asyncio

import pytest
import yaml

from cocoa_example.cocoa_data import CocoaVocab, load_cohort, load_config
from cocoa_example.run_cocoa import run, shard


def test_load_config_requires_sections(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    assert load_config(p)["generation"]["n_samp"] == 4
    del cfg["model_path"]
    yaml.safe_dump(cfg, open(p, "w"))
    with pytest.raises(ValueError, match="model_path"):
        load_config(p)


def test_dry_run_needs_no_engine(cfg, caplog):
    caplog.set_level("INFO")
    asyncio.run(run(cfg, None, None, dry=True))
    assert "4 patients x 4 samples" in caplog.text and "3 patients restricted" in caplog.text


def test_shard_keeps_rows_aligned(cfg):
    c = load_cohort(cfg, CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"]))
    sil = c.silence_minutes.copy()
    s = shard(c, 1, 2)
    assert s.subject_ids == ["s2", "s6"] and s.labels["subject_id"].to_list() == ["s2", "s6"]
    assert list(s.silence_minutes) == [sil[1], sil[3]] and len(s.prompts) == 2
