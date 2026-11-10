"""Against the real cocoa pipeline (skipped unless cocoa-tokenizer is installed).

Random-gap synthetic timelines (made up here, never real data) are collated, tokenized with time
spacers and winnowed by cocoa itself, then read with cocoa_example. The silence restriction must
allow every patient's true next token, for both of cocoa's time-based cuts.
"""

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
import yaml

cocoa_collator = pytest.importorskip("cocoa.collator", reason="needs cocoa-tokenizer")
from cocoa.tokenizer import Tokenizer  # noqa: E402
from cocoa.winnower import Winnower  # noqa: E402

from cocoa_example.cocoa_data import (CocoaVocab, build_generation_config, first_token_biases,  # noqa: E402
                                      load_cohort, time_token_bounds)
from quick_sco_re import allowed_mask  # noqa: E402

DTTM = pl.Datetime("us", "UTC")
CUT_S = 86400


def raw_tables(root, n=120, seed=0):
    """Hospitalizations with vitals at log-uniform gaps (1 min to ~300 days); the first event is
    0-6 h after admission, so a cut from the first event and one from admission differ."""
    rng = np.random.default_rng(seed)
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    hosp, vit = [], []
    for i in range(n):
        adm = base + dt.timedelta(days=int(rng.integers(0, 300)), minutes=int(rng.integers(0, 1440)))
        t = adm + dt.timedelta(minutes=float(rng.uniform(0, 360)))
        times = [t]
        while t - times[0] < dt.timedelta(hours=30) or len(times) < 4:
            t += dt.timedelta(minutes=float(np.exp(rng.uniform(0, math.log(60 * 24 * 300)))))
            times.append(t)
        hosp.append({"hospitalization_id": f"H{i}", "patient_id": f"P{i}", "admission_dttm": adm,
                     "discharge_dttm": times[-1] + dt.timedelta(minutes=1), "age_at_admission": 60.0,
                     "admission_type_category": "Inpatient", "discharge_category": ["Home", "Expired"][i % 2]})
        vit += [{"hospitalization_id": f"H{i}", "recorded_dttm": x, "vital_category": "heart_rate",
                 "vital_value": float(rng.normal(80, 10))} for x in times]
    pl.DataFrame(hosp, schema={"hospitalization_id": pl.String, "patient_id": pl.String, "admission_dttm": DTTM,
                               "discharge_dttm": DTTM, "age_at_admission": pl.Float64,
                               "admission_type_category": pl.String, "discharge_category": pl.String},
                 orient="row").write_parquet(root / "clif_hospitalization.parquet")
    pl.DataFrame(vit, schema={"hospitalization_id": pl.String, "recorded_dttm": DTTM, "vital_category": pl.String,
                              "vital_value": pl.Float64}, orient="row").write_parquet(root / "clif_vitals.parquet")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("cocoa")
    raw, out = root / "raw", root / "processed"
    raw.mkdir(), out.mkdir()
    raw_tables(raw)
    collation = {
        "subject_id": "hospitalization_id", "default_timezone": "UTC",
        "subject_splits": {"train_frac": 1.0, "tuning_frac": 0.0},
        "reference": {"table": "clif_hospitalization", "start_time": "admission_dttm", "end_time": "discharge_dttm"},
        "entries": [{"table": "clif_vitals", "prefix": "VTL", "code": "vital_category", "numeric_value": "vital_value",
                     "time": "recorded_dttm"},
                    {"table": "REFERENCE", "prefix": "DSCG", "code": "discharge_category", "time": "discharge_dttm"}],
    }
    (root / "collation.yaml").write_text(yaml.safe_dump(collation))
    cocoa_collator.Collator(collation_cfg=root / "collation.yaml", raw_data_home=raw, processed_data_home=out).save_all()
    from importlib import resources

    tok_cfg = yaml.safe_load((resources.files("cocoa.config") / "tokenization.yaml").read_text())
    (root / "tokenization.yaml").write_text(yaml.safe_dump(tok_cfg | {"insert_spacers": True}, sort_keys=False))
    Tokenizer(tokenization_cfg=root / "tokenization.yaml", processed_data_home=out).save_all()
    for name, thr in (("first_event", {"duration_s": CUT_S}), ("start_time", {"after_start_time_s": CUT_S})):
        (root / f"winnow_{name}.yaml").write_text(yaml.safe_dump(
            {"outcome_tokens": ["DSCG//*"], "threshold": thr, "splits": ["train"]}))
        Winnower(winnowing_cfg=root / f"winnow_{name}.yaml", processed_data_home=out,
                 output_home=root / name).save_all()
    return root


def cfg_for(processed, reference):
    d = processed / reference
    return {"cocoa_outputs": {"held_out_for_inference": str(d / "train_for_inference.parquet"),
                              "tokenizer_yaml": str(d / "tokenizer.yaml")},
            "model_path": "unused", "output_dir": "unused",
            "generation": {"max_len": 100000, "n_samp": 2, "tracked_events": ["DSCG//expired"],
                           "suppressed_tokens": ["BOS"],
                           "silence_restriction": {"enabled": True, "horizon_minutes": CUT_S / 60,
                                                   "reference": reference}}}


def test_vocab_reads_cocoas_tokenizer(processed):
    v = CocoaVocab(processed / "first_event" / "tokenizer.yaml")
    assert v.cocoa_version and v.tokenizer_cfg["insert_spacers"] is True
    bins = v.time_bins()
    assert len(bins) >= 8 and math.isinf(max(hi for _, hi in bins.values()))


def test_end_tokens_default_to_discharge_and_eos(processed):
    cfg = cfg_for(processed, "first_event")
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    g = build_generation_config(cfg, v)
    assert {v.id_to_name[i] for i in g.end_token_ids} == {"EOS", "DSCG//home", "DSCG//expired"}


def test_spacer_bins_match_cocoas_labels(processed):
    d = processed / "first_event"
    v = CocoaVocab(d / "tokenizer.yaml")
    bounds = time_token_bounds(cfg_for(processed, "first_event"), v)
    n = 0
    for toks, times in pl.read_parquet(d / "tokens_times.parquet").select("tokens", "times").iter_rows():
        for j, t in enumerate(toks):
            if t in bounds:
                gap = int((times[j] - max(times[:j])).total_seconds() // 60)
                lo, hi = bounds[t]
                assert lo <= gap <= hi, (v.id_to_name[t], gap)
                n += 1
    assert n > 100


@pytest.mark.parametrize("reference", ["first_event", "start_time"])
def test_silence_restriction_allows_the_true_next_token(processed, reference):
    cfg = cfg_for(processed, reference)
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    c = load_cohort(cfg, v)
    biases = first_token_biases(cfg, v, c)
    df = pl.read_parquet(cfg["cocoa_outputs"]["held_out_for_inference"])
    nxt = dict(zip(df["subject_id"].cast(pl.String), [f[0] for f in df["tokens_future"].to_list()]))
    assert len(c.prompts) > 50 and sum(b is not None for b in biases) > 0.9 * len(biases)
    assert np.all((c.silence_minutes >= 0) & (c.silence_minutes <= CUT_S / 60))
    for sid, b in zip(c.subject_ids, biases):
        assert allowed_mask(b, v.size)[nxt[sid]], (sid, v.id_to_name[nxt[sid]])


def test_restriction_refuses_a_vocab_without_spacers(processed, tmp_path):
    cfg = cfg_for(processed, "first_event")
    data = yaml.safe_load(open(cfg["cocoa_outputs"]["tokenizer_yaml"]))
    data["lookup"] = {k: i for i, k in enumerate(k for k in data["lookup"] if not k.startswith("TIME//"))}
    data["cfg"]["insert_spacers"] = False
    (tmp_path / "tokenizer.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    v = CocoaVocab(tmp_path / "tokenizer.yaml")
    c = load_cohort(cfg, v)
    with pytest.raises(ValueError, match="needs TIME"):
        first_token_biases(cfg, v, c)


def test_out_of_order_spacers_are_refused(processed, tmp_path):
    """cocoa bins gaps by the spacers' order; a reordered map (e.g. a yaml dumped with sorted keys)
    mislabels every gap, so the vocabulary's bins cannot be trusted."""
    data = yaml.safe_load(open(processed / "first_event" / "tokenizer.yaml"))
    data["cfg"]["spacers"] = dict(sorted(data["cfg"]["spacers"].items()))
    (tmp_path / "tokenizer.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="increase in the order"):
        CocoaVocab(tmp_path / "tokenizer.yaml").time_bins()
