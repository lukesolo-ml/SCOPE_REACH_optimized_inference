"""Multi-outcome SCOPE / REACH end to end on the synthetic cohort, with the scripted engine."""

import asyncio

import numpy as np
import pytest

from cocoa_example.cocoa_data import CocoaVocab, build_generation_config, first_token_biases, load_cohort
from cocoa_example.pipeline import estimates, full_vocab_features, run_outcomes, save_outputs, summarize
from quick_sco_re import load_scores
from quick_sco_re.generation import _SGLANG_OVERHEAD

from .conftest import LOOKUP, P_TRACKED, WINDOW, ScriptedEngine


@pytest.fixture
def setup(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    c = load_cohort(cfg, v)
    g = build_generation_config(cfg, v)
    b = first_token_biases(cfg, v, c)
    past = [c.past_mask(n) for n in g.tracked_names]
    eng = ScriptedEngine(seed=1)
    run = asyncio.run(run_outcomes(eng, g, c.prompts, past, b, chunk_size=3))
    return v, c, g, b, past, eng, run


def test_samples_per_outcome_and_exclusion(setup):
    v, c, g, b, past, eng, run = setup
    for k, name in enumerate(g.tracked_names):
        for i, r in enumerate(run.results[k]):
            n = 0 if past[k][i] else g.n_samp
            assert len(r.m0_samples) == len(r.m1_samples) == len(r.m2_samples) == n
    est = estimates(run, 0)
    assert np.isnan(est["M2"][1]) and np.isfinite(est["M2"][0])            # s2 already had ICU


def test_reach_values(setup):
    """No outcome: inline REACH over the whole M1 trajectory. Outcome: regenerated without it."""
    v, c, g, b, past, eng, run = setup
    regen = 0
    for tr in run.m1_trajectories:
        i = tr.patient_idx
        for k, tid in enumerate(g.tracked_ids):
            if past[k][i]:
                continue
            if tid not in tr.output_ids:
                n = len(tr.output_ids) - tr.n_conditioned_steps
                assert (1 - (1 - P_TRACKED) ** n) in [pytest.approx(x) for x in run.results[k][i].m2_samples]
            else:
                regen += 1
    assert regen == sum(run.m2_regenerated) > 0
    # every REACH regeneration forbade its outcome
    m2_calls = [cl for cl in eng.calls if "logprob_start_len" not in cl and
                any(cl["sampling_params"]["logit_bias"].get(t, 0) <= -10000 for t in g.tracked_ids if t != LOOKUP["BOS"])
                and cl["sampling_params"]["max_new_tokens"] > 1]
    assert m2_calls
    for k in range(len(g.tracked_ids)):
        vals = [x for r in run.results[k] for x in r.m2_samples]
        assert all(0 <= x <= 1 for x in vals)


def test_window_and_silence_restriction(setup):
    v, c, g, b, past, eng, run = setup
    gen = [cl for cl in eng.calls if "logprob_start_len" not in cl]
    assert all(len(cl["input_ids"]) + cl["sampling_params"]["max_new_tokens"] <= WINDOW - _SGLANG_OVERHEAD for cl in gen)
    for tr in run.m1_trajectories:
        if b[tr.patient_idx] is None:
            assert tr.n_conditioned_steps == 0
        else:
            assert tr.n_conditioned_steps == 1
            assert b[tr.patient_idx].get(tr.output_ids[0], 0) > -10000         # first token allowed by the restriction
            lo_hi = {LOOKUP[n]: n for n in LOOKUP if n.startswith("TIME//")}
            assert tr.output_ids[0] in lo_hi


def test_summary_and_saved_outputs(setup, cfg, tmp_path):
    v, c, g, b, past, eng, run = setup
    s = summarize(run, c)
    assert set(s) == set(g.tracked_names)
    assert s["XFR-IN//icu"]["eligible"] == 3 and "auc_M2" in s["XFR-IN//icu"]
    out = save_outputs(tmp_path / "o", run, c, g, cfg, s)
    for name in g.tracked_names:
        sc = load_scores(out / f"scores_{name.replace('/', '_')}.npz")
        assert len(sc["M2"]) == len(c.prompts)
    assert (out / "patient_index.parquet").exists() and (out / "run_summary.json").exists()
    assert (out / "trajectories").exists()


def test_full_vocab_features(setup):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    v, c, g, b, past, eng, run = setup
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=v.size, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512)).eval()
    f = full_vocab_features(m, run, c, g, b, token_budget=400)
    assert f["psum"].shape == (len(c.prompts), v.size)
    assert f["n_trajectories"].tolist() == [g.n_samp] * len(c.prompts)
    assert (f["psum"][:, LOOKUP["BOS"]] == 0).all()
    assert f["gen_hidden_mean"].shape == (len(c.prompts), 32)
    assert np.all((f["reach_trunc"] >= 0) & (f["reach_trunc"] <= 1))
