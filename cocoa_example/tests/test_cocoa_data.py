"""cocoa adapter: cohort, labels, silence, prompts, generation config."""

import numpy as np
import pytest

from cocoa_example.cocoa_data import CocoaVocab, build_generation_config, first_token_biases, load_cohort

from .conftest import KEEP, LOOKUP, WINDOW, tok


def test_vocab(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    assert v("XFR-IN//icu") == LOOKUP["XFR-IN//icu"] and "BOS" in v and v.size == len(LOOKUP)
    assert v.ids_with_prefix("DSCG") == {LOOKUP["DSCG//home"], LOOKUP["DSCG//expired"]}
    with pytest.raises(KeyError):
        v("NOPE//x")


def test_cohort_drops_and_labels(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    c = load_cohort(cfg, v)
    assert c.subject_ids == ["s1", "s2", "s5", "s6"]                       # s3 ended in prompt, s4 excluded by flag
    assert c.n_dropped["ended_in_prompt"] == 1 and c.n_dropped["exclude_flag"] == 1
    assert c.past_mask("XFR-IN//icu").tolist() == [False, True, False, False]
    assert c.future("RESP//imv").tolist() == [False, True, False, False]
    assert c.future("NOT//there") is None and not c.past_mask("NOT//there").any()


def test_silence_minutes(cfg):
    c = load_cohort(cfg, CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"]))
    np.testing.assert_allclose(c.silence_minutes, [1440 - 60, 1440 - 86000 / 60, 1440 - 1, 1440 - 86390 / 60])


def test_events_after_horizon_raise(cfg):
    cfg["generation"]["silence_restriction"]["horizon_minutes"] = 600
    with pytest.raises(ValueError, match="horizon"):
        load_cohort(cfg, CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"]))


def test_prefix_and_overflow(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    cfg["generation"]["prompt_prefix_tokens"] = ["EOS"]
    cfg["generation"]["max_len"] = 50
    c = load_cohort(cfg, v)
    assert c.prompts[0][:2] == tok("EOS", "BOS")
    assert len(c.prompts[2]) == 50 and c.n_truncated == 1                 # s5 left-truncated
    cfg["prompt_overflow"] = "drop"
    c = load_cohort(cfg, v)
    assert c.subject_ids == ["s1", "s2", "s6"] and c.n_dropped["too_long"] == 1
    assert len(c.silence_minutes) == 3


def test_generation_config(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    g = build_generation_config(cfg, v)
    assert g.tracked_ids == tok("XFR-IN//icu", "RESP//imv", "DSCG//expired")
    assert g.end_token_ids == set(tok("DSCG//home", "DSCG//expired"))
    assert g.suppressed_ids == tok("BOS") and (g.context_window, g.window_keep) == (WINDOW, KEEP)


def test_silence_biases(cfg):
    v = CocoaVocab(cfg["cocoa_outputs"]["tokenizer_yaml"])
    c = load_cohort(cfg, v)
    b = first_token_biases(cfg, v, c)
    assert b[3] is None                                                    # s6: silent < 5 min
    allowed = [t for t in range(v.size) if b[0].get(t, 0) > -10000]
    assert allowed == tok("TIME//12h-1d", "TIME//1d-3d", "TIME//3d-1w", "TIME//1w-2w", "TIME//2w-1mt",
                          "TIME//1mt-3mt", "TIME//3mt-6mt", "TIME//6mt+")  # 1380 min silent: bins with hi >= 1380
    cfg["generation"]["silence_restriction"]["enabled"] = False
    assert first_token_biases(cfg, v, c) is None
