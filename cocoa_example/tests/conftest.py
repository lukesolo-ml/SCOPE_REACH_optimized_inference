"""Synthetic cocoa outputs (made up in a temp folder, never real data) and a scripted fake SGLang engine."""

import math
import random

import polars as pl
import pytest
import yaml

from cocoa_example.cocoa_data import COCOA_TIME_BINS

NAMES = (["BOS", "EOS", "ADMN//ed", "LAB-RES//a_Q1", "LAB-RES//a_Q9", "VTL//hr_Q5", "XFR-IN//icu", "RESP//imv",
          "DSCG//home", "DSCG//expired"] + list(COCOA_TIME_BINS))
LOOKUP = {n: i for i, n in enumerate(NAMES)}
P_TRACKED = 0.05
WINDOW, KEEP = 64, 40


def tok(*names):
    return [LOOKUP[n] for n in names]


def row(sid, past, secs, icu=(False, False), imv=(False, False), dead=(False, False)):
    return {"subject_id": sid, "tokens_past": past, "s_elapsed_past": secs,
            "XFR-IN//icu_past": icu[0], "XFR-IN//icu_future": icu[1],
            "RESP//imv_past": imv[0], "RESP//imv_future": imv[1],
            "DSCG//expired_past": dead[0], "DSCG//expired_future": dead[1]}


@pytest.fixture
def cocoa_dir(tmp_path):
    rows = [
        row("s1", tok("BOS", "ADMN//ed", "LAB-RES//a_Q1"), [0, 0, 3600], icu=(False, True)),            # silent 23 h
        row("s2", tok("BOS", "ADMN//ed", "XFR-IN//icu", "VTL//hr_Q5"), [0, 0, 600, 86000], icu=(True, False), imv=(False, True)),
        row("s3", tok("BOS", "ADMN//ed", "DSCG//home"), [0, 0, 7200]),                                   # stay ended: dropped
        row("s4", tok("BOS", "ADMN//ed", "LAB-RES//a_Q9"), [0, 0, 80000], dead=(True, False)),         # excluded by flag
        row("s5", tok("BOS", "ADMN//ed") + tok("LAB-RES//a_Q1") * 80, [0, 0] + [60] * 80, dead=(False, True)),  # long prompt
        row("s6", tok("BOS", "ADMN//ed", "VTL//hr_Q5"), [0, 0, 86390], imv=(False, False)),             # silent < 5 min
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / "held_out_for_inference.parquet")
    with open(tmp_path / "tokenizer.yaml", "w") as f:
        yaml.safe_dump({"lookup": LOOKUP}, f)
    return tmp_path


@pytest.fixture
def cfg(cocoa_dir):
    return {
        "cocoa_outputs": {"held_out_for_inference": str(cocoa_dir / "held_out_for_inference.parquet"),
                          "tokenizer_yaml": str(cocoa_dir / "tokenizer.yaml")},
        "model_path": "unused", "output_dir": str(cocoa_dir / "out"),
        "cohort": {"exclude_flag": "DSCG//expired_past"},
        "prompt_overflow": "truncate_left",
        "generation": {
            "max_len": 200, "n_samp": 4, "tracked_events": ["XFR-IN//icu", "RESP//imv", "DSCG//expired"],
            "end_tokens": {"prefixes": ["DSCG"]}, "suppressed_tokens": ["BOS"], "prompt_prefix_tokens": [],
            "rolling_window": {"enabled": True, "context_window": WINDOW, "keep": KEEP},
            "silence_restriction": {"enabled": True, "mode": "hard", "horizon_minutes": 1440},
        },
    }


class ScriptedEngine:
    """Generation: a random walk over lab/vital/time tokens that sometimes emits ICU, IMV or death and ends
    at discharge (or the token budget). It honours logit_bias (-10000 forbids; a restricted first step
    emits the smallest allowed time token), stop_token_ids and max_new_tokens. Tracked logprobs and
    scoring passes report P = P_TRACKED for every requested token."""

    FILL = tok("LAB-RES//a_Q1", "LAB-RES//a_Q9", "VTL//hr_Q5", "TIME//15m-1h")

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = []

    async def async_generate(self, input_ids, sampling_params, **kw):
        self.calls.append(dict(input_ids=list(input_ids), sampling_params=dict(sampling_params), **kw))
        if "logprob_start_len" in kw:
            m = len(input_ids) - kw["logprob_start_len"]
            e = [(math.log(P_TRACKED), t, None) for t in kw["token_ids_logprob"]]
            return {"meta_info": {"input_token_ids_logprobs": [e] * m}}
        bias = sampling_params.get("logit_bias") or {}
        ok = lambda t: bias.get(t, 0) > -10000
        stops = set(sampling_params.get("stop_token_ids") or [])
        out, finish = [], {"type": "length"}
        for _ in range(sampling_params["max_new_tokens"]):
            if not ok(LOOKUP["LAB-RES//a_Q1"]):                     # restricted step: a time token only
                t = min((t for t in tok(*COCOA_TIME_BINS) if ok(t)), key=lambda t: t)
            else:
                r = self.rng.random()
                if r < 0.03:
                    t = LOOKUP["XFR-IN//icu"]
                elif r < 0.05:
                    t = LOOKUP["RESP//imv"]
                elif r < 0.06:
                    t = LOOKUP["DSCG//expired"]
                elif r < 0.10:
                    t = LOOKUP["DSCG//home"]
                else:
                    t = self.rng.choice(self.FILL)
                if not ok(t):
                    t = self.FILL[0]
            out.append(t)
            if t in stops:
                finish = {"type": "stop", "matched": t}
                break
        meta = {"output_ids": out, "finish_reason": finish}
        if kw.get("token_ids_logprob"):
            meta["output_token_ids_logprobs"] = [[(math.log(P_TRACKED), t, None) for t in kw["token_ids_logprob"]]] * len(out)
        return {"meta_info": meta}
