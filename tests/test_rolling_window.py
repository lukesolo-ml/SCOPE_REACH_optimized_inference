"""Rolling context window, first-token restriction and windowed scoring, against a fake engine."""

import asyncio
import math

import numpy as np
import pytest

from quick_sco_re.generation import _SGLANG_OVERHEAD, generate_trajectory
from quick_sco_re.scoring import score_trajectory
from quick_sco_re.structures import GeneratedTrajectory, GenerationConfig, TrajectoryType

TRACKED = [10, 20]
STOP = 11
TIME_TOKENS = [50, 51]
P_TRACKED = 0.01


class FakeEngine:
    """Emits FILL each step (or the first allowed time token on a restricted step),
    and STOP once `stop_after` tokens have been generated in total."""

    FILL = 7

    def __init__(self, stop_after=None):
        self.calls = []
        self.stop_after = stop_after
        self.total = 0

    async def async_generate(self, input_ids, sampling_params, **kw):
        self.calls.append(dict(input_ids=list(input_ids), sampling_params=sampling_params, **kw))
        n = sampling_params["max_new_tokens"]
        if "logprob_start_len" in kw:  # scoring pass
            m = len(input_ids) - kw["logprob_start_len"]
            entry = [(math.log(P_TRACKED), t, None) for t in kw["token_ids_logprob"]]
            return {"meta_info": {"input_token_ids_logprobs": [entry] * m}}
        bias = sampling_params.get("logit_bias", {})
        out, finish = [], {"type": "length"}
        for _ in range(n):
            if self.stop_after is not None and self.total >= self.stop_after:
                out.append(STOP)
                finish = {"type": "stop", "matched": STOP}
                break
            tok = self.FILL
            if bias.get(self.FILL, 0) <= -10000:
                tok = next(t for t in TIME_TOKENS if bias.get(t, 0) > -10000)
            out.append(tok)
            self.total += 1
        lps = None
        if kw.get("token_ids_logprob"):
            lps = [[(math.log(P_TRACKED), t, None) for t in kw["token_ids_logprob"]]] * len(out)
        meta = {"output_ids": out, "finish_reason": finish}
        if lps is not None:
            meta["output_token_ids_logprobs"] = lps
        return {"meta_info": meta}


def cfg(**kw):
    base = dict(max_len=300, n_samp=1, target_event_id=TRACKED[0], end_token_ids={STOP},
                tracked_ids=TRACKED, tracked_names=["a", "b"], context_window=64, window_keep=48)
    base.update(kw)
    return GenerationConfig(**base)


def test_window_caps_context_and_total_length():
    eng = FakeEngine()
    prompt = list(range(100, 140))
    t = asyncio.run(generate_trajectory(eng, cfg(), prompt, 0, 0, TrajectoryType.M1, stop_at_tracked_events=False))
    assert all(len(c["input_ids"]) + c["sampling_params"]["max_new_tokens"] <= 64 - _SGLANG_OVERHEAD
               for c in eng.calls)
    assert len(eng.calls) > 1
    assert len(t.output_ids) == 300 - len(prompt) - 1 - _SGLANG_OVERHEAD
    # every step contributes to the inline estimates
    assert t.scope_estimates[0] == pytest.approx(P_TRACKED * len(t.output_ids))
    assert t.n_conditioned_steps == 0


def test_long_prompt_is_cut_to_window():
    eng = FakeEngine()
    prompt = list(range(1000, 1200))
    asyncio.run(generate_trajectory(eng, cfg(), prompt, 0, 0, TrajectoryType.M1))
    assert eng.calls[0]["input_ids"] == prompt[-(64 - _SGLANG_OVERHEAD - 2):]
    assert eng.calls[0]["sampling_params"]["max_new_tokens"] == 1
    assert len(eng.calls[1]["input_ids"]) == 48


def test_stop_token_ends_generation_across_segments():
    eng = FakeEngine(stop_after=70)
    t = asyncio.run(generate_trajectory(eng, cfg(), list(range(100, 110)), 0, 0, TrajectoryType.M1,
                                  stop_at_tracked_events=False))
    assert t.output_ids[-1] == STOP and len(t.output_ids) == 71
    assert t.timeline_terminating_id == STOP


def test_first_token_restriction():
    eng = FakeEngine()
    bias = {tid: -10000.0 for tid in [FakeEngine.FILL, STOP] + TRACKED}
    bias[50] = -10000.0  # 50 too short for the silence, 51 allowed
    t = asyncio.run(generate_trajectory(eng, cfg(), list(range(100, 110)), 0, 0, TrajectoryType.M1,
                                  stop_at_tracked_events=False, first_token_bias=bias))
    first = eng.calls[0]
    assert first["sampling_params"]["max_new_tokens"] == 1
    assert t.output_ids[0] == 51 and t.n_conditioned_steps == 1
    # the restricted step adds nothing to the inline estimates
    assert t.scope_estimates[0] == pytest.approx(P_TRACKED * (len(t.output_ids) - 1))
    # later requests carry no restriction
    assert all(c["sampling_params"]["logit_bias"].get(FakeEngine.FILL, 0) == 0 for c in eng.calls[1:])


def test_first_token_bias_must_forbid_tracked():
    with pytest.raises(ValueError):
        asyncio.run(generate_trajectory(FakeEngine(), cfg(), [1, 2, 3], 0, 0, TrajectoryType.M1,
                                  first_token_bias={FakeEngine.FILL: -10000.0}))


def test_windowed_scoring():
    eng = FakeEngine()
    prompt = list(range(100, 200))
    traj = GeneratedTrajectory(patient_idx=0, sample_idx=0, traj_type=TrajectoryType.M2, prompt_len=len(prompt),
                               output_ids=[7] * 150, timeline_terminating_id=None, n_conditioned_steps=1)
    st = asyncio.run(score_trajectory(eng, cfg(), traj, prompt))
    assert all(len(c["input_ids"]) <= 64 - _SGLANG_OVERHEAD - 1 for c in eng.calls)
    assert len(eng.calls) > 1
    assert st.score == pytest.approx(1 - (1 - P_TRACKED) ** 149)


def test_no_window_keeps_single_request():
    eng = FakeEngine()
    t = asyncio.run(generate_trajectory(eng, cfg(context_window=None, window_keep=None), [1, 2, 3], 0, 0,
                                  TrajectoryType.M1, stop_at_tracked_events=False))
    assert len(eng.calls) == 1
    assert len(t.output_ids) == 300 - 3 - 1 - _SGLANG_OVERHEAD
