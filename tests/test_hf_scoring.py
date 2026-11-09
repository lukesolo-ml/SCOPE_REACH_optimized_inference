"""generation_segments against generate_trajectory, and score_full_vocab against a
brute-force step-by-step reference on a tiny random Llama."""

import asyncio

import numpy as np
import pytest

from quick_sco_re.generation import _SGLANG_OVERHEAD, generate_trajectory
from quick_sco_re.hf_scoring import generation_segments, score_full_vocab
from quick_sco_re.restrictions import silence_restriction_bias
from quick_sco_re.structures import GenerationConfig, TrajectoryType

from .test_rolling_window import STOP, TIME_TOKENS, FakeEngine

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def segments_from_calls(calls, prompt_len):
    """Recover (c0, g0, g1) from the generation requests: a request's context is
    prompt + generated[:g] starting at c0, so c0 = prompt_len + g - len(context)."""
    segs, g = [], 0
    for c in calls:
        n_out = c["n_out"]
        c0 = prompt_len + g - len(c["input_ids"])
        if segs and segs[-1][0] == c0:
            segs[-1] = (c0, segs[-1][1], g + n_out)
        else:
            segs.append((c0, g, g + n_out))
        g += n_out
    return [s for s in segs if s[2] > s[1]]


class RecordingEngine(FakeEngine):
    async def async_generate(self, input_ids, sampling_params, **kw):
        out = await super().async_generate(input_ids, sampling_params, **kw)
        self.calls[-1]["n_out"] = len(out["meta_info"]["output_ids"])
        return out


@pytest.mark.parametrize("prompt_len", [3, 40, 54, 55, 200])
@pytest.mark.parametrize("window,keep", [(64, 48), (64, 20), (40, 30)])
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("stop_after", [None, 5, 77])
def test_segments_match_generation(prompt_len, window, keep, restricted, stop_after):
    eng = RecordingEngine(stop_after=stop_after)
    cfg = GenerationConfig(max_len=320, n_samp=1, target_event_id=10, end_token_ids={STOP},
                           context_window=window, window_keep=keep)
    prompt = list(range(100, 100 + prompt_len))
    ftb = {t: -10000.0 for t in range(60) if t not in TIME_TOKENS} if restricted else None
    traj = asyncio.run(generate_trajectory(eng, cfg, prompt, 0, 0, TrajectoryType.M1, first_token_bias=ftb))
    got = generation_segments(len(prompt), len(traj.output_ids), window, keep)
    assert got == segments_from_calls(eng.calls, len(prompt))


def test_no_window_is_one_segment():
    assert generation_segments(10, 5, None) == [(0, 0, 5)]
    assert generation_segments(10, 0, 64) == []


V = 40
BOS = 0
BOUNDS = {30: (5, 15), 31: (15, 60), 32: (60, 120), 33: (120, 360)}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                                   tie_word_embeddings=True)
    m = transformers.LlamaForCausalLM(cfg).eval()
    return m


def brute(model, prompt, out, segs, ncond, bias, suppressed):
    """Each generated token's distribution from a separate forward pass over exactly
    its context; the state after reading it from another."""
    full = list(prompt) + list(out)
    P = len(prompt)
    ps, hs = [], []
    for (c0, g0, g1) in segs:
        for t in range(g0, g1):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([full[c0:P + t]]), output_hidden_states=True)
                p = torch.softmax(o.logits[0, -1].float(), -1).numpy().astype(np.float64)
                h = model.model(input_ids=torch.tensor([full[c0:P + t + 1]])).last_hidden_state[0, -1].numpy()
            p[list(suppressed)] = 0
            if t < ncond and bias:
                w = np.ones(V)
                for k, v in bias.items():
                    w[k] = 0 if v <= -5000 else np.exp(v)
                p = p * w
            ps.append(p / p.sum())
            hs.append(h)
    ps = np.array(ps)
    first = np.full(V, len(out))
    for i, tok in reversed(list(enumerate(out))):
        first[tok] = i
    upto = np.arange(len(out))[:, None] <= first[None, :]
    return dict(psum=ps.sum(0), scope=(ps * upto).sum(0),
                ls=(np.log1p(-np.clip(ps, None, 1 - 1e-7)) * upto).sum(0), hmean=np.mean(hs, 0))


def test_full_vocab_matches_brute_force(model):
    rng = np.random.default_rng(0)
    window, keep = 24, 12                          # small, so most trajectories use the window
    prompts = [list(rng.integers(1, V, n)) for n in (5, 9, 30)]
    bias = [None, silence_restriction_bias(100, BOUNDS, range(V), mode="prop"), silence_restriction_bias(20, BOUNDS, range(V))]
    outs, pidx, ncond = [], [], []
    for j, n in [(0, 3), (0, 40), (1, 25), (1, 6), (2, 18), (2, 1)]:
        o = list(rng.integers(1, V, n))
        nc = 1 if bias[j] else 0
        if nc:
            o[0] = 32 if j == 1 else 31            # an allowed time token on the restricted step
        outs.append(o); pidx.append(j); ncond.append(nc)
    groups = [0, 0, 1, 1, 2, 2]
    track = [5, 17, 32]
    res = score_full_vocab(model, prompts, outs, prompt_index=pidx, group_ids=groups, context_window=window,
                           window_keep=keep, n_conditioned=ncond, first_token_biases=bias, suppressed_ids=[BOS],
                           log_survival_ids=track, return_hidden=True, token_budget=200)
    exp = {k: np.zeros((3, V)) for k in ("psum", "scope", "reach_trunc")}
    hm = np.zeros((3, model.config.hidden_size))
    for i, o in enumerate(outs):
        segs = generation_segments(len(prompts[pidx[i]]), len(o), window, keep)
        b = brute(model, prompts[pidx[i]], o, segs, ncond[i], bias[pidx[i]], [BOS])
        g = groups[i]
        exp["psum"][g] += b["psum"] / 2; exp["scope"][g] += b["scope"] / 2
        exp["reach_trunc"][g] += -np.expm1(b["ls"]) / 2; hm[g] += b["hmean"] / 2
        np.testing.assert_allclose(res["log_survival"][i], b["ls"][track], rtol=1e-4, atol=1e-6)
    for k in exp:
        np.testing.assert_allclose(res[k], exp[k], rtol=1e-4, atol=1e-6, err_msg=k)
    np.testing.assert_allclose(res["gen_hidden_mean"], hm, rtol=1e-4, atol=1e-5)
    assert res["n_trajectories"].tolist() == [2, 2, 2]
    assert (res["psum"][:, BOS] == 0).all()
    assert any(len(generation_segments(len(prompts[pidx[i]]), len(o), window, keep)) > 1 for i, o in enumerate(outs))
