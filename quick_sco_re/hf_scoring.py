"""Full-vocabulary scoring of generated trajectories with a Hugging Face model.

score_trajectory (SGLang) returns P(target) for one token at a time. Asking
SGLang for the whole next-token distribution is CPU-bound: shipping every
logprob through its Python tokenizer manager leaves the GPU mostly idle. This
module instead re-reads each trajectory with one teacher-forced HF forward pass
and reduces the full distributions on the GPU, giving for EVERY token v at once:

  psum         sum over generated steps of p_t(v)                  expected count
  scope        the same sum up to and including v's first occurrence     SCOPE
  reach_trunc  1 - prod (1 - p_t(v)) over those same steps
               (for a token the trajectory never generated this is the REACH
               estimate along an outcome-free path; for one it did generate it
               is biased low, as the steps after it are not scored)

plus, optionally, the mean final hidden state over each trajectory's generated
tokens, and the per-trajectory log-survival of chosen tokens (to combine with
generate_m2_from_m1_trajectory regenerations into multi-outcome REACH).

Every generated token is scored in exactly the context it was sampled in:
generation_segments() reproduces the rolling window of generate_trajectory
(GenerationConfig.context_window / window_keep), so a trajectory that used the
window is read in the same segments. Steps sampled under a first-token
restriction (GeneratedTrajectory.n_conditioned_steps) are scored under that
restriction: the bias is applied to the distribution and renormalised.
Suppressed tokens (e.g. BOS) get probability 0 and the rest is renormalised,
which is the distribution generation actually sampled from.

Requires torch and transformers (pip install "quick_sco_re[hf]"). Check the HF
model against the SGLang engine first (diagnostics.check_engine_parity).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .generation import _SGLANG_OVERHEAD


def generation_segments(
    prompt_len: int,
    n_generated: int,
    context_window: int | None,
    window_keep: int | None = None,
    overhead: int = _SGLANG_OVERHEAD,
) -> list[tuple[int, int, int]]:
    """Context segments of a trajectory, as generate_trajectory produces them.

    Returns [(c0, g0, g1), ...]: generated tokens g0 .. g1-1 were sampled with
    the context starting at absolute position c0 of prompt + generated tokens.
    With no window the whole sequence is one segment from position 0.
    """
    if n_generated <= 0:
        return []
    if context_window is None:
        return [(0, 0, n_generated)]
    keep = window_keep or (context_window - context_window // 4)
    span = context_window - overhead - 1          # most tokens a request may condition on, incl. new ones
    if not 0 < keep < span:
        raise ValueError(f"window_keep={keep} must be in (0, {span})")
    ctx = min(prompt_len, span - 1)               # as much prompt as fits, leaving room for one new token
    c0, g, segs = prompt_len - ctx, 0, []
    while g < n_generated:
        take = min(span - ctx, n_generated - g)
        segs.append((c0, g, g + take))
        g += take
        ctx += take
        if g < n_generated:                       # window full: cut back to the last `keep` tokens
            ctx = keep
            c0 = prompt_len + g - keep
    return segs


def score_full_vocab(
    model,
    prompts: Sequence[Sequence[int]],
    outputs: Sequence[Sequence[int]],
    *,
    prompt_index: Sequence[int] | None = None,
    group_ids: Sequence[int] | None = None,
    n_groups: int | None = None,
    context_window: int | None = None,
    window_keep: int | None = None,
    overhead: int = _SGLANG_OVERHEAD,
    n_conditioned: Sequence[int] | None = None,
    first_token_biases: Sequence[Mapping[int, float] | None] | None = None,
    suppressed_ids: Sequence[int] = (),
    log_survival_ids: Sequence[int] | None = None,
    return_hidden: bool = False,
    token_budget: int = 60000,
    autocast_dtype=None,
) -> dict[str, np.ndarray]:
    """Score every trajectory's generated tokens against the full vocabulary.

    Args:
        model: A Hugging Face causal LM (``model.model`` the decoder, ``model.lm_head``
            the output layer), on the device to score on, in eval mode.
        prompts: Prompt token ids.
        outputs: Generated token ids per trajectory (the stop token included if
            generation ended on one, as GeneratedTrajectory.output_ids does).
        prompt_index: Per trajectory, the index of its prompt in ``prompts``
            (default: trajectory i uses prompt i).
        group_ids: Per trajectory, the group to average into (e.g. the patient);
            default one group per trajectory.
        n_groups: Number of groups (default max(group_ids) + 1).
        context_window, window_keep, overhead: As in GenerationConfig /
            generate_trajectory; must match how the trajectories were generated.
        n_conditioned: Per trajectory, leading steps sampled under a first-token
            restriction (GeneratedTrajectory.n_conditioned_steps).
        first_token_biases: Per PROMPT, the first-token bias used (or None).
        suppressed_ids: Tokens suppressed during generation (e.g. BOS).
        log_survival_ids: Token ids whose per-trajectory log-survival
            sum log(1 - p_t) up to first occurrence is returned.
        return_hidden: Also return each group's mean over trajectories of the mean
            final hidden state over the trajectory's generated tokens (the state
            after reading each token).
        token_budget: Tokens per forward batch.
        autocast_dtype: e.g. torch.bfloat16 for mixed precision (None = model dtype).

    Returns:
        dict of numpy arrays, per group averaged over its trajectories:
        ``psum``, ``scope``, ``reach_trunc`` (n_groups, V); ``n_trajectories``
        (n_groups,); ``mean_steps`` (n_groups,); and when asked ``gen_hidden_mean``
        (n_groups, H) and ``log_survival`` (n_trajectories, len(log_survival_ids)).
    """
    import torch

    nt = len(outputs)
    pidx = np.arange(nt) if prompt_index is None else np.asarray(prompt_index, dtype=np.int64)
    gid = np.arange(nt) if group_ids is None else np.asarray(group_ids, dtype=np.int64)
    ng = int(n_groups if n_groups is not None else (gid.max() + 1 if nt else 0))
    ncond = np.zeros(nt, np.int64) if n_conditioned is None else np.asarray(n_conditioned, dtype=np.int64)
    dev = next(model.parameters()).device
    V = model.lm_head.out_features if hasattr(model.lm_head, "out_features") else model.config.vocab_size
    H = model.config.hidden_size
    outs = [np.asarray(o, dtype=np.int64) for o in outputs]
    plens = np.array([len(prompts[p]) for p in pidx])

    # first-token restriction as multiplicative weights on p, per prompt
    restrict_w = None
    if first_token_biases is not None and (ncond > 0).any():
        restrict_w = torch.ones((len(prompts), V), dtype=torch.float32, device=dev)
        for j, b in enumerate(first_token_biases):
            if b:
                for t, v in b.items():
                    restrict_w[j, int(t)] = 0.0 if v <= -5000 else float(np.exp(v))
    supp = torch.tensor(sorted(set(int(s) for s in suppressed_ids)), dtype=torch.long, device=dev)
    lsid = None if log_survival_ids is None else torch.tensor(list(log_survival_ids), dtype=torch.long, device=dev)

    first = np.empty((nt, V), np.int64)            # generated index of each token's first occurrence (len if none)
    segs = []
    for i, o in enumerate(outs):
        first[i] = len(o)
        u, fi = np.unique(o, return_index=True)
        first[i, u] = fi
        segs.append(generation_segments(int(plens[i]), len(o), context_window, window_keep, overhead))
    pieces = [(i, c0, g0, g1) for i in range(nt) for (c0, g0, g1) in segs[i]]
    nseg = np.array([len(s) for s in segs])
    plen = lambda p: int(plens[p[0]]) + p[3] - p[1]
    pieces.sort(key=plen)

    acc = {k: torch.zeros((ng, V), dtype=torch.float64, device=dev) for k in ("psum", "scope", "reach_trunc")}
    multi: dict[int, torch.Tensor] = {}
    hsum = torch.zeros((nt, H), dtype=torch.float32, device=dev) if return_hidden else None
    ls_out = np.zeros((nt, 0 if lsid is None else len(lsid)))

    def run(batch):
        B = len(batch)
        L = max(plen(p) for p in batch)
        R = max(p[3] - p[2] for p in batch)
        ids = torch.zeros((B, L), dtype=torch.long)
        am = torch.zeros_like(ids)
        rowpos = torch.zeros((B, R), dtype=torch.long)
        valid = torch.zeros((B, R), dtype=torch.bool)
        gabs = torch.zeros((B, R), dtype=torch.long)
        for b, (i, c0, g0, g1) in enumerate(batch):
            P = int(plens[i])
            full = np.concatenate([np.asarray(prompts[pidx[i]], dtype=np.int64), outs[i]])
            seq = full[c0:P + g1]                                   # context through the segment's last generated token
            ids[b, :len(seq)] = torch.from_numpy(seq)
            am[b, :len(seq)] = 1
            r = g1 - g0
            rowpos[b, :r] = torch.arange(P + g0 - 1 - c0, P + g1 - 1 - c0)   # positions predicting each generated token
            valid[b, :r] = True
            gabs[b, :r] = torch.arange(g0, g1)
        ids, am, rowpos, valid, gabs = (x.to(dev) for x in (ids, am, rowpos, valid, gabs))
        ctx = torch.autocast(dev.type, dtype=autocast_dtype) if autocast_dtype is not None else torch.autocast(dev.type, enabled=False)
        with ctx:
            hs = model.model(input_ids=ids, attention_mask=am).last_hidden_state
            hpred = hs.gather(1, rowpos[..., None].expand(-1, -1, hs.size(-1)))
            logits = model.lm_head(hpred)
            hread = hs.gather(1, (rowpos + 1)[..., None].expand(-1, -1, hs.size(-1))) if return_hidden else None
        p = torch.softmax(logits.float(), -1)
        if len(supp):
            p[..., supp] = 0
        tr = torch.tensor([b_[0] for b_ in batch], device=dev)
        if restrict_w is not None:
            nc = torch.from_numpy(ncond[[b_[0] for b_ in batch]]).to(dev)
            rs = (gabs < nc[:, None]) & valid
            if rs.any():
                w = restrict_w[torch.from_numpy(pidx[[b_[0] for b_ in batch]]).to(dev)]
                p = torch.where(rs[..., None], p * w[:, None, :], p)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
        p = p * valid[..., None]
        fst = torch.from_numpy(first[[b_[0] for b_ in batch]]).to(dev)
        upto = (gabs[:, :, None] <= fst[:, None, :]) & valid[..., None]
        psum = p.sum(1, dtype=torch.float64)
        scope = (p * upto).sum(1, dtype=torch.float64)
        ls = (torch.log1p(-p.clamp(max=1 - 1e-7)) * upto).sum(1, dtype=torch.float64)
        hs_sum = (hread.float() * valid[..., None]).sum(1) if return_hidden else None
        return tr, psum, scope, ls, hs_sum

    with torch.no_grad():
        i0 = 0
        while i0 < len(pieces):
            b = 1
            while i0 + b < len(pieces) and (b + 1) * plen(pieces[i0 + b]) <= token_budget:
                b += 1
            batch = pieces[i0:i0 + b]
            i0 += b
            tr, psum, scope, ls, hs_sum = run(batch)
            g = torch.from_numpy(gid[tr.cpu().numpy()]).to(dev)
            acc["psum"].index_add_(0, g, psum)
            acc["scope"].index_add_(0, g, scope)
            single = torch.from_numpy(nseg[tr.cpu().numpy()] == 1).to(dev)
            if single.any():
                acc["reach_trunc"].index_add_(0, g[single], -torch.expm1(ls[single]))
            for b_ in torch.nonzero(~single).flatten().tolist():
                i = batch[b_][0]
                multi[i] = multi[i] + ls[b_] if i in multi else ls[b_].clone()
            if lsid is not None:
                np.add.at(ls_out, tr.cpu().numpy(), ls[:, lsid].cpu().numpy())
            if return_hidden:
                hsum.index_add_(0, tr, hs_sum)
    for i, ls in multi.items():
        acc["reach_trunc"][int(gid[i])] += -torch.expm1(ls)

    count = np.bincount(gid, minlength=ng).astype(np.float64)
    steps = np.bincount(gid, weights=[len(o) for o in outs], minlength=ng)
    den = torch.from_numpy(np.maximum(count, 1)).to(dev)[:, None]
    res = {k: (v / den).cpu().numpy() for k, v in acc.items()}
    res["n_trajectories"] = count.astype(np.int64)
    res["mean_steps"] = steps / np.maximum(count, 1)
    if return_hidden:
        n = torch.tensor([max(len(o), 1) for o in outs], dtype=torch.float32, device=dev)[:, None]
        gm = torch.zeros((ng, H), dtype=torch.float32, device=dev).index_add_(0, torch.from_numpy(gid).to(dev), hsum / n)
        res["gen_hidden_mean"] = (gm / den.float()).cpu().numpy()
    if lsid is not None:
        res["log_survival"] = ls_out
    return res
