<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-3776AB?logo=python&logoColor=white" alt="Python ≥3.10"/>
  <img src="https://img.shields.io/badge/inference-SGLang-F97316" alt="SGLang"/>
  <a href="https://arxiv.org/abs/2602.03730"><img src="https://img.shields.io/badge/arXiv-2602.03730-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"/></a>
</p>

# SCOPE & REACH — `quick_sco_re`

**Variance-reduced event-probability estimation for any autoregressive generative model.**

Standard Monte Carlo (MC) estimation generates *n* trajectory completions from a model and counts how many contain a target event. This is unbiased but wasteful — the model's full next-token probability distribution is computed at every decoding step, yet only the single sampled token is kept. SCOPE and REACH use those discarded distributions to produce better estimates from far fewer samples.

This package, `quick_sco_re`, is a **model-agnostic and data-agnostic** implementation. It works with any autoregressive model that can be served via [SGLang](https://github.com/sgl-project/sglang) — you just need to specify which token IDs represent the target event, which represent sequence termination, and which (if any) should be suppressed.

<p align="center">
  <a href="https://arxiv.org/abs/2602.03730"><strong>Paper</strong></a> ·
  <a href="#configuration-reference"><strong>Config Reference</strong></a> ·
  <a href="#quickstart"><strong>Quickstart</strong></a>
</p>

---

## Estimators

All three estimators are computed from the same generation + scoring pipeline:

**M0 (Simple MC)** — Generate *n* trajectories where the target event token can naturally occur as a stop token. The estimate is the fraction of trajectories that terminated with the target event. This is the standard baseline.

**M1 / SCOPE** (Sum of Conditional Outcome Probability Estimator) — Using the same M0 trajectories, run a scoring pass that extracts P(target event) from the model's logits at every generated position. The score is the **sum** of these probabilities. Averaged over trajectories, this is an unbiased estimator that produces continuous risk scores instead of discrete 0/n, 1/n, … fractions.

**M2 / REACH** (Risk Estimation from Anticipated Conditional Hazards) — Generate a separate set of trajectories where the target event token is **suppressed** (forced to probability zero via logit bias), so the model must "imagine" the full trajectory as if the event never happened. Then score those trajectories the same way, and compute 1 − ∏(1 − *p*ₜ). This is the probability the event *would have* occurred somewhere along the trajectory. REACH is **provably variance-reducing** over MC for any model and any outcome.

> SCOPE and REACH matched 100-sample MC discrimination (AUC) using only 10–11 samples in our experiments — a ~10× compute reduction. See the [paper](https://arxiv.org/abs/2602.03730) for details.

---

## How It Works

The pipeline has two phases, both served through a single SGLang engine:

**1. Generation** — For each input sequence, generate *n* M1 trajectories (target event allowed) and *n* M2 trajectories (target event suppressed via logit bias). Trajectories are interleaved by input sequence so that SGLang's radix cache keeps the shared prefix KV entries warm across all samples for the same input. Generation runs *without* logprobs for maximum throughput.

**2. Scoring** — Each trajectory is scored in a separate **prefill-only forward pass**: the full sequence (prompt + generated tokens) is fed back through the model, requesting only the logprob of the target event token at each generated position. The per-position probabilities are then aggregated into the SCOPE or REACH score. This two-pass design decouples generation throughput from logprob extraction.

**Persistence** — Trajectories can be saved to disk and loaded back for deferred scoring. The IO roundtrip is validated end-to-end: scores computed from reloaded trajectories are identical to those computed from the original generation results.

---

## Install

```bash
pip install "quick_sco_re[sglang]"        # generation and scoring with SGLang
pip install "quick_sco_re[sglang,hf]"     # + torch / transformers: score_full_vocab, parity checks, export_backbone
pip install quick_sco_re                  # numpy only: load and analyse saved trajectories and scores
```

SGLang pins an exact torch build, so it is an extra rather than a dependency: installing
`quick_sco_re` alone never replaces your torch. Tested with SGLang 0.5.10 on GPU; the SGLang
interfaces it uses were checked unchanged through 0.5.23. Without SGLang the package still
imports; `create_engine` then says what to install.

From a clone (to run the tests or the cocoa example):

```bash
git clone https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git
cd SCOPE_REACH_optimized_inference
pip install -e ".[all,dev]"
```

---

## <a name="quickstart"></a> Quickstart

```python
import sglang as sgl
from quick_sco_re import GenerationConfig, generate_and_score, save_scores

# 1. Point SGLang at your model
engine = sgl.Engine(
    model_path="path/to/your/model",
    skip_tokenizer_init=True,
    context_length=4096,
)

# 2. Define which tokens matter
config = GenerationConfig(
    max_len=4096,
    n_samp=20,                     # MC samples per input per trajectory type
    target_event_id=42,            # the token whose probability you want to estimate
    end_token_ids={43, 44},        # tokens that naturally terminate a sequence
    suppressed_ids=[0, 1],         # tokens to suppress (e.g. PAD) via logit bias
)

# 3. Run — input_sequences is a list[list[int]] of tokenized prompts
trajectories, results = await generate_and_score(
    engine, config, input_sequences,
    target_token_id=config.target_event_id,
)

# 4. Read results
for i, r in enumerate(results):
    mc_estimate  = sum(r.m0_samples) / len(r.m0_samples)  # M0: fraction
    scope_score  = sum(r.m1_samples) / len(r.m1_samples)  # M1: mean SCOPE
    reach_score  = sum(r.m2_samples) / len(r.m2_samples)  # M2: mean REACH

# 5. Persist
save_scores(results, "scores.npz")
```

---

## <a name="configuration-reference"></a> Configuration Reference

Everything is controlled through `GenerationConfig`:

### Required Parameters

| Parameter | Type | Description |
|:---|:---|:---|
| `max_len` | `int` | Maximum total sequence length (prompt + generated tokens). |
| `n_samp` | `int` | Number of MC samples per input sequence, per trajectory type. Total trajectories = 2 × n_samp × n_inputs (M1 + M2). |
| `target_event_id` | `int` | **The token ID of the event you want to estimate the probability of.** This is the core parameter — SCOPE and REACH compute P(this token) at every decoding step. |
| `end_token_ids` | `set[int]` | Token IDs that naturally terminate a sequence. For M1 trajectories, `target_event_id` is also added as a stop token. For M2 trajectories, only these are used. |

### Optional Parameters

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `suppressed_ids` | `list[int]` | `[]` | Token IDs to suppress via a large negative logit bias during generation (e.g. padding tokens, or any token that shouldn't appear in valid trajectories). |
| `temperature` | `float` | `1.0` | Sampling temperature. **Should be 1.0 for proper MC estimation.** |
| `trunc_id` | `int \| None` | `None` | Token ID to force when a time horizon is exceeded. Required for time-based stopping. |
| `token_id_to_minutes` | `dict[int, float]` | `{}` | Mapping from token ID → elapsed simulated time (in minutes). Used to accumulate elapsed time during generation for time-based stopping. |
| `max_time` | `float \| None` | `None` | Maximum simulated time in minutes. When elapsed time exceeds this, generation is terminated. Requires `trunc_id`. |
| `time_check_interval` | `int` | `100` | Tokens between time-horizon checks. The logit processor is a no-op for the first `time_check_interval` tokens, then checks every `time_check_interval` tokens. Higher values = less overhead, but coarser online detection (post-hoc truncation is always exact). |
| `context_window` | `int \| None` | `None` | Rolling context window in tokens. No request conditions the model on more than this; when the context fills it, it is cut back to the last `window_keep` tokens. Scoring passes are windowed the same way. Set it to the packed block length the model was trained on. `max_len` still caps the total length. `None` = whole sequence, as before. |
| `window_keep` | `int \| None` | `None` | Tokens kept when the window is cut back. Defaults to ¾ of `context_window`. |

### Engine Setup

Use `create_engine()` for correct SGLang flags:

```python
from quick_sco_re import create_engine

# Without time-based stopping
engine = create_engine(model_path="path/to/model", max_len=4096)

# With time-based stopping (enables custom logit processor + disables overlap scheduling)
engine = create_engine(model_path="path/to/model", max_len=4096, use_time_horizon=True)

# With a rolling window, the engine only needs the window as its context length
engine = create_engine(model_path="path/to/model", max_len=10000, context_length=1024)
```

Or configure SGLang directly — the key flags for time-based stopping are `disable_overlap_schedule=True` and `enable_custom_logit_processor=True`.

---

## Time-Based Stopping

Some models generate sequences where tokens carry temporal semantics (e.g. time-bin tokens in event sequences). `quick_sco_re` supports bounding generation by **simulated elapsed time** rather than token count, using a two-layer approach:

1. **Deferred logit processor** — a custom SGLang `CustomLogitProcessor` that is a complete no-op for the first `time_check_interval` tokens, then periodically checks accumulated simulated time via incremental summation. When the horizon is exceeded, it forces `trunc_id`, terminating generation. The deferred + periodic design avoids per-step throughput penalties.

2. **Post-hoc exact truncation** — because the processor only checks periodically, the trajectory may overshoot by up to `time_check_interval` tokens. After generation, the output is walked token-by-token to find the exact position where elapsed time first exceeded `max_time`, and `output_ids` are trimmed to end just before that point.

This gives exact time-horizon semantics at near-baseline generation throughput.

---

## Rolling Window and First-Token Restriction

Two options that fixed over-estimation of adverse outcomes with cocoa/cotorra models.

**Rolling context window** (`GenerationConfig.context_window`, `window_keep`). Models trained on fixed packed blocks degrade when conditioned on longer contexts. With a window set, each trajectory is generated in segments and the context is cut back to the last `window_keep` tokens whenever it fills the window. Inline SCOPE/REACH logprobs from all segments are concatenated. The scoring pass for REACH regenerations uses the same windowing. With time-based stopping, elapsed time is carried across segments.

**First-token restriction** (`first_token_bias` on `generate_trajectory`; `first_token_biases` per patient on `generate_trajectories` / `generate_and_score`). The first generated token is sampled alone, with an extra logit bias, before generation continues normally. One use is a *silence restriction*. A prompt cut at 24 hours ends at the patient's last event, so a patient silent for *x* minutes before the cut must have their next event at least *x* minutes later. Only time-gap tokens whose bin can contain or exceed *x* are allowed. The bias must forbid every tracked token. That step's tracked-event probability is therefore 0, and it is recorded as `GeneratedTrajectory.n_conditioned_steps`, which inline estimates and scoring passes skip.

`silence_restriction_bias(silence_minutes, time_token_bounds, vocab_ids, mode="hard")` builds that bias. `time_token_bounds` maps each time-gap token id to its `(lo, hi]` bin in minutes; gaps are floored to whole minutes. Mode `"hard"` allows every bin with `hi >= floor(x)`. Mode `"prop"` also scales the bin containing *x* by the share of its whole-minute values that are at least *x*. It returns `None` (no restriction) for unknown silences and for silences shorter than the smallest bin.

```python
from quick_sco_re import silence_restriction_bias

bounds = {time_5m_15m: (5, 15), time_15m_1h: (15, 60), time_1h_2h: (60, 120)}  # token id -> (lo, hi]
biases = [silence_restriction_bias(x, bounds, range(vocab_size)) for x in silence_minutes]
trajectories = await generate_trajectories(engine, config, prompts, ["M1"], first_token_biases=biases)
```

---

## Full-Vocabulary Scoring with Hugging Face

`score_trajectory` asks SGLang for one token's probability at a time. Asking SGLang for the whole next-token distribution is CPU-bound: it ships every logprob through its Python tokenizer manager and leaves the GPU mostly idle. `score_full_vocab` re-reads trajectories with a teacher-forced Hugging Face forward pass instead, and reduces the full distributions on the GPU. For **every** token at once it returns:

| Output | Meaning |
|:---|:---|
| `psum` | sum of p_t(v) over generated steps: expected count |
| `scope` | the same sum up to and including v's first occurrence (SCOPE) |
| `reach_trunc` | 1 − ∏(1 − p_t(v)) over those steps. This is REACH along an outcome-free path, and biased low for tokens the trajectory generated. Combine it with `generate_m2_from_m1_trajectory` regenerations for multi-outcome REACH (use `log_survival_ids`). |
| `gen_hidden_mean` | optional: the mean final hidden state over each trajectory's generated tokens |

All of these are averaged over the trajectories of each group (e.g. patient). Every generated token is scored in exactly the context it was sampled in. `generation_segments` reproduces the rolling window of `generate_trajectory`. Restricted first steps are scored under their restriction, and suppressed tokens get probability 0. Install with `pip install -e ".[hf]"`.

```python
import torch
from transformers import AutoModelForCausalLM
from quick_sco_re import score_full_vocab, check_engine_parity, check_rope_config

check_rope_config(model_path)                         # transformers reads RoPE as saved
hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).cuda().eval()
await check_engine_parity(engine, hf, prompts[:16], context_window=4096)   # SGLang and HF agree
feats = score_full_vocab(
    hf, prompts, [t.output_ids for t in trajectories],
    prompt_index=[t.patient_idx for t in trajectories], group_ids=[t.patient_idx for t in trajectories],
    context_window=4096, window_keep=3072,
    n_conditioned=[t.n_conditioned_steps for t in trajectories], first_token_biases=biases,
    suppressed_ids=[bos_id], return_hidden=True, autocast_dtype=torch.bfloat16)
```

Generate without logprobs, shut the SGLang engine down, then score. Never mix requests that ask for logprobs with requests that don't in one SGLang batch: its scheduler fails on such batches.

---

## cocoa Example

[`cocoa_example/`](https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference/tree/main/cocoa_example) (in the repository, not the PyPI package) is a complete pipeline for models trained on [cocoa](https://github.com/bbj-lab/cocoa) EHR timelines. It runs multi-outcome M0 / SCOPE / REACH and optional full-vocabulary features, with the rolling window and silence restriction on, from cocoa's winnower output and a YAML config. Written against cocoa-tokenizer 26.11; its tests run without a GPU, on synthetic files and, when cocoa is installed, on timelines cocoa itself collates, tokenizes and winnows.

```bash
python -m cocoa_example.run_cocoa --config cocoa_example/config_example.yaml
```

## cotorra Models

[cotorra](https://github.com/bbj-lab/cotorra) (26.11) scores with this package (`cotorra[gen]`). A model it trains
without secondary objectives is a stock `*ForCausalLM` and loads as is. One trained with `tte` / `tnt` /
`disposition` heads is saved as `CotorraForCausalLM` (`model_type: cotorra`), which SGLang cannot load;
`create_engine` and `check_rope_config` refuse it with a pointer to:

```python
from quick_sco_re import export_backbone
export_backbone("path/to/mdl-run", "path/to/mdl-run-backbone")   # drops heads.*, unwraps text_config
```

The exported backbone has the same next-token distribution (identical logits under transformers).

---

## API Overview

### Scheduler Functions

| Function | Description |
|:---|:---|
| `generate_and_score(engine, config, input_sequences, target_token_id)` | Generate all trajectories, then score them. |
| `generate_trajectories(engine, config, input_sequences)` | Generation only. Returns `list[GeneratedTrajectory]`. |
| `score_trajectories(engine, config, trajectories, input_sequences)` | Scoring only. Takes previously generated trajectories and returns `list[ScoredTrajectory]`. |
| `aggregate_results(scored_trajectories, num_inputs, target_event_id)` | Aggregates scored trajectories into per-input `PatientResults` (M0, M1, M2 sample lists). |

### Persistence

| Function | Description |
|:---|:---|
| `save_trajectories(trajectories, output_dir)` | Save raw trajectories as compressed `.npz` + optional `config.json`. |
| `load_trajectories(input_dir)` | Load trajectories back. Returns `(list[GeneratedTrajectory], config_or_None)`. |
| `save_scores(results, output_path)` | Save per-input mean M0/M1/M2 scores as a `.npz` file. |
| `load_scores(input_path)` | Load scores. Returns `{"M0": array, "M1": array, "M2": array}`. |

---

## Repository Structure

```
├── quick_sco_re/              # Core package
│   ├── structures.py         # GenerationConfig, GeneratedTrajectory, ScoredTrajectory, PatientResults
│   ├── generation.py         # Trajectory generation + DeferredTimeHorizonProcessor
│   ├── scoring.py            # Prefill-only logprob extraction → SCOPE/REACH scores
│   ├── scheduler.py          # Orchestration (generate-and-score, generate-only, score-only)
│   ├── io.py                 # Save/load trajectories and scores (.npz)
│   ├── restrictions.py       # First-token biases (silence restriction)
│   ├── hf_scoring.py         # Full-vocabulary teacher-forced scoring with Hugging Face ([hf] extra)
│   └── diagnostics.py        # Trajectory logging; SGLang/HF parity and RoPE config checks
├── benchmarks/               # End-to-end benchmarks (requires GPU + model + data)
│   ├── run_benchmarks.py     # CLI entry point
│   ├── common.py             # PhaseRunner, config builders, metric computation
│   ├── resampling.py         # Length-biased stratified resampling for AUC analysis
│   ├── bench_baseline.py     # MC-only baseline
│   ├── bench_time_horizon.py # SCOPE/REACH with time-based stopping
│   ├── bench_comparison.py   # Side-by-side timing & AUC comparison
│   ├── bench_truncation.py   # Post-hoc vs. online truncation equivalence
│   └── bench_io_roundtrip.py # Validates save/load roundtrip produces identical scores
├── cocoa_example/            # Complete cocoa pipeline: adapter, multi-outcome runner, config, tests
├── tests/                    # Unit tests (no GPU required)
├── slurm/                    # HPC job scripts
└── pyproject.toml
```

---

## Tests

Unit tests run without a GPU or model (`test_hf_scoring.py` builds a tiny random Llama; it is skipped without torch/transformers):

```bash
pytest            # tests/ and cocoa_example/tests/
```

---

## Citation

```bibtex
@article{solo2026efficient,
  title   = {Efficient Variance-Reduced Estimation from Generative {EHR} Models:
             The {SCOPE} and {REACH} Estimators},
  author  = {Solo, Luke and McDermott, Matthew B.A. and Parker, William F.
             and Ramadan, Bashar and Burkhart, Michael C.
             and Beaulieu-Jones, Brett K.},
  journal = {arXiv preprint arXiv:2602.03730},
  year    = {2026}
}
```
