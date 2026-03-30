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

---

## Install

```bash
git clone https://github.com/lukesolo-ml/SCOPE_REACH_optimized_inference.git
cd SCOPE_REACH_optimized_inference
pip install -e .            # numpy + sglang
pip install -e ".[dev]"     # adds pytest, pytest-asyncio
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

### Engine Setup

Use `create_engine()` for correct SGLang flags:

```python
from quick_sco_re import create_engine

# Without time-based stopping
engine = create_engine(model_path="path/to/model", max_len=4096)

# With time-based stopping (enables custom logit processor + disables overlap scheduling)
engine = create_engine(model_path="path/to/model", max_len=4096, use_time_horizon=True)
```

Or configure SGLang directly — the key flags for time-based stopping are `disable_overlap_schedule=True` and `enable_custom_logit_processor=True`.

---

## Time-Based Stopping

Some models generate sequences where tokens carry temporal semantics (e.g. time-bin tokens in event sequences). `quick_sco_re` supports bounding generation by **simulated elapsed time** rather than token count, using a two-layer approach:

1. **Deferred logit processor** — a custom SGLang `CustomLogitProcessor` that is a complete no-op for the first `time_check_interval` tokens, then periodically checks accumulated simulated time via incremental summation. When the horizon is exceeded, it forces `trunc_id`, terminating generation. The deferred + periodic design avoids per-step throughput penalties.

2. **Post-hoc exact truncation** — because the processor only checks periodically, the trajectory may overshoot by up to `time_check_interval` tokens. After generation, the output is walked token-by-token to find the exact position where elapsed time first exceeded `max_time`, and `output_ids` are trimmed to end just before that point.

This gives exact time-horizon semantics at near-baseline generation throughput.

---

## API Overview

### Scheduler Functions

| Function | Description |
|:---|:---|
| `generate_and_score(engine, config, input_sequences, target_token_id)` | **Recommended.** Generate all trajectories, then score them. Keeps radix cache warm for the scoring pass. |
| `generate_and_score_interleaved(engine, config, input_sequences, target_token_id)` | Alternative: score each trajectory immediately after its generation completes. Maximizes per-sequence cache locality. |
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
│   ├── scheduler.py          # Orchestration (sequential, interleaved, generate-only, score-only)
│   ├── io.py                 # Save/load trajectories and scores (.npz)
│   └── diagnostics.py        # Trajectory length, termination, and truncation logging
├── benchmarks/               # End-to-end benchmarks (requires GPU + model + data)
│   ├── run_benchmarks.py     # CLI entry point
│   ├── common.py             # PhaseRunner, config builders, metric computation
│   ├── resampling.py         # Length-biased stratified resampling for AUC analysis
│   ├── bench_baseline.py     # MC-only baseline
│   ├── bench_time_horizon.py # SCOPE/REACH with time-based stopping
│   ├── bench_comparison.py   # Side-by-side timing & AUC comparison
│   ├── bench_truncation.py   # Post-hoc vs. online truncation equivalence
│   └── bench_interleave.py   # Sequential vs. interleaved comparison
├── tests/                    # Unit tests (no GPU required)
├── slurm/                    # HPC job scripts
└── pyproject.toml
```

---

## Tests

Unit tests run without a GPU or model:

```bash
pytest tests/ -v
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
