# cocoa example: SCOPE / REACH on cocoa EHR timelines

A complete pipeline for models trained on [cocoa](https://github.com/bbj-lab/cocoa) tokenized timelines,
built on `quick_sco_re`. It reads cocoa's winnower output and estimates, for every tracked outcome, its
probability after the prediction cut:

- **M0:** Monte Carlo
- **SCOPE**
- **REACH**

Optionally, it also computes full-vocabulary features of the generated trajectories.

## Inputs

From cocoa's `pipeline` / `winnow`:

| File | What it holds |
|---|---|
| `held_out_for_inference.parquet` | `subject_id`, `tokens_past` (the prompt), `s_elapsed_past` (seconds since admission per prompt token), and per event `<EVENT>_past` / `<EVENT>_future` booleans |
| `tokenizer.yaml` | `lookup:` token name → id |

**Ground truth is always the winnower's `<EVENT>_future` flag.** Patients whose prompt already holds an
event (`<EVENT>_past`) get no estimate for it.

The model is a Hugging Face causal LM directory, readable by both SGLang and transformers.

## Run

```bash
pip install -e ".[cocoa]"        # from the repository root: torch, transformers, polars, pyyaml, scikit-learn
python -m cocoa_example.run_cocoa --config cocoa_example/config_example.yaml --dry-run   # cohort + settings, no GPU
python -m cocoa_example.run_cocoa --config cocoa_example/config_example.yaml
python -m cocoa_example.run_cocoa --config my.yaml --shard-idx 3 --n-shards 16         # patients i % 16 == 3
```

Outputs, in `output_dir`, or `output_dir/shard_NNN` when sharded:

| File | Contents |
|---|---|
| `patient_index.parquet` | `patient_idx`, `subject_id`, prompt length, every winnower flag |
| `scores_<EVENT>.npz` | per-patient M0 / M1 (SCOPE) / M2 (REACH) means and every sample (`quick_sco_re.load_scores`) |
| `trajectories/` | the M1 trajectories (`quick_sco_re.load_trajectories`) |
| `features.npz` | with `features.enabled`: per patient, `psum`, `scope`, `reach_trunc` over the whole vocabulary and `gen_hidden_mean` |
| `run_summary.json` | settings, token counts, and per outcome the means, prevalence and AUCs |

**These outputs are patient-level. Keep them off shared or hosted locations, and never commit them.**

## What it does

1. **Cohort.** Rows failing `cohort.require_flag` / `exclude_flag` are dropped, and so are prompts that already hold
   an end token (the stay ended before the cut). `prompt_prefix_tokens` are put in front of every prompt.
   Prompts longer than `max_len` are left-truncated (`prompt_overflow: truncate_left`) or dropped (`drop`).
2. **Checks.** transformers must read the model's RoPE settings as saved (`check_rope_config`). SGLang's
   next-token distribution must match HF's on 16 prompts (`check_engine_parity`). This catches a wrong attention
   backend or a misread config before anything is generated.
3. **M1 pass.** There are `n_samp` trajectories per patient, each run to discharge (any `DSCG//` token) or `max_len`.
   Every tracked event gets inline estimates from the same trajectory.
   - **Rolling window** (`generation.rolling_window`): the model is never conditioned on more than
     `context_window` tokens, which should be the packed block length it was trained on. When the context
     fills, it is cut back to the last `keep` tokens.
   - **Silence restriction** (`generation.silence_restriction`): a prompt cut at 24 h ends at the patient's
     last event. A patient silent for *x* minutes must next have a time gap of at least *x*, so the first
     generated token is restricted to the `TIME//` bins that can hold it (`quick_sco_re.silence_restriction_bias`).
4. **Per outcome:**
   - **M0** is whether the outcome appeared.
   - **SCOPE** is the inline sum of its probability up to the first occurrence.
   - **REACH:** trajectories without the outcome are outcome-free paths already, so their inline
     1 − ∏(1 − p) is used as is. Trajectories with it are cut before it, and the rest is regenerated with the
     outcome suppressed (`generate_m2_from_m1_trajectory`), then scored.
5. **Features** (optional): `quick_sco_re.score_full_vocab` re-reads the M1 trajectories with HF, in the
   windows and under the restriction they were generated with.

## Settings that matter

| Setting | Use |
|---|---|
| `rolling_window.context_window` | the training block length (e.g. 4096). Without it, long trajectories condition on more context than the model ever saw, which inflated adverse-outcome estimates. |
| `silence_restriction.horizon_minutes` | must equal the winnower's prediction cut. The loader refuses prompts with events after it. |
| `prompt_prefix_tokens` | e.g. `[EOS]` if training blocks put EOS before each stay. |
| `methods: [M1]` | skips the REACH regenerations (M0 and SCOPE only). |

## Tests

```bash
pytest cocoa_example/tests        # synthetic cocoa files built in a temp folder + a scripted fake engine; no GPU
```

The tests cover cohort filtering, labels, silence minutes, prompt prefixes and overflow, the generation config,
silence biases, multi-outcome REACH (inline vs. regenerated), the rolling-window limits, saved outputs,
full-vocabulary features on a tiny random Llama, sharding and the dry run.

A harmless `KeyError: '/loky-…'` from Python's `multiprocessing/resource_tracker.py` may print when the
SGLang engine shuts down. The run has still finished: check the exit code and `run_summary.json`.
