# cocoa_inference_testing

Scripts and notebooks for running SCOPE/REACH inference on cocoa-tokenized EHR timelines using a model trained with cotorra.

The pipeline reads held-out timelines produced by cocoa's Winnower (`held_out_for_inference.parquet`), runs SCOPE and REACH trajectory generation and scoring via `quick_sco_re`, and saves per-outcome probability scores for downstream analysis.

---

## Prerequisites

- `quick_sco_re` installed (see repo root `README.md`)
- A cotorra-trained model checkpoint (SGLang-compatible)
- cocoa Winnower outputs: `held_out_for_inference.parquet` and `tokenizer.yaml`

---

## Workflow

### 1. Configure

Edit `pipeline_config.yaml` (or copy it) to point at your data and model:

```yaml
cocoa_outputs:
  held_out_for_inference: /path/to/held_out_for_inference.parquet
  tokenizer_yaml: /path/to/tokenizer.yaml

model_path: /path/to/cotorra/checkpoint

output_dir: ./scope_reach_output

generation:
  max_len: 10000
  n_samp: 100
  score_inline: true   # recommended; computes SCOPE/REACH during generation
  methods: [M1, M2]
  tracked_events:      # cocoa token names to score
    - DSCG//expired
    - XFR-IN//icu
    - RESP//imv
    # … add others from your tokenizer
  end_tokens:
    prefixes: [DSCG]   # tokens that naturally end a timeline
```

**Dry-run first** to validate paths and token resolution without launching a GPU job:

```bash
python run_timelines.py --config pipeline_config.yaml --dry-run
```

### 2. Run inference

```bash
# Locally (requires GPU)
python run_timelines.py --config pipeline_config.yaml

# On SLURM
sbatch run_inference.sh pipeline_config.yaml
```

Useful CLI overrides:

| Flag | Description |
|:--|:--|
| `--n-samp N` | Override `generation.n_samp` |
| `--max-patients N` | Override `cohort.max_patients` |
| `--output-dir PATH` | Override `output_dir` |
| `--score-inline` / `--no-score-inline` | Force inline or two-pass scoring |

### 3. Inspect outputs

```
scope_reach_output/
├── pipeline_config.yaml      # snapshot of the run config
├── patient_index.parquet     # patient_idx, subject_id, prompt_length, outcome flags
├── run_summary.json          # wall time, per-outcome event rates and AUC
├── scores_DSCG__expired.npz  # per-patient M0 / M1 (SCOPE) / M2 (REACH) arrays
├── scores_XFR-IN__icu.npz
├── …                         # one file per tracked_event
└── trajectories/
    ├── config.json           # GenerationConfig serialised
    └── trajectories.npz      # all M1 trajectories (flat compressed arrays)
```

Each `scores_*.npz` contains:

| Array | Description |
|:--|:--|
| `M0` | Mean MC estimate per patient (fraction of trajectories where event occurred) |
| `M1` | Mean SCOPE score per patient |
| `M2` | Mean REACH score per patient |
| `M0_raw` | Per-sample occurrence flags `(n_patients, n_samp)` |

### 4. Analyse results

Open `analysis.ipynb`. It reads `pipeline_config.yaml` to discover tracked events, loads the corresponding `scores_*.npz` files, and produces:

- Per-outcome AUC and Brier score with bootstrap 95% CIs
- AUC-vs-sample-size subsampling plots (efficiency comparison of MC, SCOPE, REACH)
- Token-cost efficiency tables

Set `OUTPUT_DIR` and `CONFIG_YAML` in the first cell to match your run.

---

## File reference

| File | Description |
|:--|:--|
| `run_timelines.py` | Main inference script — loads cocoa data, builds `GenerationConfig`, runs `quick_sco_re`, saves scores |
| `pipeline_config.yaml` | YAML config for the inference pipeline |
| `run_inference.sh` | SLURM job script wrapping `run_timelines.py` |
| `rescue.py` | Corrects early-terminated M1 trajectories (single-GPU) |
| `run_rescue.sh` | SLURM job script wrapping `rescue.py` |
| `run_rescue_array.sh` | SLURM array job script running `rescue_segment.py` in parallel |
| `rescue_merge.py` | Merges per-segment partial scores from array rescue into final score files |
| `rebuild_m0_raw.py` | Patches `scores_*.npz` files with `M0_raw` per-sample occurrence arrays |
| `analyze_generated_lengths.py` | Diagnostic: trajectory length statistics from `trajectories.npz` metadata |
| `analysis.ipynb` | Analysis notebook — AUC, Brier, efficiency plots for all tracked outcomes |
| `mortality.ipynb` | Exploratory notebook for hospital mortality (uses `ethos`/MIMIC-IV) |

### Output directories

| Directory | Contents |
|:--|:--|
| `scope_reach_output/` | Primary run output |
| `scope_reach_output_rescue/` | Corrected trajectories and scores from `rescue.py` |
| `scope_reach_output_final/` | Merged final scores after array rescue + `rescue_merge.py` |
| `scope_reach_output_shards/` | Per-shard intermediate outputs |
| `scope_reach_output_ucmc/` | Run on UCMC cohort |
