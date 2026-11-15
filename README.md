# ModelForge

ModelForge searches for small neural-network architectures for tabular classification. Deliverable 1 provides a reproducible CPU pipeline from dataset ingestion to evaluation and remote model registration. The later platform will add a dashboard, API, deployment, and monitoring.

- [GitHub repository](https://github.com/saadzafar0/modelforge)
- [DagsHub repository and data storage](https://dagshub.com/saadzafar0/modelforge)
- [Hosted MLflow experiments and model registry](https://dagshub.com/saadzafar0/modelforge.mlflow)
- [Deliverable checklist](docs/deliverable_1_modelforge.md)

## Dataset and experiment

This milestone uses [UCI's Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): 569 records, 30 numeric predictors, and a binary target. Record IDs are excluded from the model. Labels are encoded as benign=0 and malignant=1.

The source archive is downloaded directly and checked against a pinned SHA-256 hash. Validation checks the ordered schema, record/feature limits, missing values, unique IDs, numeric types, finite/non-negative feature values, a broad safety upper bound, and the B/M label vocabulary. This is a benchmark demonstration of the MLOps platform.

Stratified train/validation/test splits use seed 42 and proportions 60/20/20. Median imputation and standardization are fitted on training data only. No categorical predictor encoding is needed for this numerical dataset. The fitted preprocessing, feature order, labels, and splits are saved.

The baseline is a fixed 64-unit ReLU MLP. The main model uses ten sequential Optuna TPE trials, median pruning, and validation-loss early stopping. The search covers 1–4 hidden layers, widths 32/64/128/256, ReLU/GELU/Tanh, dropout 0–0.4, and optional residual projections. Training uses a maximum of 100 epochs, patience 12, Adam, and one CPU thread.

The selected architecture maximizes **validation macro F1**, breaking ties by fewer parameters and then trial number. The held-out test set is used only for final evaluation, never architecture selection. Accuracy, macro F1, parameter count, checkpoint bytes, and batch-one CPU inference latency are reported. Latency excludes preprocessing and network overhead and varies with hardware/load. A NAS result is reported honestly even if it does not beat the baseline.

## Setup

Use Python 3.12 on Linux x86-64. The dependency lock includes CPU-only PyTorch and does not require CUDA or an AWS account.

```bash
git clone https://github.com/saadzafar0/modelforge.git
cd modelforge
python3.12 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements.txt
```

If you use `uv`, Python 3.12 and the environment can also be created with `uv python install 3.12` and `uv venv .venv-ml --python 3.12`. `requirements.in` records direct dependencies; `requirements.txt` pins transitive versions. The CPU PyTorch index is included in both files. Browser/PDF dependencies support submission evidence generation; no browser is required for model training.

## Authentication

The default DVC remote is already declared in `.dvc/config`. Each contributor configures their own DagsHub token locally. Run in Bash with the environment active:

```bash
read -rsp "DagsHub token: " MODELFORGE_DVC_TOKEN
printf '\n'
dvc remote modify --local origin access_key_id "$MODELFORGE_DVC_TOKEN"
dvc remote modify --local origin secret_access_key "$MODELFORGE_DVC_TOKEN"
unset MODELFORGE_DVC_TOKEN
```

The token is stored in Git-ignored `.dvc/config.local`. Training reads it to authenticate remote MLflow, or you may provide `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` through your environment. No token is committed or included in the PDF or screenshots. Remote tracking failures stop the pipeline rather than silently creating local runs.

If DVC reports a permission error opening its system state cache, set a project-local ignored cache:

```bash
dvc config --local core.site_cache_dir "$PWD/.dvc/tmp/site-cache"
```

See [DagsHub setup details](docs/dagshub_setup.md). Private repositories require contributor access for MLflow and appropriate data access for DVC.

## Run and reproduce

From the repository root, with `.venv-ml` active:

```bash
dvc repro
dvc push
dvc dag
```

`dvc repro` runs ingestion, validation, preprocessing, baseline training, NAS, both evaluations, comparison, and registration. On a first uncached run, it needs internet access to UCI and DagsHub. Data, preprocessing, and checkpoints are DVC-tracked directory outputs; `dvc.yaml` and `dvc.lock` record their versions. Raw data and weights are excluded from Git.

For an existing data/model version:

```bash
dvc pull
dvc repro
```

To rerun every stage from source instead of restoring cached computation:

```bash
dvc repro --force --no-run-cache
```

Fresh training creates fresh remote run IDs. With fixed seeds, software, and CPU settings, the data split and predictive computation are repeatable; run IDs, timestamps, serialization metadata, and wall-clock latency are not promised to be byte-identical across executions or machines.

All training runs log parameters, epoch metrics, and artifacts to the hosted MLflow experiment. The main run contains child runs for each trial; pruned trials are labeled. Evaluation logs a loadable MLflow Python model containing the selected weights and preprocessing. Registration sets the main model to **Staging**, as the assignment requests, and verifies hosted-model predictions against the local checkpoint. Registration reuses the existing model version for the same run ID.

MLflow stages are deprecated upstream, but the pinned client and DagsHub server retain them for this submission's required Staging screenshot. No deployment or production promotion is performed.

## Verification

```bash
python -m pytest -q
dvc status
dvc status --cloud
```

Tests cover invalid-input rejection, checkpoint round trips with residual projections, non-overlapping real data splits, and training-only scaler statistics. The real-split integration check requires preprocessing outputs (`dvc repro preprocess` or `dvc pull`). Model registration additionally tests predictions from the actual remotely stored MLflow artifact.

## Outputs and submission

- `reports/validation.json`, `reports/splits.json`: data checks and split counts.
- `reports/baseline_metrics.json`, `reports/nas_metrics.json`, `reports/comparison.json`: final evaluation.
- `reports/registry.json`: remote model version, Staging state, and verified model URI.
- `screenshots/deleiverable_1/`: actual terminal and hosted-MLflow evidence.
- `submission/`: the 3–4 page deliverable PDF and supporting write-up.

Evidence and the PDF can be refreshed after a completed pipeline:

```bash
PLAYWRIGHT_BROWSERS_PATH=.browsers python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=.browsers python scripts/capture_evidence.py
python scripts/build_submission.py
```

The capture script signs in to the real hosted MLflow UI using local credentials and captures it. It does not generate imitation experiment/registry screens. Graphviz (`dot`) is used for the DVC diagram. Browser downloads are ignored by Git.

## Structure

```text
modelforge/       ingestion, validation, preprocessing, MLP, NAS, evaluation, registry
tests/            data integrity, leakage, and model serialization checks
scripts/          evidence capture and submission PDF generation
docs/             proposal, assignment, setup guide, and checklist
data/             DVC-managed raw and processed data (ignored by Git)
models/           DVC-managed checkpoints and search outputs (ignored by Git)
reports/          validation, evaluation, and registration results
screenshots/      actual submission evidence
submission/       final PDF and write-up
dvc.yaml          named reproducible stages
dvc.lock          data, dependency, and parameter versions
params.yaml       dataset, split, training, search, and tracking configuration
requirements.txt  pinned Python dependency lock
```

## Dataset attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Team supplied for the report

- Sarem Waheed — 23l-0885
- Mamoon Chishti — 23l-6050

Docker, REST APIs, AWS deployment, monitoring, and the dashboard are later milestones.
