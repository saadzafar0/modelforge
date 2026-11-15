# ModelForge — Deliverable 1 Checklist

Build a reproducible pipeline that takes a structured classification dataset through validation, preprocessing, baseline training, neural architecture search (NAS), evaluation, and remote experiment tracking/model registration.

ModelForge follows the **Standard ML** requirements in [the original deliverable](delieverable_1.md). The RAG, LLM, agentic, and reinforcement learning alternatives do not apply.

The checklists below contain the applicable submission requirements. The implementation notes explain how to apply them to ModelForge; suggested choices are not additional assignment requirements.

## 1. GitHub repository

- [ ] Keep the repository clean and organized.
- [ ] Include a README explaining the project, environment setup, how to run the pipeline, and the folder structure.
- [ ] Ensure every group member has commits.
- [ ] Configure `.gitignore` so raw datasets, model weights, and local generated artifacts are not committed to Git.
- [ ] Commit the source code, dependency definitions, configuration, and DVC metadata needed to reproduce the work.

## 2. Data ingestion, validation, and versioning

- [ ] Ingest a structured classification dataset from its source using a repeatable script.
- [ ] Validate the dataset before training: expected columns/schema, data types, missing values, and relevant value ranges.
- [ ] Version the dataset using DVC.
- [ ] Configure a DVC remote on DagsHub or AWS S3 and push the versioned data to it.
- [ ] Make ingestion and validation fully re-runnable.

**ModelForge implementation notes:** Start with one suitable dataset. Check that the target column exists and contains valid classification labels. Keep the initial dataset within the proposal's limits of 10,000 records and 30 input features. DagsHub is the suggested remote to keep the setup simple; AWS is not required for this deliverable.

## 3. Preprocessing

- [ ] Implement preprocessing as a repeatable pipeline step.
- [ ] Prepare the dataset for the baseline and main model, including cleaning, categorical encoding, feature scaling, and data splitting where appropriate.

**ModelForge implementation notes:** Use reproducible train/validation/test splits. Fit learned preprocessing only on the training split, then apply it to validation and test data. Save the fitted preprocessing with the selected model so predictions can be reproduced. Use validation data for NAS selection and reserve the test split for final evaluation.

## 4. Baseline and main model

- [ ] Train at least one baseline model.
- [ ] Train the main ModelForge model through repeatable scripts.
- [ ] Evaluate both models with appropriate classification metrics.

**ModelForge implementation notes:** A fixed-architecture MLP is a suggested baseline because it lets us compare a manually chosen architecture against the NAS-selected architecture. The baseline choice is still open; the assignment does not prescribe a particular algorithm.

The main model is a PyTorch MLP selected using Optuna. The proposal specifies:

- 1–4 hidden layers.
- 32, 64, 128, or 256 neurons per layer.
- ReLU, GELU, or Tanh activation.
- Dropout between 0.0 and 0.4.
- Optional residual/skip connections.
- TPE sampling, trial pruning, and early stopping.

Keep trials sequential and cap each search job at 10–15 trials, following the proposal. Define the search settings and training limits in configuration so the experiment can be repeated.

## 5. Evaluation and model selection

- [ ] Produce evaluation results for both the baseline and the main model.
- [ ] Record appropriate metrics and evaluation artifacts.

**ModelForge implementation notes:** The project proposal calls for accuracy, macro F1-score, inference latency, and model size. Compare both models on the same held-out data and measure latency under the same conditions. Define the validation-based model-selection rule explicitly; the proposal does not yet specify how to trade off predictive performance, latency, and size.

## 6. Remote MLflow tracking and model registry

- [ ] Use MLflow hosted on DagsHub or another remote server, rather than only a local tracking server.
- [ ] Log every training run, including hyperparameters, metrics, and artifacts.
- [ ] Track both the baseline and the main model's training/search runs.
- [ ] Register at least one model/checkpoint in the MLflow Model Registry with a status of Staging or Production.
- [ ] Keep evidence of at least two experiment runs for the submission.

**ModelForge implementation notes:** Log each Optuna trial's architecture and training parameters, available validation metrics, and completion/pruning/failure status. Save the selected model, preprocessing, and evaluation outputs. A parent NAS run with child trial runs is an optional organization choice.

The submission instructions specifically request a screenshot of a registered model **in Staging**, so plan to capture that state even though the general requirement allows Staging or Production. Model registration does not require deploying an API.

## 7. Reproducible DVC pipeline

- [ ] Define a `dvc.yaml` with named stages.
- [ ] Declare the stages' dependencies, parameters, and outputs so DVC can track changes.
- [ ] Ensure `dvc repro` can reproduce the complete ML pipeline from scratch, with the documented environment, data access, and remote credentials configured.
- [ ] Make the pipeline diagram available through `dvc dag`.

**Suggested ModelForge stages:**

```text
ingest → validate → preprocess → train_baseline → evaluate_baseline
                             ↘ search_and_train → evaluate_main
```

MLflow logging belongs inside the relevant training/evaluation steps. Provide a repeatable command or documented step to register the selected model; the assignment does not require registration to be a separate DVC stage.

Record dependencies, random seeds, data-split settings, and search configuration so another group member can rerun the work. Raw data and weights stay outside Git; their DVC metadata belongs in Git.

## 8. Submission PDF

- [ ] Prepare a 3–4 page PDF.
- [ ] Include the GitHub repository link.
- [ ] Include the DagsHub/remote MLflow tracking server link.
- [ ] Include screenshots showing at least two MLflow experiment runs.
- [ ] Include a screenshot of the registered model in Staging.
- [ ] Include the DVC pipeline diagram from `dvc dag`.
- [ ] Write a few sentences per component explaining the implementation: repository/setup, data validation and versioning, preprocessing, baseline and NAS model, evaluation, MLflow, and pipeline reproducibility.

## Outside Deliverable 1

Docker, Docker Compose, AWS deployment, FastAPI/REST endpoints, drift monitoring, operational retraining, deployment rollback, and the web dashboard are not required for this submission. The dashboard is a later project feature; the other deployment/monitoring components are explicitly outside the first deliverable's scope.

Document chunking, embeddings, vector databases, prompt versioning, Langfuse/token traces, agent loops, RL environments, baseline policies, and reward curves do not apply to ModelForge's tabular classification pipeline.
