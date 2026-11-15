# Deliverable 1 evidence

This folder contains real setup evidence for ModelForge. No credentials are displayed in the captures.

## Current evidence

| File | What it demonstrates |
| --- | --- |
| `01_dvc_remote_setup.png` | Installed DVC version, default S3-compatible remote, non-secret endpoint configuration, and Git ignoring the local credentials file. |
| `02_dvc_remote_roundtrip.png` | A real DVC upload/download round trip in a temporary test project, byte-for-byte integrity verification, and removal of the unique remote test object. |
| `dvc_remote_verification.txt` | The original command transcript underlying the round-trip screenshot. |
| `03_mlflow_baseline_run.png` | Actual hosted baseline run, including held-out metrics and its finished state. |
| `04_mlflow_nas_run.png` | Actual hosted main NAS run and final evaluation metrics. |
| `05_registered_model_staging.png` | Actual ModelForge-WDBC-MLP version 1 page showing Staging. |
| `06_mlflow_experiment_overview.png` | The real hosted experiment overview. |
| `07_dvc_pipeline.png` | Graphviz rendering of the real `dvc dag --dot` output. |
| `08_dvc_dag_terminal.png` | Screenshot of the actual `dvc dag` terminal output. |
| `capture_manifest.json` | Source URLs, run IDs, and commands underlying the evidence. |
| `dvc_dag.dot`, `dvc_dag.txt` | Actual DVC diagram outputs, with trailing whitespace removed. |

Images 01, 02, and 08 capture a dedicated terminal window. The round-trip terminal displays the saved transcript of the actual verification run. The temporary connection-check file is not the project dataset. Images 03–06 capture the actual remote MLflow interface, with UI onboarding dismissed. Image 07 renders DVC's actual dependency graph.

Suggested captions:

1. **DVC remote configuration:** ModelForge uses DagsHub's S3-compatible DVC remote; credentials remain in a Git-ignored local configuration.
2. **Storage verification:** A temporary file was uploaded, downloaded from an empty local cache, and checked for an exact content match. The remote test object was then removed.

## Final PDF

The four-page submission is at [`submission/deliverable_1/ModelForge_Deliverable_1.pdf`](../../submission/deliverable_1/ModelForge_Deliverable_1.pdf). It includes images 03, 04, 05, and 07: the required two experiment runs, registered model in Staging, and DVC pipeline diagram. The actual terminal diagram is also retained as image 08.

The real dataset, splits, preprocessing, and model outputs have been pushed to the configured DagsHub DVC remote. A separate empty-cache checkout restored all outputs and reproduced an up-to-date pipeline; the verification record is in `reports/verification/`.
