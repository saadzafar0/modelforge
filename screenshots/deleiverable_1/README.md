# Deliverable 1 evidence

This folder contains real setup evidence for ModelForge. No credentials are displayed in the captures.

## Current evidence

| File | What it demonstrates |
| --- | --- |
| `01_dvc_remote_setup.png` | Installed DVC version, default S3-compatible remote, non-secret endpoint configuration, and Git ignoring the local credentials file. |
| `02_dvc_remote_roundtrip.png` | A real DVC upload/download round trip in a temporary test project, byte-for-byte integrity verification, and removal of the unique remote test object. |
| `dvc_remote_verification.txt` | The original command transcript underlying the round-trip screenshot. |

The PNGs capture a dedicated terminal window. The round-trip terminal displays the saved transcript of the actual verification run. The temporary connection-check file is not the project dataset.

Suggested captions:

1. **DVC remote configuration:** ModelForge uses DagsHub's S3-compatible DVC remote; credentials remain in a Git-ignored local configuration.
2. **Storage verification:** A temporary file was uploaded, downloaded from an empty local cache, and checked for an exact content match. The remote test object was then removed.

## Capture when the corresponding work is complete

These images have not been created because the corresponding project components do not exist yet:

- Actual dataset versioning and successful `dvc push` (useful supporting evidence).
- MLflow baseline run with parameters, metrics, and artifacts.
- MLflow main/NAS run with parameters, metrics, and artifacts.
- Registered model showing the Staging state required by the submission instructions.
- The real project pipeline diagram from `dvc dag` after `dvc.yaml` is implemented.

The assignment explicitly requires at least two MLflow experiment runs, the registered model in Staging, and the DVC pipeline diagram. The setup screenshots supplement those required images; they do not replace them.
