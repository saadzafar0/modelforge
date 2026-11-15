# Pipeline reports

| Folder | Generated contents |
| --- | --- |
| `data/` | Schema/range checks and stratified split counts. |
| `metrics/` | Accuracy, macro F1, latency, model size, and baseline/main comparison. |
| `classification/` | Per-class precision, recall, and F1. |
| `figures/` | Confusion matrices. |
| `tracking/` | Remote registry version, Staging state, and prediction verification. |
| `verification/` | Reproduction verification evidence when run. |

These reports contain summaries rather than datasets or model weights. DVC declares the pipeline-generated reports as non-cached outputs/metrics so they can be reviewed in Git.
