# ModelForge — Deliverable 1

Team: Sarem Waheed (23l-0885); Mamoon Chishti (23l-6050).

GitHub: https://github.com/saadzafar0/modelforge
DagsHub: https://dagshub.com/saadzafar0/modelforge
MLflow: https://dagshub.com/saadzafar0/modelforge.mlflow

The four-page PDF includes implementation notes, real MLflow screenshots for the baseline and main NAS runs,
the model registered in Staging, and the DVC dependency graph. Data is the UCI Wisconsin Diagnostic benchmark,
with 569 records and 30 numeric predictors. Splits are stratified 60/20/20; preprocessing is fitted on training data only.

| Held-out metric | Baseline | NAS |
| --- | --- | --- |
| Accuracy | 0.991228 | 0.982456 |
| Macro F1 | 0.990528 | 0.980956 |
| Parameters | 2114 | 2018 |

Registered model: ModelForge-WDBC-MLP, version 1, stage Staging.
Hosted predictions matched the local checkpoint on 114 held-out records.

Reproduction: see README.md. Source dataset attribution: https://doi.org/10.24432/C5DW2B (CC BY 4.0).
