from __future__ import annotations

from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch

from modelforge.common import config, configure_tracking, read_json, seed_everything, write_json
from modelforge.inference import TabularMLP
from modelforge.model import load_checkpoint


def evaluate(role: str) -> None:
    cfg = config()
    seed_everything(cfg['project']['seed'], cfg['project']['cpu_threads'])
    folder = Path('models') / role
    model, checkpoint = load_checkpoint(folder / 'checkpoint.pt')
    test = np.load('data/processed/test.npz')
    X = torch.from_numpy(test['X'])
    with torch.inference_mode():
        probabilities = torch.softmax(model(X), dim=1).numpy()
    predictions = probabilities.argmax(1)
    sample = X[:1]
    durations = []
    with torch.inference_mode():
        for _ in range(cfg['evaluation']['latency_warmups']):
            model(sample)
        for _ in range(cfg['evaluation']['latency_repeats']):
            start = time.perf_counter_ns()
            model(sample)
            durations.append((time.perf_counter_ns() - start) / 1_000_000)
    metrics = {
        'test_accuracy': float(accuracy_score(test['y'], predictions)),
        'test_macro_f1': float(f1_score(test['y'], predictions, average='macro', zero_division=0)),
        'inference_latency_median_ms': float(np.median(durations)),
        'inference_latency_p95_ms': float(np.percentile(durations, 95)),
        'checkpoint_size_bytes': (folder / 'checkpoint.pt').stat().st_size,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'test_rows': len(test['y']),
    }
    write_json(f'reports/{role}_metrics.json', metrics)
    report = classification_report(test['y'], predictions, labels=[0, 1],
                                    target_names=['benign', 'malignant'], output_dict=True, zero_division=0)
    write_json(f'reports/{role}_classification.json', report)
    cm = confusion_matrix(test['y'], predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Benign', 'Malignant'],
           yticklabels=['Benign', 'Malignant'], xlabel='Predicted', ylabel='Actual', title=f'{role}: held-out test')
    fig.tight_layout()
    fig.savefig(f'reports/{role}_confusion_matrix.png', dpi=160)
    plt.close(fig)
    summary = read_json(folder / 'summary.json')
    mlflow = configure_tracking()
    original = pd.read_csv('data/processed/test_features.csv').head(3)
    example_output = pd.DataFrame({'prediction': predictions[:3],
                                   'probability_malignant': probabilities[:3, 1]})
    with mlflow.start_run(run_id=summary['run_id']):
        mlflow.log_metrics(metrics)
        for suffix in ('metrics.json', 'classification.json', 'confusion_matrix.png'):
            mlflow.log_artifact(f'reports/{role}_{suffix}', artifact_path='evaluation')
        mlflow.set_tag('latency_scope', 'CPU model forward pass; batch=1; excludes preprocessing and network')
        mlflow.pyfunc.log_model(
            artifact_path='model', python_model=TabularMLP(),
            artifacts={'checkpoint': str(folder / 'checkpoint.pt'),
                       'preprocessor': 'data/processed/preprocessor.joblib',
                       'features': 'data/processed/features.json'},
            code_paths=['modelforge'], pip_requirements='requirements.txt',
            signature=infer_signature(original, example_output), input_example=original,
        )
    print(f'{role}: accuracy={metrics["test_accuracy"]:.4f}, macro F1={metrics["test_macro_f1"]:.4f}')


def compare() -> None:
    report = {
        'baseline': read_json('reports/baseline_metrics.json'),
        'nas': read_json('reports/nas_metrics.json'),
        'selection_used_test_data': False,
        'selection_rule': config()['evaluation']['selection_rule'],
        'latency_scope': 'CPU model forward pass, batch=1; preprocessing and network excluded',
        'latency_note': 'Wall-clock measurements vary with machine load; predictive results use fixed seeds.',
    }
    write_json('reports/comparison.json', report)
    print('Saved baseline/main comparison on the same held-out test split.')
