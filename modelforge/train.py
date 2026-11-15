from __future__ import annotations

import copy
from pathlib import Path
import time

import mlflow
from mlflow.entities import Metric
import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from modelforge.common import config, configure_tracking, data_tags, seed_everything, write_json
from modelforge.model import MLP


def log_history(run_id: str, history: list[dict]) -> None:
    metrics = []
    timestamp = int(time.time() * 1000)
    for row in history:
        metrics.extend(Metric(key, value, timestamp, row['epoch'])
                       for key, value in row.items() if key != 'epoch')
    client = mlflow.MlflowClient()
    for offset in range(0, len(metrics), 500):
        client.log_batch(run_id, metrics=metrics[offset:offset + 500])
    mlflow.log_dict({'epochs': history}, 'training_history.json')


def train_one(architecture: dict, checkpoint_path: Path, history: list[dict], trial=None) -> dict:
    cfg = config()
    seed = cfg['project']['seed']
    seed_everything(seed, cfg['project']['cpu_threads'])
    train = np.load('data/processed/train.npz')
    val = np.load('data/processed/validation.npz')
    X = torch.from_numpy(train['X'])
    y = torch.from_numpy(train['y'])
    Xv = torch.from_numpy(val['X'])
    yv = val['y']
    model = MLP(**architecture)
    settings = cfg['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    best_key = (-1.0, float('-inf'))
    best = None
    lowest_loss = float('inf')
    stale = 0
    for epoch in range(settings['max_epochs']):
        model.train()
        total_loss = 0.0
        permutation = torch.randperm(len(X))
        for batch in permutation.split(settings['batch_size']):
            optimizer.zero_grad()
            loss = criterion(model(X[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            logits = model(Xv)
            validation_loss = float(criterion(logits, torch.from_numpy(yv)))
            predictions = logits.argmax(1).numpy()
        score = float(f1_score(yv, predictions, average='macro', zero_division=0))
        row = {'epoch': epoch, 'train_loss': total_loss / len(X),
               'validation_loss': validation_loss, 'validation_macro_f1': score,
               'validation_accuracy': float(accuracy_score(yv, predictions))}
        history.append(row)
        if (score, -validation_loss) > best_key:
            best_key = (score, -validation_loss)
            best = {'state_dict': copy.deepcopy(model.state_dict()),
                    'architecture': architecture, 'best_epoch': epoch,
                    'validation_macro_f1': score, 'validation_accuracy': row['validation_accuracy'],
                    'seed': seed}
        if validation_loss < lowest_loss - settings['min_delta']:
            lowest_loss = validation_loss
            stale = 0
        else:
            stale += 1
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stale >= settings['patience']:
            break
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, checkpoint_path)
    return {key: value for key, value in best.items() if key != 'state_dict'} | {
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'epochs_trained': len(history),
    }


def baseline() -> None:
    cfg = config()
    mlflow = configure_tracking()
    architecture = {'input_size': 30, **cfg['baseline']}
    output = Path('models/baseline')
    output.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name='baseline-fixed-MLP', tags={**data_tags(), 'role': 'baseline'}) as run:
        mlflow.log_params({**architecture, **cfg['training'], 'seed': cfg['project']['seed']})
        history = []
        summary = train_one(architecture, output / 'checkpoint.pt', history)
        log_history(run.info.run_id, history)
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact('reports/validation.json', artifact_path='data')
        mlflow.log_artifact('reports/splits.json', artifact_path='data')
        mlflow.log_artifact('params.yaml')
        summary['run_id'] = run.info.run_id
        write_json(output / 'summary.json', summary)
        write_json(output / 'history.json', history)
    print(f'Baseline complete: validation macro F1={summary["validation_macro_f1"]:.4f}')


def search() -> None:
    cfg = config()
    mlflow = configure_tracking()
    output = Path('models/nas')
    output.mkdir(parents=True, exist_ok=True)
    settings = cfg['search']
    if not 1 <= settings['trials'] <= 15:
        raise ValueError('The CPU project supports at most 15 sequential NAS trials.')
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=cfg['project']['seed'], n_startup_trials=settings['startup_trials']),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=settings['startup_trials'],
                                         n_warmup_steps=settings['pruning_warmup_epochs']),
    )
    results = []
    with mlflow.start_run(run_name='main-Optuna-NAS', tags={**data_tags(), 'role': 'main'}) as parent:
        mlflow.log_params({'trials': settings['trials'], 'sampler': 'TPE', 'pruner': 'MedianPruner',
                           'selection': cfg['evaluation']['selection_rule'], 'seed': cfg['project']['seed'],
                           **cfg['training']})
        mlflow.log_dict(settings, 'search_space.json')

        def objective(trial):
            layers = trial.suggest_int('hidden_layers', *settings['hidden_layers'])
            architecture = {
                'input_size': 30,
                'hidden_sizes': [trial.suggest_categorical(f'width_{i}', settings['widths']) for i in range(layers)],
                'activation': trial.suggest_categorical('activation', settings['activations']),
                'dropout': trial.suggest_float('dropout', *settings['dropout']),
                'residual': trial.suggest_categorical('residual', settings['residual']),
            }
            history = []
            path = output / 'trials' / f'{trial.number:02d}.pt'
            pruned = False
            with mlflow.start_run(run_name=f'trial-{trial.number:02d}', nested=True,
                                  tags={**data_tags(), 'role': 'nas-trial'}) as child:
                mlflow.log_params({**trial.params, 'seed': cfg['project']['seed'], **cfg['training']})
                try:
                    result = train_one(architecture, path, history, trial)
                except optuna.TrialPruned:
                    pruned = True
                    result = {'validation_macro_f1': history[-1]['validation_macro_f1'],
                              'epochs_trained': len(history)}
                log_history(child.info.run_id, history)
                mlflow.set_tag('trial_state', 'PRUNED' if pruned else 'COMPLETE')
                mlflow.log_dict(architecture, 'architecture.json')
                if not pruned:
                    mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
                record = {'trial': trial.number, 'state': 'PRUNED' if pruned else 'COMPLETE',
                          'run_id': child.info.run_id, **result}
                results.append(record)
            print(f'Trial {trial.number}: {record["state"]}, validation macro F1={result["validation_macro_f1"]:.4f}', flush=True)
            if pruned:
                raise optuna.TrialPruned()
            return result['validation_macro_f1']

        study.optimize(objective, n_trials=settings['trials'], n_jobs=1)
        completed = [r for r in results if r['state'] == 'COMPLETE']
        selected = min(completed, key=lambda r: (-r['validation_macro_f1'], r['parameter_count'], r['trial']))
        (output / 'checkpoint.pt').write_bytes((output / 'trials' / f'{selected["trial"]:02d}.pt').read_bytes())
        summary = {**selected, 'trial_run_id': selected['run_id'], 'run_id': parent.info.run_id,
                   'trials_completed': len(completed), 'trials_pruned': len(results) - len(completed),
                   'selection_rule': cfg['evaluation']['selection_rule']}
        write_json(output / 'summary.json', summary)
        write_json(output / 'trials.json', results)
        mlflow.log_metrics({'selected_validation_macro_f1': selected['validation_macro_f1'],
                            'selected_parameter_count': selected['parameter_count'],
                            'trials_completed': len(completed), 'trials_pruned': len(results) - len(completed)})
        mlflow.log_param('selected_trial', selected['trial'])
        mlflow.log_dict(selected['architecture'], 'selected_architecture.json')
        mlflow.log_artifact(str(output / 'trials.json'))
        mlflow.log_artifact('reports/validation.json', artifact_path='data')
        mlflow.log_artifact('reports/splits.json', artifact_path='data')
        mlflow.log_artifact('params.yaml')
    print(f'Selected trial {selected["trial"]}; held-out test data were not used for search.')
