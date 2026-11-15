from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import yaml


def config() -> dict:
    return yaml.safe_load(Path('params.yaml').read_text())


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed: int, threads: int = 1) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def configure_tracking():
    """Require remote tracking; never silently fall back to a local MLflow store."""
    import mlflow
    cfg = config()['tracking']
    if not cfg['uri'].startswith('https://'):
        raise ValueError('Deliverable 1 requires an HTTPS remote MLflow server.')
    if not os.environ.get('MLFLOW_TRACKING_PASSWORD'):
        local = configparser.ConfigParser()
        local.read('.dvc/config.local')
        token = local.get('remote "origin"', 'secret_access_key', fallback=None)
        if token:
            os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    if not os.environ.get('MLFLOW_TRACKING_PASSWORD'):
        raise RuntimeError('Set MLFLOW_TRACKING_PASSWORD or local DVC S3 credentials.')
    os.environ.setdefault('MLFLOW_TRACKING_USERNAME', cfg['username'])
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '60')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '2')
    mlflow.set_tracking_uri(cfg['uri'])
    mlflow.set_registry_uri(cfg['uri'])
    mlflow.set_experiment(cfg['experiment'])
    return mlflow


def data_tags() -> dict:
    return {
        'dataset': 'UCI-WDBC-17',
        'data_sha256': sha256('data/raw/dataset.csv'),
        'params_sha256': sha256('params.yaml'),
        'purpose': 'course-deliverable-1',
    }
