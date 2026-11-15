from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelforge.utils.common import config, sha256, write_json

MEASUREMENTS = ('radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
                'concavity', 'concave_points', 'symmetry', 'fractal_dimension')
FEATURES = [f'{stat}_{name}' for stat in ('mean', 'se', 'worst') for name in MEASUREMENTS]
COLUMNS = ['id', 'diagnosis', *FEATURES]


def ingest() -> None:
    cfg = config()['dataset']
    response = requests.get(cfg['url'], timeout=(15, 60))
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != cfg['sha256']:
        raise ValueError('Source archive checksum differs from the pinned dataset version.')
    output = Path('data/raw')
    output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    source = archive.read('wdbc.data')
    (output / 'wdbc.data').write_bytes(source)
    (output / 'wdbc.names').write_bytes(archive.read('wdbc.names'))
    frame = pd.read_csv(io.BytesIO(source), header=None, names=COLUMNS, dtype={'id': str})
    frame.to_csv(output / 'dataset.csv', index=False)
    write_json(output / 'source.json', {
        'dataset': cfg['name'], 'url': cfg['url'], 'archive_sha256': digest,
        'source_file_sha256': hashlib.sha256(source).hexdigest(),
        'csv_sha256': sha256(output / 'dataset.csv'), 'license': 'CC BY 4.0',
        'citation': 'Wolberg et al. (1993). UCI. https://doi.org/10.24432/C5DW2B',
    })
    print(f'Ingested {len(frame)} rows and {len(FEATURES)} features from the pinned UCI archive.')


def validate_frame(frame: pd.DataFrame, cfg: dict) -> dict:
    if list(frame.columns) != COLUMNS:
        raise ValueError('Schema mismatch: expected ID, diagnosis, and 30 ordered numeric features.')
    if len(frame) != cfg['expected_rows'] or len(frame) > cfg['max_rows']:
        raise ValueError('Unexpected row count or project record limit exceeded.')
    if len(FEATURES) > cfg['max_features']:
        raise ValueError('Feature limit exceeded.')
    if frame.isna().any().any():
        raise ValueError('Missing values detected in this complete-source dataset.')
    if not frame['id'].is_unique:
        raise ValueError('Duplicate record IDs detected.')
    if set(frame['diagnosis']) != {'B', 'M'}:
        raise ValueError('Invalid classification labels; expected B and M.')
    if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in FEATURES):
        raise ValueError('Non-numeric feature detected.')
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1_000_000).any():
        raise ValueError('Non-finite or out-of-range feature: expected 0 <= value <= 1,000,000.')
    return {
        'passed': True, 'rows': len(frame), 'features': len(FEATURES), 'missing_values': 0,
        'duplicate_ids': 0, 'class_counts': frame['diagnosis'].value_counts().to_dict(),
        'checks': ['ordered_schema', 'row_count', 'feature_count', 'nulls', 'unique_ids',
                   'target_labels', 'numeric_types', 'finite_values', 'feature_ranges'],
        'feature_ranges': {col: {'min': float(frame[col].min()), 'max': float(frame[col].max())}
                           for col in FEATURES},
    }


def validate() -> None:
    frame = pd.read_csv('data/raw/dataset.csv', dtype={'id': str})
    report = validate_frame(frame, config()['dataset'])
    report['csv_sha256'] = sha256('data/raw/dataset.csv')
    write_json('reports/data/validation.json', report)
    print('Validation passed: schema, nulls, labels, IDs, numeric types, and ranges.')


def preprocess() -> None:
    cfg = config()
    frame = pd.read_csv('data/raw/dataset.csv', dtype={'id': str})
    validate_frame(frame, cfg['dataset'])
    labels = frame['diagnosis'].map({'B': 0, 'M': 1}).to_numpy(dtype=np.int64)
    rows = np.arange(len(frame))
    seed = cfg['project']['seed']
    split = cfg['split']
    train_val, test = train_test_split(rows, test_size=split['test_fraction'],
                                      random_state=seed, stratify=labels)
    train, val = train_test_split(
        train_val, test_size=split['validation_fraction'] / (1 - split['test_fraction']),
        random_state=seed, stratify=labels[train_val])
    transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
    transformer.fit(frame.iloc[train][FEATURES])
    output = Path('data/processed')
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, output / 'preprocessor.joblib')
    write_json(output / 'features.json', FEATURES)
    report = {'seed': seed, 'label_mapping': {'B': 0, 'M': 1}, 'splits': {},
              'preprocessing_fit_split': 'train', 'id_excluded_from_features': True}
    for name, indexes in (('train', train), ('validation', val), ('test', test)):
        X = transformer.transform(frame.iloc[indexes][FEATURES]).astype(np.float32)
        np.savez_compressed(output / f'{name}.npz', X=X, y=labels[indexes],
                            ids=frame.iloc[indexes]['id'].to_numpy(dtype=str))
        report['splits'][name] = {
            'rows': len(indexes),
            'class_counts': {str(c): int((labels[indexes] == c).sum()) for c in (0, 1)},
        }
    frame.iloc[test][FEATURES].to_csv(output / 'test_features.csv', index=False)
    write_json('reports/data/splits.json', report)
    print('Saved stratified 60/20/20 splits and preprocessing fitted on training data only.')
