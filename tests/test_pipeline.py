import numpy as np
import pandas as pd
import pytest
import torch

from modelforge.data.pipeline import FEATURES, validate_frame
from modelforge.models.mlp import MLP, load_checkpoint


@pytest.fixture
def frame():
    values = np.ones((6, 30))
    result = pd.DataFrame(values, columns=FEATURES)
    result.insert(0, 'diagnosis', ['B', 'M'] * 3)
    result.insert(0, 'id', [str(i) for i in range(6)])
    return result


@pytest.mark.parametrize('corruption', ['missing_column', 'null', 'negative', 'infinite', 'label', 'duplicate_id', 'text'])
def test_validation_rejects_bad_input(frame, corruption):
    cfg = {'expected_rows': 6, 'max_rows': 10000, 'max_features': 30}
    if corruption == 'missing_column':
        frame = frame.drop(columns=FEATURES[0])
    elif corruption == 'null':
        frame.loc[0, FEATURES[0]] = np.nan
    elif corruption == 'negative':
        frame.loc[0, FEATURES[0]] = -1
    elif corruption == 'infinite':
        frame.loc[0, FEATURES[0]] = np.inf
    elif corruption == 'label':
        frame.loc[0, 'diagnosis'] = 'unknown'
    elif corruption == 'duplicate_id':
        frame.loc[1, 'id'] = frame.loc[0, 'id']
    else:
        frame[FEATURES[0]] = 'text'
    with pytest.raises(ValueError):
        validate_frame(frame, cfg)


@pytest.mark.parametrize('residual', [False, True])
def test_checkpoint_roundtrip_and_variable_widths(tmp_path, residual):
    architecture = {'input_size': 30, 'hidden_sizes': [32, 64, 32],
                    'activation': 'GELU', 'dropout': 0.2, 'residual': residual}
    model = MLP(**architecture).eval()
    X = torch.randn(5, 30)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'architecture': architecture, 'state_dict': model.state_dict()}, path)
    restored, _ = load_checkpoint(path)
    with torch.inference_mode():
        torch.testing.assert_close(restored(X), model(X), rtol=0, atol=0)
    assert restored(X).shape == (5, 2)


def test_real_split_integrity_and_train_only_scaling():
    """Integration check after preprocess: detect split overlap and preprocessing leakage."""
    from pathlib import Path
    import joblib
    if not Path('data/processed/train.npz').exists():
        pytest.skip('Run dvc repro preprocess to generate the integration fixture.')
    splits = {name: np.load(f'data/processed/{name}.npz') for name in ('train', 'validation', 'test')}
    groups = [set(value['ids']) for value in splits.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert len(set.union(*groups)) == 569
    raw = pd.read_csv('data/raw/dataset.csv', dtype={'id': str}).set_index('id')
    train_rows = raw.loc[splits['train']['ids'], FEATURES]
    preprocessor = joblib.load('data/processed/preprocessor.joblib')
    np.testing.assert_allclose(preprocessor.named_steps['scaler'].mean_, train_rows.mean().to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(splits['train']['X'].mean(0), 0, atol=2e-6)
