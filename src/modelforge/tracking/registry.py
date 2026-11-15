from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
import torch

from modelforge.utils.common import config, configure_tracking, read_json, write_json
from modelforge.models.mlp import load_checkpoint


def register() -> None:
    mlflow = configure_tracking()
    settings = config()['tracking']
    run_id = read_json('models/nas/summary.json')['run_id']
    client = mlflow.MlflowClient()
    name = settings['model_name']
    # Reuse an existing version for this exact run on repeated pipeline execution.
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException as exc:
        if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
            raise
        versions = []
    existing = next((v for v in versions if v.run_id == run_id), None)
    version = existing or mlflow.register_model(f'runs:/{run_id}/model', name)
    transitioned = client.transition_model_version_stage(name, version.version, 'Staging',
                                                          archive_existing_versions=False)
    if transitioned.current_stage != 'Staging':
        raise RuntimeError('Registry did not confirm the required Staging state.')
    # Test the actual hosted model artifact, including raw-feature preprocessing.
    hosted = mlflow.pyfunc.load_model(f'models:/{name}/{version.version}')
    raw = pd.read_csv('data/processed/test_features.csv')
    predictions = hosted.predict(raw)['prediction'].to_numpy()
    model, _ = load_checkpoint('models/nas/checkpoint.pt')
    X = torch.from_numpy(np.load('data/processed/test.npz')['X'])
    with torch.inference_mode():
        expected = model(X).argmax(1).numpy()
    if not np.array_equal(predictions, expected):
        raise RuntimeError('Hosted model predictions differ from the evaluated local checkpoint.')
    write_json('reports/tracking/registry.json', {
        'model_name': name, 'version': str(version.version), 'stage': transitioned.current_stage,
        'run_id': run_id, 'tracking_uri': settings['uri'],
        'model_uri': f'models:/{name}/{version.version}',
        'hosted_predictions_match_local': True, 'verified_rows': len(predictions),
        'ui_url': f'{settings["uri"]}/#/models/{name}/versions/{version.version}',
    })
    print(f'Registered {name} version {version.version} in Staging; hosted prediction check passed.')
