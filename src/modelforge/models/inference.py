from __future__ import annotations

import json

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from modelforge.models.mlp import load_checkpoint


class TabularMLP(mlflow.pyfunc.PythonModel):
    """Accept the original 30 feature columns, then preprocess and predict."""
    def load_context(self, context):
        torch.set_num_threads(1)
        self.model, _ = load_checkpoint(context.artifacts['checkpoint'])
        self.preprocessor = joblib.load(context.artifacts['preprocessor'])
        self.features = json.loads(open(context.artifacts['features']).read())

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing feature columns: {sorted(missing)}')
        X = self.preprocessor.transform(frame[self.features]).astype(np.float32)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(X)), dim=1).numpy()
        return pd.DataFrame({'prediction': probabilities.argmax(axis=1),
                             'probability_malignant': probabilities[:, 1]})
