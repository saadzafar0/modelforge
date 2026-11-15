"""Restore the full pipeline in an isolated checkout and check deterministic training."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

from modelforge.utils.common import config, sha256, write_json


def main():
    import numpy as np
    import torch
    from modelforge.models.mlp import load_checkpoint
    from modelforge.training.runner import train_one

    transcript = []
    with tempfile.TemporaryDirectory(prefix='modelforge-reproduce-') as temporary:
        work = Path(temporary)
        for folder in ('src', 'reports'):
            shutil.copytree(ROOT / folder, work / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '.work'))
        for filename in ('dvc.yaml', 'dvc.lock', 'params.yaml', 'requirements.txt', 'pyproject.toml', '.gitignore', '.dvcignore'):
            shutil.copyfile(ROOT / filename, work / filename)
        (work / '.dvc').mkdir()
        for filename in ('config', '.gitignore'):
            shutil.copyfile(ROOT / '.dvc' / filename, work / '.dvc' / filename)
        local = configparser.ConfigParser()
        local.read(ROOT / '.dvc/config.local')
        if not local.has_section('core'):
            local.add_section('core')
        local['core']['site_cache_dir'] = str(work / '.dvc/tmp/site-cache')
        with (work / '.dvc/config.local').open('w') as stream:
            local.write(stream)
        env = dict(os.environ, DVC_NO_ANALYTICS='1', PYTHONPATH=str(work / 'src'),
                   PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])

        def run(args):
            result = subprocess.run(args, cwd=work, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            label = ' '.join(args).replace(sys.executable, 'python')
            transcript.append('$ ' + label + '\n' + result.stdout.strip())
            print(label, flush=True)
            print(result.stdout.strip(), flush=True)
            if result.returncode:
                raise RuntimeError(f'Clean-checkout verification failed: {label}')
            return result.stdout

        run(['git', 'init', '-q'])
        run(['git', 'add', '.'])
        if (work / 'data').exists() or (work / 'models').exists() or (work / '.dvc/cache').exists():
            raise RuntimeError('Verification must begin without local data, weights, or DVC cache.')
        run([sys.executable, '-m', 'dvc', 'pull'])
        run([sys.executable, '-m', 'dvc', 'repro'])
        state = json.loads(run([sys.executable, '-m', 'dvc', 'status', '--json']))
        if state:
            raise RuntimeError('The restored pipeline is not up to date.')
        for relative in ('data/raw/dataset.csv', 'data/processed/train.npz',
                         'data/processed/test.npz', 'models/nas/checkpoint.pt'):
            if sha256(ROOT / relative) != sha256(work / relative):
                raise RuntimeError(f'Restored content differs: {relative}')
    with tempfile.TemporaryDirectory(prefix='modelforge-determinism-') as temporary:
        path = Path(temporary) / 'baseline.pt'
        architecture = {'input_size': 30, **config()['baseline']}
        history = []
        train_one(architecture, path, history)
        repeated, _ = load_checkpoint(path)
        original, _ = load_checkpoint('models/baseline/checkpoint.pt')
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[key], rtol=0, atol=0)
        X = torch.from_numpy(np.load('data/processed/test.npz')['X'])
        with torch.inference_mode():
            torch.testing.assert_close(original(X), repeated(X), rtol=0, atol=0)
    write_json('reports/verification/reproduction.json', {
        'clean_checkout_dvc_pull': True, 'restored_pipeline_up_to_date': True,
        'raw_data_and_checkpoint_hashes_match': True,
        'baseline_retraining_weights_match_exactly': True,
        'baseline_retraining_test_logits_match_exactly': True,
        'dvc_lock_sha256': sha256('dvc.lock'), 'requirements_sha256': sha256('requirements.txt'),
    })
    Path('reports/verification/reproduction_transcript.txt').write_text('\n\n'.join(transcript) + '\n')
    print('PASS: Empty-cache restoration and deterministic baseline retraining verified.')


if __name__ == '__main__':
    main()
