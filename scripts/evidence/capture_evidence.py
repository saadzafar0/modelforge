"""Capture the real hosted MLflow UI and a diagram produced by dvc dag."""
from __future__ import annotations

import configparser
import base64
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

from modelforge.utils.common import config, read_json, write_json


def main():
    from playwright.sync_api import sync_playwright
    import mlflow

    output = ROOT / 'screenshots/deleiverable_1'
    output.mkdir(parents=True, exist_ok=True)
    settings = config()['tracking']
    registry = read_json('reports/tracking/registry.json')
    local = configparser.ConfigParser()
    local.read('.dvc/config.local')
    token = os.environ.get('MLFLOW_TRACKING_PASSWORD') or local.get('remote "origin"', 'secret_access_key')
    username = os.environ.get('MLFLOW_TRACKING_USERNAME', settings['username'])
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    client = mlflow.MlflowClient(tracking_uri=settings['uri'])
    experiment = client.get_experiment_by_name(settings['experiment'])
    base_run = read_json('models/baseline/summary.json')['run_id']
    main_run = read_json('models/nas/summary.json')['run_id']
    items = [
        ('03_mlflow_baseline_run.png', f'{settings["uri"]}/#/experiments/{experiment.experiment_id}/runs/{base_run}', 'baseline-fixed-MLP'),
        ('04_mlflow_nas_run.png', f'{settings["uri"]}/#/experiments/{experiment.experiment_id}/runs/{main_run}', 'main-Optuna-NAS'),
        ('05_registered_model_staging.png', registry['ui_url'], 'Staging'),
    ]
    manifest = {'tracking_uri': settings['uri'], 'experiment_id': experiment.experiment_id,
                'baseline_run_id': base_run, 'main_run_id': main_run, 'captures': []}
    with sync_playwright() as p:
        cdp_url = os.environ.get('MODELFORGE_CDP_URL')
        browser = (p.chromium.connect_over_cdp(cdp_url) if cdp_url
                   else p.chromium.launch(headless=True, args=['--no-sandbox']))
        context = (browser.contexts[0] if cdp_url else browser.new_context(
            viewport={'width': 1500, 'height': 1000}, device_scale_factor=1,
            http_credentials={'username': username, 'password': token, 'origin': 'https://dagshub.com'}))
        # Send credentials only to DagsHub, never to third-party assets/analytics.
        authorization = 'Basic ' + base64.b64encode(f'{username}:{token}'.encode()).decode()
        context.route('https://dagshub.com/**', lambda route: route.continue_(
            headers={**route.request.headers, 'authorization': authorization}))
        page = context.pages[0] if cdp_url and context.pages else context.new_page()
        page.set_viewport_size({'width': 1500, 'height': 1000})
        for filename, url, expected in items:
            page.goto(url, wait_until='domcontentloaded', timeout=90000)
            page.wait_for_function('(value) => document.body.innerText.includes(value)',
                                   arg=expected, timeout=90000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(output / filename), full_page=False)
            manifest['captures'].append({'file': filename, 'source_url': url,
                                         'expected_visible_text': expected})
            print('Captured real hosted UI:', filename, flush=True)
        # Also retain an overview showing the baseline, parent NAS run, and trial hierarchy.
        overview = f'{settings["uri"]}/#/experiments/{experiment.experiment_id}'
        page.goto(overview, wait_until='domcontentloaded', timeout=90000)
        page.wait_for_function("() => document.body.innerText.includes('baseline-fixed-MLP')", timeout=90000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(output / '06_mlflow_experiment_overview.png'), full_page=False)
        manifest['captures'].append({'file': '06_mlflow_experiment_overview.png', 'source_url': overview})
        browser.close()
    graph = subprocess.check_output([sys.executable, '-m', 'dvc', 'dag', '--dot'], text=True)
    (output / 'dvc_dag.dot').write_text(graph)
    subprocess.run(['dot', '-Tpng', '-Gdpi=160', '-Grankdir=TB', '-Gbgcolor=white',
                    '-Nfontname=DejaVu Sans', '-Nfontsize=13', '-Nshape=box', '-Nstyle=rounded,filled',
                    '-Nfillcolor=#eef4ff', '-Ncolor=#5078a0', '-Ecolor=#5078a0',
                    '-o', str(output / '07_dvc_pipeline.png')], input=graph, text=True, check=True)
    (output / 'dvc_dag.txt').write_text(subprocess.check_output([sys.executable, '-m', 'dvc', 'dag'], text=True))
    manifest['captures'].append({'file': '07_dvc_pipeline.png',
                                 'source_command': 'dvc dag --dot | dot -Tpng',
                                 'kind': 'rendered actual DVC dependency graph'})
    write_json(output / 'capture_manifest.json', manifest)
    print('Saved actual DVC graph and capture provenance.')


if __name__ == '__main__':
    main()
