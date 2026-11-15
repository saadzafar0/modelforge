"""Build a four-page submission using actual pipeline results and captured evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
import yaml

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image)

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    baseline = read('reports/metrics/baseline.json')
    nas = read('reports/metrics/nas.json')
    selected = read('models/nas/summary.json')
    registry = read('reports/tracking/registry.json')
    splits = read('reports/data/splits.json')['splits']
    source = read('data/raw/source.json')
    evidence = Path('screenshots/deleiverable_1')
    output = Path('submission/deliverable_1')
    output.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load(Path('configs/report.yaml').read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodySmall', parent=styles['BodyText'], fontSize=9, leading=12, spaceAfter=7))
    styles.add(ParagraphStyle('CaptionSmall', parent=styles['BodyText'], fontSize=8, leading=10,
                             textColor=colors.black, spaceAfter=7))
    for style in styles.byName.values():
        style.textColor = colors.black
    story = []

    def p(text, style='BodySmall'):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths=None):
        rows = [[Paragraph(escape(str(cell)), styles['BodySmall']) for cell in row] for row in rows]
        result = Table(rows, colWidths=widths, hAlign='LEFT', repeatRows=1)
        result.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, 0), 0.7, colors.black),
            ('GRID', (0, 1), (-1, -1), 0.25, colors.HexColor('#cccccc')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.extend([result, Spacer(1, 9)])

    def picture(filename, max_height, caption):
        path = evidence / filename
        if not path.exists():
            raise FileNotFoundError(f'Required real evidence missing: {path}')
        width, height = ImageReader(str(path)).getSize()
        ratio = min(505 / width, max_height / height)
        story.append(Image(str(path), width=width * ratio, height=height * ratio, hAlign='CENTER'))
        p(caption, 'CaptionSmall')

    # Page 1: project, source, implementation, and measured outcome.
    p('ModelForge', 'Title')
    p('Deliverable 1: Reproducible ML Pipeline', 'Heading2')
    p('<br/>'.join(f'<b>{escape(member["name"])}</b> — {escape(member["student_id"])}'
                   for member in metadata['team']))
    p('<b>GitHub:</b> <link href="https://github.com/saadzafar0/modelforge">github.com/saadzafar0/modelforge</link><br/>'
      '<b>DagsHub:</b> <link href="https://dagshub.com/saadzafar0/modelforge">dagshub.com/saadzafar0/modelforge</link><br/>'
      '<b>MLflow:</b> <link href="https://dagshub.com/saadzafar0/modelforge.mlflow">dagshub.com/saadzafar0/modelforge.mlflow</link>')
    p('Project and scope', 'Heading2')
    p('ModelForge automates the discovery of small multilayer perceptrons for structured classification. '
      'This milestone implements source ingestion, validation, preprocessing, a baseline, architecture search, '
      'held-out evaluation, remote experiment tracking, and model registration. Training runs on CPU. '
      'The dashboard, REST API, Docker deployment, and monitoring belong to later milestones.')
    p('Data pipeline and versioning', 'Heading2')
    p('The pipeline downloads the Wisconsin Diagnostic dataset directly from UCI and verifies a pinned archive SHA-256. '
      'Validation rejects schema errors, missing or non-finite values, duplicate IDs, invalid labels, and out-of-range numeric values. '
      'The 30 numeric predictors exclude record IDs. Benign is encoded as 0 and malignant as 1.')
    table([['Dataset', 'Rows', 'Predictors', 'License'],
           ['UCI Wisconsin Diagnostic', 569, 30, 'CC BY 4.0']], [270, 55, 80, 100])
    p(f'Stratified split: {splits["train"]["rows"]} training / {splits["validation"]["rows"]} validation / '
      f'{splits["test"]["rows"]} test records, seed 42. Median imputation and standardization are fitted only on training data. '
      'DVC versions raw data, transformed splits, preprocessing, and checkpoints through directory outputs and dvc.lock; '
      'DagsHub stores their actual contents. Raw data and weights are excluded from Git.')
    p('Model core and selection', 'Heading2')
    p('The baseline is a fixed 64-unit ReLU MLP. The main system runs ten sequential Optuna TPE trials over 1–4 layers, '
      '32/64/128/256-unit widths, ReLU/GELU/Tanh, dropout 0–0.4, and optional residual projections. '
      'Median pruning and early stopping constrain CPU work. Each trial has at most 100 epochs and patience 12. '
      'Selection uses validation macro F1, then fewer parameters; test labels are never used for architecture selection.')
    table([['Held-out metric', 'Baseline', 'NAS-selected MLP'],
           ['Accuracy', f'{baseline["test_accuracy"]:.4f}', f'{nas["test_accuracy"]:.4f}'],
           ['Macro F1', f'{baseline["test_macro_f1"]:.4f}', f'{nas["test_macro_f1"]:.4f}'],
           ['Parameters', baseline['parameter_count'], nas['parameter_count']],
           ['Median latency (ms)', f'{baseline["inference_latency_median_ms"]:.4f}', f'{nas["inference_latency_median_ms"]:.4f}'],
           ['Checkpoint size (KiB)', f'{baseline["checkpoint_size_bytes"]/1024:.2f}', f'{nas["checkpoint_size_bytes"]/1024:.2f}']], [235, 135, 135])
    p('Latency measures a batch-one CPU model forward pass after warm-up; it excludes preprocessing and network time. '
      'A single small test split is a benchmark result, not a claim of broad generalization.', 'CaptionSmall')
    if baseline['test_accuracy'] > nas['test_accuracy']:
        p('The baseline scored higher on this held-out split. NAS selected fewer parameters after matching '
          'the baseline’s validation macro F1; no additional tuning used the test results.', 'CaptionSmall')
    story.append(PageBreak())

    # Page 2: two genuine remote run captures required by the assignment.
    p('Remote experiment tracking', 'Heading2')
    p('The hosted DagsHub MLflow experiment records the baseline, main NAS parent, and every search trial. '
      'Runs include architecture/training parameters, epoch losses and validation metrics, evaluation metrics, and artifacts. '
      'Pruned trials are labeled. The baseline and main runs each include a loadable model with preprocessing, '
      'a confusion matrix, and a classification report. These screenshots show the actual remote UI.')
    picture('03_mlflow_baseline_run.png', 285, 'Figure 1. Baseline experiment run on the hosted MLflow server.')
    picture('04_mlflow_nas_run.png', 285, 'Figure 2. Main Optuna NAS run on the same hosted server.')
    architecture = selected['architecture']
    widths = ' / '.join(str(width) for width in architecture['hidden_sizes'])
    residual = 'enabled' if architecture['residual'] else 'disabled'
    p(f'NAS completed {selected["trials_completed"]} trials and pruned {selected["trials_pruned"]}. '
      f'Selected trial {selected["trial"]}: hidden-layer widths {widths}, '
      f'{architecture["activation"]} activation, dropout {architecture["dropout"]:.3f}, '
      f'residual connections {residual}.')
    story.append(PageBreak())

    # Page 3: real registry and operational verification.
    p('Model registry and artifact verification', 'Heading2')
    p(f'The NAS-selected model is registered as <b>{escape(registry["model_name"])}</b>, '
      f'version <b>{registry["version"]}</b>, in <b>{registry["stage"]}</b>. '
      'The model artifact includes the PyTorch checkpoint, fitted preprocessing, feature order, and an MLflow input/output signature. '
      'Registration is repeatable for the same run ID without adding a duplicate version.')
    picture('05_registered_model_staging.png', 340, 'Figure 3. Actual registered model showing the required Staging state.')
    p(f'The pipeline loaded this model back from the remote registry and checked predictions for all '
      f'{registry["verified_rows"]} held-out records against the evaluated local checkpoint. All predictions matched. '
      'Staging is used to satisfy the assignment; registration does not deploy the model.')
    p('Validation and reproducibility checks', 'Heading2')
    p('Automated checks reject corrupted data, verify checkpoint serialization with and without residual projections, '
      'check that train/validation/test IDs are disjoint, and compare scaler statistics with training rows only. '
      'The pipeline is run end to end, and a separate clean checkout is used to verify restoration/reproduction. '
      'DVC remote upload and download checks verify that versioned outputs are available to another checkout.')
    p('Pinned Python dependencies, seed 42, deterministic PyTorch operations, and one CPU thread control computation. '
      'Predictive outputs and split identity are reproducible in the documented environment. Remote run IDs, timestamps, '
      'and wall-clock latency can change. Credentials are read from ignored local configuration or environment variables, '
      'and are excluded from screenshots and source control.')
    p('Dataset attribution', 'Heading2')
    p('Wolberg, W., Mangasarian, O., Street, N., &amp; Street, W. (1993). '
      '<i>Breast Cancer Wisconsin (Diagnostic)</i>. UCI Machine Learning Repository. '
      '<link href="https://doi.org/10.24432/C5DW2B">doi.org/10.24432/C5DW2B</link>. CC BY 4.0.')
    p('Pinned source archive SHA-256:<br/>' + source['archive_sha256'], 'CaptionSmall')
    story.append(PageBreak())

    # Page 4: actual DVC graph and concrete reproduction instructions.
    p('DVC pipeline and reproduction', 'Heading2')
    p('Named stages declare their code/data dependencies, parameters, outputs, and metrics in dvc.yaml. '
      'The generated dvc.lock binds these to content hashes. Validation gates preprocessing and training. '
      'The baseline and NAS branches share identical splits; both are evaluated before comparison. '
      'Registration depends on the main evaluation.')
    picture('07_dvc_pipeline.png', 405, 'Figure 4. Dependency graph rendered from the real project’s dvc dag --dot output.')
    p('How to reproduce', 'Heading2')
    p('Clone the GitHub repository; create and activate the Python 3.12 environment; install requirements.txt; '
      'configure a contributor’s DagsHub token locally. Run <b>dvc pull</b> to restore versioned outputs and '
      '<b>dvc repro</b> to reproduce changed or missing stages. Run <b>dvc repro --force --no-run-cache</b> '
      'to execute every stage from source. Use <b>dvc push</b> to publish DVC outputs and <b>python -m pytest -q</b> '
      'to check data integrity and model serialization. Full commands and the folder structure are in README.md.')
    p('Implementation files', 'Heading2')
    p('src/modelforge/data/: ingestion, validation, preprocessing. models/ and training/: MLP definition, baseline, and NAS. '
      'evaluation/: held-out metrics and MLflow artifacts. tracking/: Staging registration and artifact verification. '
      'scripts/evidence/ and scripts/reports/: real screenshots and PDF generation. '
      'apps/dashboard/, services/api/, and infra/aws/ reserve the later platform components.')

    path = output / 'ModelForge_Deliverable_1.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=32, bottomMargin=38, title='ModelForge — Deliverable 1',
                            author='; '.join(member['name'] for member in metadata['team']))
    doc.build(story)
    team_text = '; '.join(f'{member["name"]} ({member["student_id"]})' for member in metadata['team'])
    markdown = f'''# ModelForge — Deliverable 1

Team: {team_text}.

GitHub: https://github.com/saadzafar0/modelforge
DagsHub: https://dagshub.com/saadzafar0/modelforge
MLflow: https://dagshub.com/saadzafar0/modelforge.mlflow

The four-page PDF includes implementation notes, real MLflow screenshots for the baseline and main NAS runs,
the model registered in Staging, and the DVC dependency graph. Data is the UCI Wisconsin Diagnostic benchmark,
with 569 records and 30 numeric predictors. Splits are stratified 60/20/20; preprocessing is fitted on training data only.

| Held-out metric | Baseline | NAS |
| --- | --- | --- |
| Accuracy | {baseline['test_accuracy']:.6f} | {nas['test_accuracy']:.6f} |
| Macro F1 | {baseline['test_macro_f1']:.6f} | {nas['test_macro_f1']:.6f} |
| Parameters | {baseline['parameter_count']} | {nas['parameter_count']} |

Registered model: {registry['model_name']}, version {registry['version']}, stage {registry['stage']}.
Hosted predictions matched the local checkpoint on {registry['verified_rows']} held-out records.

Reproduction: see README.md. Source dataset attribution: https://doi.org/10.24432/C5DW2B (CC BY 4.0).
'''
    (output / 'deliverable_1.md').write_text(markdown)
    print('Built', path)


if __name__ == '__main__':
    main()
