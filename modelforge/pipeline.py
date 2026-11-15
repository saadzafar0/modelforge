"""Named commands used by dvc.yaml; run from the repository root."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['ingest', 'validate', 'preprocess', 'baseline', 'search',
                                         'evaluate-baseline', 'evaluate-nas', 'compare', 'register'])
    stage = parser.parse_args().stage
    if stage in ('ingest', 'validate', 'preprocess'):
        from modelforge import data
        getattr(data, stage)()
    elif stage in ('baseline', 'search'):
        from modelforge import train
        getattr(train, stage)()
    elif stage.startswith('evaluate-'):
        from modelforge.evaluate import evaluate
        evaluate(stage.split('-', 1)[1])
    elif stage == 'compare':
        from modelforge.evaluate import compare
        compare()
    else:
        from modelforge.registry import register
        register()


if __name__ == '__main__':
    main()
