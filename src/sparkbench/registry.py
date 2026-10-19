"""Small file-based model/experiment registry with explicit coverage semantics."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def resolve(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Registry path escapes repository')
    return path


def catalog(root):
    root = Path(root)
    rows = []
    for path in sorted((root / 'models').glob('*/model.json')):
        model = read(path)
        rows.append({'id': model['id'], 'name': model['name'], 'experiments': model['experiments']})
    return rows


def validate(root):
    root = Path(root).resolve()
    models = catalog(root)
    if not models:
        raise ValueError('No models registered')
    ids = set()
    experiments = []
    for model in models:
        if model['id'] in ids:
            raise ValueError('Duplicate model ID')
        ids.add(model['id'])
        for ref in model['experiments']:
            directory = resolve(root, ref)
            experiment = read(directory / 'experiment.json')
            if experiment['model_id'] != model['id']:
                raise ValueError('Experiment/model identity mismatch')
            if experiment['status'] not in ('planned', 'running', 'execution_ended', 'cancelled'):
                raise ValueError('Unknown execution status')
            if not isinstance(experiment['evidence_complete'], bool):
                raise ValueError('Coverage must be explicit')
            for lang in ('zh', 'en'):
                for name in ('README', 'conclusions'):
                    if not (directory / f'{name}.{lang}.md').is_file():
                        raise ValueError(f'Missing bilingual experiment document: {name}.{lang}')
            for artifact in read(directory / 'artifacts.json')['files']:
                path = resolve(directory, artifact['path'])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                    raise ValueError('Artifact missing or modified: ' + artifact['path'])
            experiments.append(experiment['id'])
    return {'models': len(models), 'experiments': experiments, 'artifact_hashes': 'verified'}
