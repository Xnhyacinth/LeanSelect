"""Select a checkpoint by benchmark macro-average on held-out development sets.

Manifest: [{"checkpoint": "selector-000020.pt", "results": ["dev/task/summary.json", ...]}, ...].
Paths resolve relative to the manifest. All candidates must cover identical
input files/tasks, model identities, and generation settings. Test summaries
are rejected. The caller defines a development split before running evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .data import file_sha256


def select_best(manifest, output):
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError('best checkpoint output already exists')
    candidates = json.loads(manifest.read_text())
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidate manifest must be a nonempty list')
    reference = None
    results = []
    for candidate in candidates:
        checkpoint = manifest.parent / candidate['checkpoint']
        digest = file_sha256(checkpoint)
        values, identities = {}, {}
        for filename in candidate['results']:
            report = json.loads((manifest.parent / filename).read_text())
            if report.get('split') != 'development' or report.get('method') != 'leanselect':
                raise ValueError('selection accepts only LeanSelect development summaries')
            if report.get('selector_sha256') != digest:
                raise ValueError('development summary is bound to a different checkpoint')
            key = (report['benchmark'], report['task'], report['max_length'])
            if key in values:
                raise ValueError('duplicate development task/length')
            metric = report['metrics']
            score = metric.get('score', metric.get('normalized_exact_match'))
            if not isinstance(score, (int, float)) or not 0 <= score <= 100:
                raise ValueError('development summary lacks a valid percentage score')
            values[key] = float(score)
            identities[key] = {name: report[name] for name in (
                'input_sha256', 'count', 'backbone', 'base_model_sha256',
                'model_config_sha256', 'source_sha256', 'generation', 'seed', 'upstream')}
        if not values:
            raise ValueError('candidate lacks development results')
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError('candidates were not evaluated on the same development protocol')
        benchmark_scores = {}
        for benchmark in sorted({key[0] for key in values}):
            task_scores = [value for key, value in values.items() if key[0] == benchmark]
            benchmark_scores[benchmark] = sum(task_scores) / len(task_scores)
        results.append({'checkpoint': str(checkpoint), 'sha256': digest,
                        'development_macro': sum(benchmark_scores.values()) / len(benchmark_scores),
                        'benchmark_scores': benchmark_scores,
                        'task_scores': [{'benchmark': k[0], 'task': k[1], 'max_length': k[2], 'score': v}
                                        for k, v in sorted(values.items())]})
    winner = min(results, key=lambda row: (-row['development_macro'], row['sha256']))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(winner['checkpoint'], output)
    receipt = {'criterion': 'equal-weight benchmark means of development task-length scores; SHA256 lexical tie-break',
               'test_results_used': False, 'selected_sha256': winner['sha256'],
               'selected_development_macro': winner['development_macro'], 'candidates': results}
    output.with_suffix('.selection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    select_best(args.manifest, args.output)


if __name__ == '__main__':
    main()
