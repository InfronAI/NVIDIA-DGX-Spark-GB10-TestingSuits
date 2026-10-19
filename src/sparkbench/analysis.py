"""Recompute a compact result from published tables; no serving/GPU calls."""
import csv
import json
from pathlib import Path
from .registry import read


def summarize(experiment):
    root = Path(experiment)
    presentation = read(root / 'data/service/presentation.json')
    capacity = []
    for group in presentation['groups']:
        if not group['key'].startswith('P1_'):
            continue
        blocks = [b for b in presentation['eligible_blocks'] if b['id'] in group['block_ids']]
        # Aggregate token count over aggregate complete-block wall time, never
        # the unweighted average of per-request token rates.
        if not blocks:
            raise ValueError('Capacity group has no eligible measurement blocks')
        tokens = sum(b['output_tokens'] for b in blocks)
        seconds = sum(b['window_s'] for b in blocks)
        measured = tokens / seconds
        if abs(measured - group['tps']) > 1e-6:
            raise ValueError('Capacity numerator/denominator mismatch')
        capacity.append({'concurrency': int(group['key'].split('_c')[1].split('_')[0]),
                         'output_tokens_s': measured, 'visible_latency_p95_s': group['ttft']['p95'],
                         'responses': group['n'], 'complete_blocks': len(blocks)})
    hardware = read(root / 'data/hardware/summary.json')
    complete = read(root / 'data/completion/summary.json')
    coverage = read(root / 'data/coverage.json')
    longest = max(complete['native_requests'], key=lambda r: r.get('actual_output') or 0)
    return {'model_id': read(root / 'experiment.json')['model_id'], 'capacity': sorted(capacity, key=lambda r:r['concurrency']),
            'hardware_valid_rounds': sum(v[s].get('measured',0) for v in hardware['coverage_by_suite'].values() for s in ['base','extensions']),
            'paired_collective_trials': hardware['paired_collective_trials'],
            'paired_overlap_trials': hardware['paired_overlap_trials'],
            'native_longest_output_tokens': longest['actual_output'],
            'native_longest_decode_tokens_s': longest['decode_token_s'],
            'native_longest_scope': 'Forced output; includes repeated whitespace; not semantic output quality.',
            'public_length': coverage['public_length'], 'interference': coverage['interference'],
            'execution_complete': coverage['execution_complete'], 'evidence_complete': coverage['full_evidence_coverage']}


def write_summary(experiment, output):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    summary = summarize(experiment)
    (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (target / 'capacity.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary['capacity'][0]))
        writer.writeheader(); writer.writerows(summary['capacity'])
    return summary
