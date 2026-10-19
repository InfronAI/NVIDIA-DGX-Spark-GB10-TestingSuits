"""Length and prefix designs. A design target is never a measured cache hit."""
import hashlib
import json


def length_matrix(context=262144):
    tiers = [2**n for n in range(10,19)]
    return [{'input_tokens': i, 'output_tokens': o, 'admissible': i+o<=context,
             'reason': None if i+o<=context else 'input_plus_output_exceeds_context'} for i in tiers for o in tiers]


def prefix_pair(shared_words=256, suffix_words=256, identity='example'):
    if min(shared_words, suffix_words) < 0:
        raise ValueError('Lengths must be nonnegative')
    shared = ' '.join(f'{identity}_record_{i}' for i in range(shared_words))
    first = shared + ' ' + ' '.join(f'A_{i}' for i in range(suffix_words))
    second = shared + ' ' + ' '.join(f'B_{i}' for i in range(suffix_words))
    return {'seed': first, 'probe': second, 'scope': 'Word-based design only; tokenize full chat templates before claiming token overlap.'}


def load_cases(path, limit):
    cases = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row.get('messages'),list) and not isinstance(row.get('input_ids'),list):
                raise ValueError('Each workload requires messages or input_ids')
            cases.append(row)
            if len(cases)>=limit:break
    if not cases:raise ValueError('Empty workload file')
    if len({r['id'] for r in cases})!=len(cases):raise ValueError('Duplicate workload IDs')
    return cases
