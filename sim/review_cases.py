"""Replay labelled real observation cases to the supervisor, without robot actions.

Expected choices and privileged offline labels are never included in requests.
Extraction produces UNLABELLED cases; it is not evidence of model accuracy.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.goals import normalized
from core.policy_http import SupervisorClient
from core.observations import rgb_diagnostics


def extract_cases(trace):
    trace = Path(trace)
    cases = []
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        if event.get('event') != 'supervisor_request':
            continue
        images = [{'label': item['label'], 'path': str((trace.parent / item['path']).resolve())}
                  for item in event['images'] if item['path'].endswith(('.jpg', '.png'))]
        cases.append({'id': f'{trace.parent.name}-request-{event["request"]}',
                      'partition': 'unassigned', 'episode_id': str(trace.parent.resolve()),
                      'operation': event['operation'], 'context': event['context'], 'images': images,
                      'expected': None})
    return {'schema_version': 1, 'cases': cases,
            'annotation_note': 'Assign whole episodes to development or held_out. Add expected eligible_first_targets '
                               '(multiple accepted choices allowed) for plans, or expected outcome '
                               '(satisfied/unsatisfied) for decisions. Keep labels outside context.'}


def evaluate_cases(manifest, client):
    results = []
    episodes = {}
    for case in manifest['cases']:
        partition = case['partition']
        episode = case['episode_id']
        if episode in episodes and episodes[episode] != partition:
            raise ValueError('An episode cannot occur in both development and held-out partitions')
        episodes[episode] = partition
    for case in manifest['cases']:
        expected = case.get('expected')
        if expected is None or case['partition'] not in ('development', 'held_out'):
            results.append({'id': case['id'], 'status': 'unlabelled', 'partition': case['partition']})
            continue
        try:
            if case['operation'] == 'plan':
                acceptable = expected.get('eligible_first_targets')
                if (not isinstance(acceptable, list) or not acceptable
                        or any(not isinstance(item, str) or not item.strip() for item in acceptable)):
                    raise ValueError('Planning annotations need one or more acceptable eligible_first_targets')
            elif case['operation'] == 'decide':
                label = expected.get('outcome')
                if label not in ('satisfied', 'unsatisfied'):
                    raise ValueError('Completion labels must be satisfied or unsatisfied')
            else:
                raise ValueError('Only plan and decide observations can be replayed')
            context = deepcopy(case['context'])
            context['api_calls_left'] = 2
            context['diagnostic_replay'] = True
            # Context is the captured public request. Expected labels stay here only.
            images = {}
            for item in case['images']:
                with Image.open(item['path']) as saved_image:
                    images[item['label']] = np.asarray(saved_image.convert('RGB'))
            # JPEG audit snapshots are not byte-identical to original raw arrays.
            context['image_diagnostics'] = rgb_diagnostics({key: frame for key, frame in images.items()
                                                           if key.startswith(('current:', 'before:'))})
            response = client.ask(case['operation'], context, images)
            if response.get('validation_error'):
                raise ValueError(response['validation_error'])
            proposal = response['result']
            result = {'id': case['id'], 'partition': case['partition'], 'status': 'completed',
                      'api_calls': response['api_calls'], 'response': response}
            if case['operation'] == 'plan':
                result['eligible_first_choice'] = normalized(proposal['steps'][0]['target']) in {normalized(item) for item in acceptable}
            else:
                actual = proposal['assessment'].get('outcome', 'unknown')
                result.update(expected_outcome=label, reported_outcome=actual,
                              false_positive=label == 'unsatisfied' and actual == 'satisfied',
                              false_negative=label == 'satisfied' and actual == 'unsatisfied',
                              abstained=actual == 'unknown')
            results.append(result)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            results.append({'id': case['id'], 'partition': case['partition'], 'status': 'error', 'error': str(error)})
    summary = {}
    for partition in ('development', 'held_out'):
        rows = [item for item in results if item['partition'] == partition]
        completed = [item for item in rows if item['status'] == 'completed']
        planning = [item for item in completed if 'eligible_first_choice' in item]
        negatives = [item for item in completed if item.get('expected_outcome') == 'unsatisfied']
        summary[partition] = {
            'attempted': sum(item['status'] != 'unlabelled' for item in rows),
            'completed': len(completed), 'errors': sum(item['status'] == 'error' for item in rows),
            'planning_cases': len(planning),
            'eligible_first_choice_rate': sum(item['eligible_first_choice'] for item in planning) / len(planning) if planning else None,
            'negative_completion_cases': len(negatives),
            'false_positive_rate': sum(item['false_positive'] for item in negatives) / len(negatives) if negatives else None,
            'abstentions': sum(item.get('abstained', False) for item in completed),
        }
    return {'results': results, 'summary': summary,
            'limitation': 'Replay uses saved image snapshots (RGB JPEG may be lossy), measures supervisor judgments, and does not establish closed-loop robot success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--extract', type=Path, help='Trace JSONL to turn into unlabelled observation cases')
    source.add_argument('--cases', type=Path, help='Annotated case manifest; makes real supervisor API requests')
    parser.add_argument('--supervisor-url', default='http://supervisor:8001')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    result = extract_cases(args.extract) if args.extract else evaluate_cases(
        json.loads(args.cases.read_text()), SupervisorClient(args.supervisor_url, timeout=480))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
