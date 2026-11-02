"""Run matched scene seeds for GR00T, Muse, and Muse + SAM inside the sim container."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import re
from itertools import product

from core.policy_http import PolicyClient, SupervisorClient


CONDITIONS = ('baseline', 'muse', 'muse_sam')


def check_services(policy_url, supervisor_url):
    """Check local service metadata before spending a baseline rollout budget."""
    errors = []
    for name, client in (
        ('inference', PolicyClient(policy_url, timeout=10)),
        ('supervisor', SupervisorClient(supervisor_url, timeout=10)),
    ):
        try:
            client.metadata()
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f'{name} ({client.url}): {error}')
    if errors:
        raise SystemExit(
            'Comparison service preflight failed before any rollout:\n' + '\n'.join(errors)
            + '\nOn Thor, start the services and wait for health:\n'
            'sudo docker compose --profile agent up -d --wait inference supervisor\n'
            'If startup fails, inspect: sudo docker compose --profile agent logs --tail=80 inference supervisor\n'
            'Building images or running the disposable test container does not start the supervisor service.'
        )


def run_command(args, condition, seed, output, plan=None):
    command = [sys.executable, '-m', 'sim.run_agent', '--task', args.task,
               '--split', args.split, '--seed', str(seed), '--episodes', '1',
               '--mode', 'baseline' if condition == 'baseline' else 'agent',
               '--decision-steps', str(args.decision_steps),
               '--max-decisions', str(args.max_decisions), '--max-api-calls', str(args.max_api_calls),
               '--max-replans', str(args.max_replans), '--policy-url', args.policy_url,
               '--supervisor-url', args.supervisor_url, '--output', str(output),
               '--sam' if condition == 'muse_sam' else '--no-sam']
    if args.action_steps is not None:
        command += ['--action-steps', str(args.action_steps)]
    if getattr(args, 'expected_policy', None) is not None:
        command += ['--expected-policy', args.expected_policy]
    if args.max_steps is not None:
        command += ['--max-steps', str(args.max_steps)]
    if args.context_file is not None:
        command += ['--context-file', str(args.context_file)]
    if plan is not None:
        command += ['--plan', str(plan)]
    if condition != 'baseline':
        command += ['--boundary-review' if getattr(args, 'boundary_review', True) else '--no-boundary-review']
        command += ['--verification-mode', getattr(args, 'verification_mode', 'selective')]
        command += ['--execution-policy', getattr(args, 'execution_policy', 'episode')]
        if getattr(args, 'depth', False):
            command += ['--depth', '--depth-max-metres', str(getattr(args, 'depth_max_metres', 3.0))]
        command += ['--subtask-budget-mode', args.subtask_budget_mode]
        command += ['--planning-context', getattr(args, 'planning_context', 'summary')]
        command += ['--supervisor-stop-policy', getattr(args, 'supervisor_stop_policy', 'halt')]
        command += ['--max-supervisor-failures', str(getattr(args, 'max_supervisor_failures', 3))]
        command += ['--max-stop-continuations', str(getattr(args, 'max_stop_continuations', 2))]
        command += ['--max-goal-failures', str(getattr(args, 'max_goal_failures', 3)),
                    '--max-goal-recoveries', str(getattr(args, 'max_goal_recoveries', 2))]
        command += ['--reference-scene', str(output.parent / 'baseline' / 'episode-000' / 'scene.json')]
    if getattr(args, 'require_unseen_composite', False):
        command += ['--require-unseen-composite']
    if getattr(args, 'sam_video', False):
        command += ['--sam-video']
    return command


def episode_metrics(directory):
    result = json.loads((directory / 'result.json').read_text())
    counts = {'decisions': 0, 'replans': 0, 'subtasks_marked_complete': 0,
              'accepted_handoffs': 0, 'blocked_handoffs': 0,
              'navigation_bursts': 0, 'inspection_decisions': 0, 'sam_tracking_calls': 0,
              'sam_calls': 0, 'sam_calls_with_boxes': 0, 'supervisor_wall_seconds': 0.0,
              'planner_calls': 0, 'verification_calls': 0, 'supervisor_failed_requests': 0}
    for line in (directory / 'trace.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'decision':
            counts['decisions'] += 1
            counts['replans'] += int(event['status'] == 'replan')
            counts['subtasks_marked_complete'] += int(event['status'] == 'complete')
            counts['inspection_decisions'] += int(event['status'] == 'inspect')
        elif event['event'] == 'supervisor_response':
            counts['supervisor_wall_seconds'] += event.get('wall_seconds', 0.0)
            counts['planner_calls'] += sum(item.get('role') == 'planner' for item in event.get('audit', []))
            counts['verification_calls'] += sum(item.get('role') in (
                'boundary_review', 'completion_confirmation', 'selective_verification') for item in event.get('audit', []))
            if event.get('sam') is not None:
                counts['sam_calls'] += 1
                counts['sam_calls_with_boxes'] += int(bool(event['sam']['boxes']))
                counts['sam_tracking_calls'] += int(event['sam'].get('mode') == 'track')
        elif event['event'] == 'supervisor_request_failed':
            counts['supervisor_wall_seconds'] += event.get('wall_seconds', 0.0)
            counts['supervisor_failed_requests'] += 1
        elif event['event'] == 'tool' and ('navigation_source' in event or event['tool'] == 'navigate'):
            counts['navigation_bursts'] += 1
        elif event['event'] == 'subtask_transition':
            counts['accepted_handoffs'] += 1
        elif event['event'] == 'transition_blocked':
            counts['blocked_handoffs'] += 1
    return {**result, **counts}


def summarize(rows, conditions=CONDITIONS):
    summary = {}
    for condition in conditions:
        selected = [row for row in rows if row['condition'] == condition]
        completed = [row for row in selected if row['status'] == 'completed']
        attempted = [row for row in selected if row['status'] != 'skipped']
        successes = sum(bool(row['success']) for row in completed)
        summary[condition] = {
            'attempted': len(attempted), 'completed': len(completed),
            'skipped': len(selected) - len(attempted),
            'errors': sum(row['status'] == 'error' for row in selected),
            'error_categories': {category: sum(row.get('error_category', 'unclassified') == category for row in selected if row['status'] == 'error')
                                 for category in sorted({row.get('error_category', 'unclassified') for row in selected if row['status'] == 'error'})},
            'successes': successes,
            'completed_failure_categories': {
                category: sum((row.get('failure_category') or 'unclassified') == category for row in completed if not row['success'])
                for category in sorted({row.get('failure_category') or 'unclassified' for row in completed if not row['success']})},
            'all_attempted_success_rate': successes / len(attempted) if attempted else None,
            'completed_rollout_success_rate': successes / len(completed) if completed else None,
            'success_rate': (sum(bool(row['success']) for row in completed) / len(completed)) if completed else None,
            **{f'mean_{key}': sum(row[key] for row in completed) / len(completed) if completed else None
               for key in ('steps', 'wall_seconds', 'api_calls', 'sam_calls')},
        }
    pairs = []
    keys = sorted({(row.get('task', ''), row['seed']) for row in rows})
    for task, seed in keys:
        matched = {row['condition']: row for row in rows if row['seed'] == seed and row.get('task', '') == task and row['status'] == 'completed'}
        if len(matched) == len(conditions):
            goals_match = len({row['original_goal'] for row in matched.values()}) == 1
            pairs.append({'task': task, 'seed': seed, 'original_goals_match': goals_match,
                          'scene_identities_match': len({json.dumps(row.get('scene_identity'), sort_keys=True)
                                                         for row in matched.values()}) == 1
                          if all(row.get('scene_identity') for row in matched.values()) else None,
                          'success': {key: bool(row['success']) for key, row in matched.items()}})
    paired = {}
    for condition in conditions:
        if condition == 'baseline':
            continue
        counts = dict(both_succeed=0, both_fail=0, agent_improvement=0, baseline_success_regression=0,
                      operational_failure_pairs=0, unverified_pairs=0)
        for task, seed in keys:
            selected = {row['condition']: row for row in rows if row['seed'] == seed and row.get('task', '') == task}
            if 'baseline' not in selected or condition not in selected:
                continue
            baseline, agent = selected['baseline'], selected[condition]
            if (baseline['status'] != 'completed' or agent['status'] != 'completed'
                    or baseline.get('failure_category') == 'infrastructure' or agent.get('failure_category') == 'infrastructure'):
                counts['operational_failure_pairs'] += 1
                continue
            if (not baseline.get('scene_identity') or baseline['scene_identity'] != agent.get('scene_identity')
                    or baseline.get('original_goal') != agent.get('original_goal')):
                counts['unverified_pairs'] += 1
                continue
            b, a = bool(baseline['success']), bool(agent['success'])
            key = 'both_succeed' if b and a else 'both_fail' if not b and not a else 'agent_improvement' if a else 'baseline_success_regression'
            counts[key] += 1
        denominator = counts['both_succeed'] + counts['baseline_success_regression']
        counts['baseline_success_regression_rate'] = counts['baseline_success_regression'] / denominator if denominator else None
        paired[condition] = counts
    per_task = {}
    for task in sorted({row.get('task', '') for row in rows}):
        per_task[task] = {}
        for condition in conditions:
            selected = [row for row in rows if row.get('task', '') == task and row['condition'] == condition]
            attempted = [row for row in selected if row['status'] != 'skipped']
            completed = [row for row in attempted if row['status'] == 'completed']
            successes = sum(bool(row['success']) for row in completed)
            per_task[task][condition] = {'attempted': len(attempted), 'completed': len(completed), 'successes': successes,
                                         'all_attempted_success_rate': successes / len(attempted) if attempted else None}
    return {'conditions': summary, 'per_task': per_task, 'paired_seeds': pairs, 'paired_outcomes': paired}


def run_child(command, log_path, child_env):
    with log_path.open('w') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return process.wait()


def save_report(output, rows, conditions=CONDITIONS):
    report = {'runs': rows, **summarize(rows, conditions), 'limitations': [
        'Success is the original simulator task check; Muse subtask verdicts and SAM boxes are diagnostics only.',
        'Explicit scene pairs, process seeds, and initial scene fingerprints are checked before supervised execution. '
        'Muse and SAM share the initial plan for each seed.',
        'Scene seeds do not make stochastic policy/API outputs identical. Small samples do not establish improvement.',
        'SAM supplies candidate boxes to Muse, not 3D control or a new GR00T skill.',
        'Replans may diverge. Errors are separated from completed-rollout task failures and included in all-attempted reliability.',
        'The selected frozen policy owns all control actions in every condition. Muse/SAM change instructions and planning, not action values.',
    ]}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Closed-loop comparison', '',
             '| Condition | Attempted | Completed | Errors | Successes | Success/completed | Success/attempted |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for condition, metrics in report['conditions'].items():
        rates = ['—' if metrics[key] is None else f'{metrics[key]:.1%}'
                 for key in ('completed_rollout_success_rate', 'all_attempted_success_rate')]
        lines.append(f'| {condition} | {metrics["attempted"]} | {metrics["completed"]} | {metrics["errors"]} | '
                     f'{metrics["successes"]} | ' + ' | '.join(rates) + ' |')
    lines += ['', 'Paired outcomes (verified matching scenes):', '', '```json',
              json.dumps(report['paired_outcomes'], indent=2), '```']
    lines += ['', *[f'- {note}' for note in report['limitations']], '']
    (output / 'comparison.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='OpenCabinet')
    parser.add_argument('--tasks', nargs='+', help='Optional task-family suite; applies the same settings to every task')
    parser.add_argument('--split', choices=('pretrain', 'target'), default='pretrain')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--action-steps', type=int,
                        help='Actions per prediction; default uses backend recommendation in every condition')
    parser.add_argument('--expected-policy', choices=('gr00t', 'patchwam'), default=os.environ.get('EXPECTED_POLICY'))
    parser.add_argument('--decision-steps', type=int, default=40)
    parser.add_argument('--planning-context', choices=('summary', 'recent'), default='summary',
                        help='Persistent episode summary or recent events only; object memory is retained in both')
    parser.add_argument('--max-decisions', type=int, default=90)
    parser.add_argument('--max-api-calls', type=int, default=200)
    parser.add_argument('--max-replans', type=int, default=2)
    parser.add_argument('--max-goal-failures', type=int, default=3)
    parser.add_argument('--max-goal-recoveries', type=int, default=2)
    parser.add_argument('--max-supervisor-failures', type=int, default=3)
    parser.add_argument('--max-stop-continuations', type=int, default=2)
    parser.add_argument('--supervisor-stop-policy', choices=('halt', 'continue'), default='halt',
                        help='Simulator only: honor Muse stop, or continue short guarded policy bursts within remaining budgets')
    parser.add_argument('--subtask-budget-mode', choices=('plan', 'episode'), default='episode')
    parser.add_argument('--sam-video', action='store_true', help='Opt into SAM clips; default is image-only SAM')
    parser.add_argument('--boundary-review', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verification-mode', choices=('selective', 'always'), default='selective')
    parser.add_argument('--depth', action='store_true', help='Give Muse aligned depth views in supervised conditions only')
    parser.add_argument('--depth-max-metres', type=float, default=3.0)
    parser.add_argument('--require-unseen-composite', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS),
                        help='Ordered conditions; baseline must be first, muse precedes muse_sam')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--execution-policy', choices=('episode', 'bounded'), default='episode')
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--policy-url', default='http://inference:8000')
    parser.add_argument('--supervisor-url', default='http://supervisor:8001')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = args.tasks or [args.task]
    if len(tasks) != len(set(tasks)) or any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', task) is None for task in tasks):
        parser.error('Tasks must be unique registered-style task names')
    if not math.isfinite(args.depth_max_metres) or args.depth_max_metres <= 0:
        parser.error('--depth-max-metres must be finite and positive')
    if args.max_supervisor_failures < 0:
        parser.error('--max-supervisor-failures must be nonnegative')
    if args.max_goal_failures < 1 or args.max_goal_recoveries < 0:
        parser.error('Goal failure limit must be positive and goal recovery limit nonnegative')
    if args.max_stop_continuations < 0:
        parser.error('--max-stop-continuations must be nonnegative')
    if (not args.conditions or args.conditions[0] != 'baseline' or len(set(args.conditions)) != len(args.conditions)
            or ('muse_sam' in args.conditions and ('muse' not in args.conditions
                or args.conditions.index('muse') > args.conditions.index('muse_sam')))):
        parser.error('Conditions must start with baseline, contain no duplicates, and run muse before muse_sam')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if not ((args.action_steps is None or args.action_steps > 0) and 1 <= args.decision_steps <= 80
            and args.max_decisions >= 0 and args.max_api_calls >= 0 and args.max_replans >= -1):
        parser.error('Invalid execution budgets')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    check_services(args.policy_url, args.supervisor_url)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'experiment.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2) + '\n')
    rows = []
    for task, seed in product(tasks, args.seeds):
        args.task = task
        plan = None
        for condition in args.conditions:
            root = args.output / task if len(tasks) > 1 else args.output
            output = root / f'seed-{seed}' / condition
            prerequisite = output.parent / 'baseline' / 'episode-000' / 'scene.json'
            if condition != 'baseline' and (not prerequisite.is_file() or (condition == 'muse_sam' and plan is None)):
                rows.append({'task': task, 'seed': seed, 'condition': condition, 'status': 'skipped',
                             'reason': 'Missing matched baseline scene or initial Muse plan'})
                save_report(args.output, rows, args.conditions)
                continue
            command = run_command(args, condition, seed, output, plan if condition == 'muse_sam' else None)
            print(f'\nStarting {condition}, scene seed {seed}: {output}', flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            child_env = {**os.environ, 'PYTHONHASHSEED': str(seed % (2**32))}
            try:
                code = run_child(command, output.parent / f'{condition}.log', child_env)
            except OSError as error:
                rows.append({'task': task, 'seed': seed, 'condition': condition, 'status': 'error',
                             'error_category': 'infrastructure', 'error': str(error)})
                save_report(args.output, rows, args.conditions)
                continue
            if code:
                artifact = output / 'episode-000' / 'error.json'
                try:
                    detail = json.loads(artifact.read_text()) if artifact.is_file() else {}
                except (OSError, ValueError):
                    detail = {'error_category': 'artifact_or_harness'}
                rows.append({'task': task, 'seed': seed, 'condition': condition, 'status': 'error', 'returncode': code,
                             'error_category': detail.get('error_category', 'unclassified'), 'error': detail.get('error', '')})
                save_report(args.output, rows, args.conditions)
                continue
            try:
                row = episode_metrics(output / 'episode-000')
            except (OSError, ValueError, KeyError) as error:
                rows.append({'task': task, 'seed': seed, 'condition': condition, 'status': 'error',
                             'error_category': 'artifact_or_harness', 'error': str(error)})
                save_report(args.output, rows, args.conditions)
                continue
            rows.append({**row, 'condition': condition, 'status': 'completed'})
            save_report(args.output, rows, args.conditions)
            if condition == 'muse':
                plan = output / 'episode-000' / 'plan.json'
    print(json.dumps(summarize(rows, args.conditions), indent=2), flush=True)


if __name__ == '__main__':
    main()
