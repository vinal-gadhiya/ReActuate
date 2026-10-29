"""Run matched scene seeds for GR00T, Muse, and Muse + SAM inside the sim container."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

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
               '--action-steps', str(args.action_steps), '--decision-steps', str(args.decision_steps),
               '--max-decisions', str(args.max_decisions), '--max-api-calls', str(args.max_api_calls),
               '--max-replans', str(args.max_replans), '--policy-url', args.policy_url,
               '--supervisor-url', args.supervisor_url, '--output', str(output),
               '--sam' if condition == 'muse_sam' else '--no-sam']
    if args.max_steps is not None:
        command += ['--max-steps', str(args.max_steps)]
    if args.context_file is not None:
        command += ['--context-file', str(args.context_file)]
    if plan is not None:
        command += ['--plan', str(plan)]
    if condition != 'baseline':
        command += ['--subtask-budget-mode', args.subtask_budget_mode]
        command += ['--planning-context', getattr(args, 'planning_context', 'summary')]
        command += ['--supervisor-stop-policy', getattr(args, 'supervisor_stop_policy', 'halt')]
        command += ['--max-supervisor-failures', str(getattr(args, 'max_supervisor_failures', 3))]
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
              'sam_calls': 0, 'sam_calls_with_boxes': 0, 'supervisor_wall_seconds': 0.0}
    for line in (directory / 'trace.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'decision':
            counts['decisions'] += 1
            counts['replans'] += int(event['status'] == 'replan')
            counts['subtasks_marked_complete'] += int(event['status'] == 'complete')
            counts['inspection_decisions'] += int(event['status'] == 'inspect')
        elif event['event'] == 'supervisor_response':
            counts['supervisor_wall_seconds'] += event.get('wall_seconds', 0.0)
            if event.get('sam') is not None:
                counts['sam_calls'] += 1
                counts['sam_calls_with_boxes'] += int(bool(event['sam']['boxes']))
                counts['sam_tracking_calls'] += int(event['sam'].get('mode') == 'track')
        elif event['event'] == 'tool' and event['tool'] == 'navigate':
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
        summary[condition] = {
            'completed': len(completed), 'errors': len(selected) - len(completed),
            'successes': sum(bool(row['success']) for row in completed),
            'success_rate': (sum(bool(row['success']) for row in completed) / len(completed)) if completed else None,
            **{f'mean_{key}': sum(row[key] for row in completed) / len(completed) if completed else None
               for key in ('steps', 'wall_seconds', 'api_calls', 'sam_calls')},
        }
    pairs = []
    for seed in sorted({row['seed'] for row in rows}):
        matched = {row['condition']: row for row in rows if row['seed'] == seed and row['status'] == 'completed'}
        if len(matched) == len(conditions):
            goals_match = len({row['original_goal'] for row in matched.values()}) == 1
            pairs.append({'seed': seed, 'original_goals_match': goals_match,
                          'scene_identities_match': len({json.dumps(row.get('scene_identity'), sort_keys=True)
                                                         for row in matched.values()}) == 1
                          if all(row.get('scene_identity') for row in matched.values()) else None,
                          'success': {key: bool(row['success']) for key, row in matched.items()}})
    return {'conditions': summary, 'paired_seeds': pairs}


def save_report(output, rows, conditions=CONDITIONS):
    report = {'runs': rows, **summarize(rows, conditions), 'limitations': [
        'Success is the original simulator task check; Muse subtask verdicts and SAM boxes are diagnostics only.',
        'Explicit scene pairs, process seeds, and initial scene fingerprints are checked before supervised execution. '
        'Muse and SAM share the initial plan for each seed.',
        'Scene seeds do not make stochastic policy/API outputs identical. Small samples do not establish improvement.',
        'SAM supplies candidate boxes to Muse, not 3D control or a new GR00T skill.',
        'Replans may diverge. Errors are reported separately and are not counted as completed task failures.',
        'The selected frozen policy owns all control actions in every condition. Muse/SAM change instructions and planning, not action values.',
    ]}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Closed-loop comparison', '',
             '| Condition | Completed | Errors | Successes | Mean steps | Mean seconds | Mean API calls |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for condition, metrics in report['conditions'].items():
        values = [f'{metrics[key]:.1f}' if metrics[key] is not None else '—'
                  for key in ('mean_steps', 'mean_wall_seconds', 'mean_api_calls')]
        lines.append(f'| {condition} | {metrics["completed"]} | {metrics["errors"]} | '
                     f'{metrics["successes"]} | ' + ' | '.join(values) + ' |')
    lines += ['', *[f'- {note}' for note in report['limitations']], '']
    (output / 'comparison.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='OpenCabinet')
    parser.add_argument('--split', choices=('pretrain', 'target'), default='pretrain')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--action-steps', type=int, default=8)
    parser.add_argument('--decision-steps', type=int, default=40)
    parser.add_argument('--planning-context', choices=('summary', 'recent'), default='summary',
                        help='Persistent episode summary or recent events only; object memory is retained in both')
    parser.add_argument('--max-decisions', type=int, default=90)
    parser.add_argument('--max-api-calls', type=int, default=200)
    parser.add_argument('--max-replans', type=int, default=2)
    parser.add_argument('--max-supervisor-failures', type=int, default=3)
    parser.add_argument('--supervisor-stop-policy', choices=('halt', 'continue'), default='halt',
                        help='Simulator only: honor Muse stop, or continue short guarded policy bursts within remaining budgets')
    parser.add_argument('--subtask-budget-mode', choices=('plan', 'episode'), default='episode')
    parser.add_argument('--sam-video', action='store_true', help='Opt into SAM clips; default is image-only SAM')
    parser.add_argument('--require-unseen-composite', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS),
                        help='Ordered conditions; baseline must be first, muse precedes muse_sam')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--policy-url', default='http://inference:8000')
    parser.add_argument('--supervisor-url', default='http://supervisor:8001')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_supervisor_failures < 0:
        parser.error('--max-supervisor-failures must be nonnegative')
    if (not args.conditions or args.conditions[0] != 'baseline' or len(set(args.conditions)) != len(args.conditions)
            or ('muse_sam' in args.conditions and ('muse' not in args.conditions
                or args.conditions.index('muse') > args.conditions.index('muse_sam')))):
        parser.error('Conditions must start with baseline, contain no duplicates, and run muse before muse_sam')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if not (args.action_steps > 0 and 1 <= args.decision_steps <= 80
            and args.max_decisions > 0 and 1 <= args.max_api_calls <= 200 and args.max_replans >= 0):
        parser.error('Invalid execution budgets')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    check_services(args.policy_url, args.supervisor_url)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'experiment.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2) + '\n')
    rows = []
    for seed in args.seeds:
        plan = None
        for condition in args.conditions:
            output = args.output / f'seed-{seed}' / condition
            command = run_command(args, condition, seed, output, plan if condition == 'muse_sam' else None)
            print(f'\nStarting {condition}, scene seed {seed}: {output}', flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            with (output.parent / f'{condition}.log').open('w') as log:
                # Child output is streamed to Docker logs and retained for failed runs too.
                child_env = {**os.environ, 'PYTHONHASHSEED': str(seed % (2**32))}
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
            if code:
                rows.append({'seed': seed, 'condition': condition, 'status': 'error', 'returncode': code})
                save_report(args.output, rows, args.conditions)
                raise SystemExit(f'{condition} failed. Inspect {output.parent / (condition + ".log")}; comparison stopped.')
            row = episode_metrics(output / 'episode-000')
            rows.append({**row, 'condition': condition, 'status': 'completed'})
            save_report(args.output, rows, args.conditions)
            if condition == 'muse':
                plan = output / 'episode-000' / 'plan.json'
    print(json.dumps(summarize(rows, args.conditions), indent=2), flush=True)


if __name__ == '__main__':
    main()
