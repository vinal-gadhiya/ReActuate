"""Run matched scene seeds for GR00T, Muse, and Muse + SAM inside the sim container."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


CONDITIONS = ('baseline', 'muse', 'muse_sam')


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
    return command


def episode_metrics(directory):
    result = json.loads((directory / 'result.json').read_text())
    counts = {'decisions': 0, 'replans': 0, 'subtasks_marked_complete': 0,
              'sam_calls': 0, 'sam_calls_with_boxes': 0, 'supervisor_wall_seconds': 0.0}
    for line in (directory / 'trace.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'decision':
            counts['decisions'] += 1
            counts['replans'] += int(event['status'] == 'replan')
            counts['subtasks_marked_complete'] += int(event['status'] == 'complete')
        elif event['event'] == 'supervisor_response':
            counts['supervisor_wall_seconds'] += event.get('wall_seconds', 0.0)
            if event.get('sam') is not None:
                counts['sam_calls'] += 1
                counts['sam_calls_with_boxes'] += int(bool(event['sam']['boxes']))
    return {**result, **counts}


def summarize(rows):
    summary = {}
    for condition in CONDITIONS:
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
        if len(matched) == len(CONDITIONS):
            goals_match = len({row['original_goal'] for row in matched.values()}) == 1
            pairs.append({'seed': seed, 'original_goals_match': goals_match,
                          'success': {key: bool(row['success']) for key, row in matched.items()}})
    return {'conditions': summary, 'paired_seeds': pairs}


def save_report(output, rows):
    report = {'runs': rows, **summarize(rows), 'limitations': [
        'Success is the original simulator task check; Muse subtask verdicts and SAM boxes are diagnostics only.',
        'Scene seeds, horizon, and action chunk sizes match. Muse and SAM share the initial plan for each seed.',
        'Scene seeds do not make stochastic policy/API outputs identical. Small samples do not establish improvement.',
        'SAM supplies candidate boxes to Muse, not 3D control or a new GR00T skill.',
        'Replans may diverge. Errors are reported separately and are not counted as completed task failures.',
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
    parser.add_argument('--max-decisions', type=int, default=90)
    parser.add_argument('--max-api-calls', type=int, default=200)
    parser.add_argument('--max-replans', type=int, default=2)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--policy-url', default='http://inference:8000')
    parser.add_argument('--supervisor-url', default='http://supervisor:8001')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if not (args.action_steps > 0 and 1 <= args.decision_steps <= 80
            and args.max_decisions > 0 and 1 <= args.max_api_calls <= 200 and args.max_replans >= 0):
        parser.error('Invalid execution budgets')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'experiment.json').write_text(json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2) + '\n')
    rows = []
    for seed in args.seeds:
        plan = None
        for condition in CONDITIONS:
            output = args.output / f'seed-{seed}' / condition
            command = run_command(args, condition, seed, output, plan if condition == 'muse_sam' else None)
            print(f'\nStarting {condition}, scene seed {seed}: {output}', flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            with (output.parent / f'{condition}.log').open('w') as log:
                # Child output is streamed to Docker logs and retained for failed runs too.
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
            if code:
                rows.append({'seed': seed, 'condition': condition, 'status': 'error', 'returncode': code})
                save_report(args.output, rows)
                raise SystemExit(f'{condition} failed. Inspect {output.parent / (condition + ".log")}; comparison stopped.')
            row = episode_metrics(output / 'episode-000')
            rows.append({**row, 'condition': condition, 'status': 'completed'})
            save_report(args.output, rows)
            if condition == 'muse':
                plan = output / 'episode-000' / 'plan.json'
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
