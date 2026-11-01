"""Run an overnight experiment queue sequentially inside the Thor sim container."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


RESERVED = {'--task', '--mode', '--seed', '--episodes', '--output'}


def validate_args(values):
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError('args must be a list of command-line strings')
    if any(value.split('=', 1)[0] in RESERVED for value in values):
        raise ValueError('Set task, mode and seed in experiment fields; output and episodes are managed by the runner')
    return values


def validate_queue(queue):
    if not isinstance(queue, dict) or set(queue) - {'common_args', 'experiments'}:
        raise ValueError('Queue must contain experiments and optional common_args')
    validate_args(queue.get('common_args', []))
    jobs = queue.get('experiments')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Queue needs at least one experiment')
    names = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) - {'name', 'task', 'mode', 'seed', 'args', 'timeout_seconds'}:
            raise ValueError('Invalid experiment fields')
        name = job.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in names:
            raise ValueError('Experiment names must be unique and contain only letters, numbers, underscores or hyphens')
        names.add(name)
        task = job.get('task', '')
        if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', task):
            raise ValueError('Each experiment needs a task name')
        if job.get('mode', 'agent') not in ('agent', 'baseline', 'fixed') or type(job.get('seed', 0)) is not int:
            raise ValueError('Invalid experiment mode or seed')
        validate_args(job.get('args', []))
        timeout = job.get('timeout_seconds')
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('timeout_seconds must be positive and finite')
    return queue


def command_for(job, common_args, output):
    return [sys.executable, '-u', '-m', 'sim.run_agent',
            *common_args, *job.get('args', []),
            '--task', job['task'], '--mode', job.get('mode', 'agent'),
            '--seed', str(job.get('seed', 0)), '--episodes', '1', '--output', str(output)]


def terminate(process):
    """Reap the experiment before another process can open a rendering context."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_child(command, log, timeout, seed):
    with log.open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, 'PYTHONHASHSEED': str(seed % (2**32))})
        try:
            return 'completed', process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return 'timeout', process.returncode
        except BaseException:
            terminate(process)
            raise


def save_report(output, rows):
    report = {'experiments': rows,
              'note': 'Process completion is separate from environment task success. Shared policy RNG is not reset between jobs.'}
    temporary = output / 'batch-summary.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output / 'batch-summary.json')
    lines = ['| Experiment | Task | Status | Task success | Seconds |',
             '|---|---|---|---|---:|']
    for row in rows:
        success = row.get('task_success')
        lines.append(f'| {row["name"]} | {row["task"]} | {row["status"]} | '
                     f'{"unknown" if success is None else str(success).lower()} | {row.get("wall_seconds", 0):.1f} |')
    (output / 'batch-summary.md').write_text('\n'.join(lines) + '\n')


def run_queue(queue, output, timeout=None):
    validate_queue(queue)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'queue.json').write_text(json.dumps(queue, indent=2) + '\n')
    rows = [{'name': job['name'], 'task': job['task'], 'mode': job.get('mode', 'agent'),
             'seed': job.get('seed', 0), 'status': 'pending'} for job in queue['experiments']]
    save_report(output, rows)
    for number, (job, row) in enumerate(zip(queue['experiments'], rows), 1):
        directory = output / job['name']
        directory.mkdir()
        command = command_for(job, queue.get('common_args', []), directory / 'run')
        limit = job.get('timeout_seconds', timeout)
        row.update(status='running', command=command, output=str(directory / 'run'),
                   log=str(directory / 'run.log'), timeout_seconds=limit)
        save_report(output, rows)
        print(f'[{number}/{len(rows)}] Starting {job["name"]}: {job["task"]}', flush=True)
        started = time.monotonic()
        try:
            status, code = run_child(command, directory / 'run.log', limit, row['seed'])
            row.update(status='failed' if status == 'completed' and code else status, returncode=code)
            artifact = directory / 'run' / 'episode-000'
            result = artifact / 'result.json'
            if result.is_file():
                data = json.loads(result.read_text())
                if (not isinstance(data, dict) or type(data.get('success')) is not bool
                        or type(data.get('steps')) is not int or not isinstance(data.get('stop_reason'), str)):
                    raise ValueError('Invalid episode result artifact')
                row.update(task_success=data['success'], steps=data['steps'], stop_reason=data['stop_reason'])
            elif row['status'] == 'completed':
                row.update(status='failed', error='Process exited without an episode result')
            error = artifact / 'error.json'
            if error.is_file():
                row['error_detail'] = json.loads(error.read_text())
        except KeyboardInterrupt:
            row.update(status='interrupted', wall_seconds=time.monotonic() - started)
            save_report(output, rows)
            raise
        except (OSError, ValueError, KeyError) as error:
            row.update(status='failed', error=str(error))
        row['wall_seconds'] = time.monotonic() - started
        save_report(output, rows)
        print(f'[{number}/{len(rows)}] {job["name"]}: {row["status"]}; '
              f'task_success={row.get("task_success", "unknown")}; {row["wall_seconds"]:.0f}s', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing runs are never overwritten')
    parser.add_argument('--timeout-seconds', type=float, default=None,
                        help='Optional wall-clock limit per experiment; default: no time limit')
    args = parser.parse_args()
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error('--timeout-seconds must be positive and finite')
    try:
        queue = validate_queue(json.loads(args.queue.read_text()))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    run_queue(queue, args.output, args.timeout_seconds)


if __name__ == '__main__':
    main()
