"""Overnight queue lifecycle regressions; execute inside Thor containers."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sim.batch import command_for, run_child, run_queue, validate_queue


class BatchTests(unittest.TestCase):
    def queue(self):
        return {'common_args': ['--max-steps', '2000'], 'experiments': [
            {'name': 'first', 'task': 'OpenFridge', 'seed': 0},
            {'name': 'second', 'task': 'CloseCabinet', 'seed': 1},
            {'name': 'third', 'task': 'GarnishPancake', 'seed': 2}]}

    def test_queue_rejects_duplicate_paths_and_managed_argument_overrides(self):
        for change in ('duplicate', 'path', 'output_override', 'bad_timeout'):
            queue = deepcopy(self.queue())
            if change == 'duplicate':
                queue['experiments'][1]['name'] = 'first'
            elif change == 'path':
                queue['experiments'][0]['name'] = '../outside'
            elif change == 'output_override':
                queue['common_args'] = ['--output=/outputs/other']
            else:
                queue['experiments'][0]['timeout_seconds'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_queue(queue)

    def test_failure_and_timeout_do_not_prevent_next_experiment(self):
        calls = []

        def child(command, log, timeout, seed):
            calls.append((command, timeout, seed))
            log.write_text('saved child log\n')
            if len(calls) == 1:
                return 'completed', 1
            if len(calls) == 2:
                return 'timeout', -15
            artifact = Path(command[-1]) / 'episode-000'
            artifact.mkdir(parents=True)
            (artifact / 'result.json').write_text(json.dumps({
                'success': False, 'steps': 2000, 'stop_reason': 'episode_step_budget'}))
            return 'completed', 0

        with TemporaryDirectory() as temporary, patch('sim.batch.run_child', side_effect=child):
            output = Path(temporary) / 'overnight'
            queue = self.queue()
            queue['experiments'][1]['timeout_seconds'] = 12
            rows = run_queue(queue, output, timeout=60)
            self.assertEqual([row['status'] for row in rows], ['failed', 'timeout', 'completed'])
            self.assertFalse(rows[2]['task_success'])
            self.assertEqual([call[1] for call in calls], [60, 12, 60])
            self.assertEqual([call[2] for call in calls], [0, 1, 2])
            report = json.loads((output / 'batch-summary.json').read_text())
            self.assertEqual(report['experiments'], rows)
            self.assertIn('episode_step_budget', report['experiments'][2]['stop_reason'])
            self.assertTrue((output / 'first' / 'run.log').is_file())
            with self.assertRaises(FileExistsError):
                run_queue(queue, output)

    def test_interrupt_records_progress_and_leaves_remaining_jobs_pending(self):
        with TemporaryDirectory() as temporary, patch('sim.batch.run_child', side_effect=KeyboardInterrupt):
            output = Path(temporary) / 'interrupted'
            with self.assertRaises(KeyboardInterrupt):
                run_queue(self.queue(), output)
            report = json.loads((output / 'batch-summary.json').read_text())
            self.assertEqual([row['status'] for row in report['experiments']],
                             ['interrupted', 'pending', 'pending'])

    def test_timeout_terminates_and_reaps_real_child_process(self):
        with TemporaryDirectory() as temporary:
            status, code = run_child([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     Path(temporary) / 'child.log', timeout=0.1, seed=0)
        self.assertEqual(status, 'timeout')
        self.assertLess(code, 0)

    def test_default_queue_has_no_wall_clock_limit(self):
        timeouts = []

        def child(command, log, timeout, seed):
            timeouts.append(timeout)
            return 'completed', 1

        with TemporaryDirectory() as temporary, patch('sim.batch.run_child', side_effect=child):
            run_queue(self.queue(), Path(temporary) / 'unlimited')
        self.assertEqual(timeouts, [None, None, None])

    def test_overnight_example_uses_only_a_4000_step_execution_budget(self):
        path = Path(__file__).resolve().parents[1] / 'sim' / 'overnight.example.json'
        queue = validate_queue(json.loads(path.read_text()))
        args = queue['common_args']
        for flag, value in (('--max-steps', '4000'), ('--max-decisions', '0'),
                            ('--max-api-calls', '0'), ('--max-replans', '-1')):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertTrue(all(job.get('timeout_seconds') is None for job in queue['experiments']))

    def test_command_preserves_explicit_experiment_arguments(self):
        job = {'name': 'baseline', 'task': 'OpenFridge', 'mode': 'baseline', 'seed': 7,
               'args': ['--no-boundary-review']}
        command = command_for(job, ['--max-steps', '2000'], Path('/outputs/job/run'))
        self.assertEqual(command[:4], [sys.executable, '-u', '-m', 'sim.run_agent'])
        self.assertIn('--no-boundary-review', command)
        self.assertEqual(command[-10:], ['--task', 'OpenFridge', '--mode', 'baseline',
                                        '--seed', '7', '--episodes', '1', '--output', '/outputs/job/run'])


if __name__ == '__main__':
    unittest.main()
