"""Comparison checks without a simulator, GPU, or live API requests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from sim.compare import check_services, episode_metrics, run_command, summarize
from sim.compare import main


class CompareTests(unittest.TestCase):
    def test_paired_regressions_and_all_attempted_rates_include_operational_errors(self):
        rows = []
        for seed, baseline, agent in ((0, True, False), (1, False, True), (2, True, True), (3, False, False)):
            for condition, success in (('baseline', baseline), ('muse', agent)):
                rows.append({'task': 'TaskFamilyA', 'seed': seed, 'condition': condition, 'success': success,
                             'status': 'completed', 'scene_identity': {'seed': seed}, 'original_goal': 'Original',
                             'steps': 20, 'wall_seconds': 1, 'api_calls': 1, 'sam_calls': 0})
        rows.append({'task': 'TaskFamilyA', 'seed': 4, 'condition': 'muse', 'status': 'error', 'error_category': 'infrastructure'})
        report = summarize(rows, ('baseline', 'muse'))
        metrics = report['conditions']['muse']
        self.assertEqual(metrics['completed_rollout_success_rate'], 0.5)
        self.assertEqual(metrics['all_attempted_success_rate'], 0.4)
        outcomes = report['paired_outcomes']['muse']
        for key in ('both_succeed', 'both_fail', 'agent_improvement', 'baseline_success_regression'):
            self.assertEqual(outcomes[key], 1)
        self.assertEqual(outcomes['baseline_success_regression_rate'], 0.5)

    def test_comparison_continues_to_later_seed_after_child_failure(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'suite'

            def child(command, log, env):
                seed = int(command[command.index('--seed') + 1])
                condition = command[command.index('--mode') + 1]
                calls.append((seed, condition))
                if seed == 0:
                    return 1
                episode = Path(command[command.index('--output') + 1]) / 'episode-000'
                episode.mkdir(parents=True)
                (episode / 'scene.json').write_text('{}')
                (episode / 'trace.jsonl').write_text('')
                (episode / 'plan.json').write_text('{}')
                (episode / 'result.json').write_text(json.dumps({
                    'task': 'ExampleTask', 'seed': seed, 'success': True, 'original_goal': 'Original',
                    'scene_identity': {'seed': seed}, 'steps': 1, 'wall_seconds': 1, 'api_calls': 0}))
                return 0

            argv = ['compare', '--task', 'ExampleTask', '--seeds', '0', '1', '--conditions', 'baseline', 'muse', '--output', str(output)]
            with patch('sys.argv', argv), patch('sim.compare.check_services'), patch('sim.compare.run_child', side_effect=child):
                main()
            report = json.loads((output / 'comparison.json').read_text())
            self.assertEqual(calls, [(0, 'baseline'), (1, 'baseline'), (1, 'agent')])
            self.assertEqual(report['conditions']['baseline']['errors'], 1)
            self.assertEqual(report['conditions']['muse']['skipped'], 1)
            self.assertEqual(report['conditions']['muse']['completed'], 1)

    def test_missing_supervisor_fails_preflight_with_startup_instructions(self):
        with patch('sim.compare.PolicyClient.metadata', return_value={'protocol_version': 1}), \
                patch('sim.compare.SupervisorClient.metadata', side_effect=URLError('name resolution failed')):
            with self.assertRaises(SystemExit) as failure:
                check_services('http://inference:8000', 'http://supervisor:8001')
        message = str(failure.exception)
        self.assertIn('before any rollout', message)
        self.assertIn('supervisor (http://supervisor:8001)', message)
        self.assertIn('up -d --wait inference supervisor', message)

    def test_preflight_checks_both_services_without_any_policy_or_meta_inference(self):
        with patch('sim.compare.PolicyClient.metadata', return_value={'protocol_version': 1}) as policy, \
                patch('sim.compare.SupervisorClient.metadata', return_value={'protocol_version': 1}) as supervisor:
            check_services('http://inference:8000', 'http://supervisor:8001')
        policy.assert_called_once_with()
        supervisor.assert_called_once_with()

    def test_sam_condition_reuses_plan_and_identical_control_settings(self):
        args = SimpleNamespace(task='OpenCabinet', split='pretrain', action_steps=8,
                               decision_steps=40, max_decisions=90, max_api_calls=200,
                               max_replans=2, policy_url='http://inference:8000',
                               supervisor_url='http://supervisor:8001', max_steps=600, context_file=None,
                               subtask_budget_mode='episode')
        baseline = run_command(args, 'baseline', 3, Path('/outputs/base'))
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        sam = run_command(args, 'muse_sam', 3, Path('/outputs/sam'), Path('/outputs/muse/episode-000/plan.json'))
        self.assertIn('--no-sam', muse)
        self.assertIn('--sam', sam)
        self.assertEqual(sam[sam.index('--subtask-budget-mode') + 1], 'episode')
        self.assertEqual(muse[muse.index('--verification-mode') + 1], 'selective')
        args.verification_mode = 'always'
        legacy = run_command(args, 'muse', 3, Path('/outputs/legacy'))
        self.assertEqual(legacy[legacy.index('--verification-mode') + 1], 'always')
        self.assertNotIn('--verification-mode', baseline)
        self.assertNotIn('--plan', baseline)
        self.assertNotIn('--reference-scene', baseline)
        self.assertIn('--reference-scene', muse)
        self.assertIn('--reference-scene', sam)
        self.assertEqual(sam[sam.index('--plan') + 1], '/outputs/muse/episode-000/plan.json')
        for flag in ('--seed', '--max-steps', '--action-steps', '--decision-steps'):
            self.assertEqual(baseline[baseline.index(flag) + 1], sam[sam.index(flag) + 1])
        args.action_steps, args.expected_policy = None, 'patchwam'
        for condition in ('baseline', 'muse', 'muse_sam'):
            command = run_command(args, condition, 3, Path('/outputs/test'))
            self.assertNotIn('--action-steps', command)
            self.assertEqual(command[command.index('--expected-policy') + 1], 'patchwam')
        args.hybrid_control, args.require_unseen_composite = True, True
        baseline = run_command(args, 'baseline', 3, Path('/outputs/base'))
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        self.assertNotIn('--hybrid-control', baseline)
        self.assertNotIn('--hybrid-control', muse)
        self.assertIn('--require-unseen-composite', baseline)
        self.assertIn('--require-unseen-composite', muse)
        args.supervisor_stop_policy = 'continue'
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        self.assertEqual(muse[muse.index('--supervisor-stop-policy') + 1], 'continue')
        self.assertNotIn('--supervisor-stop-policy', baseline)
        args.max_supervisor_failures = 5
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        self.assertEqual(muse[muse.index('--max-supervisor-failures') + 1], '5')
        self.assertNotIn('--max-supervisor-failures', baseline)
        self.assertEqual(muse[muse.index('--planning-context') + 1], 'summary')
        self.assertNotIn('--planning-context', baseline)
        args.planning_context = 'recent'
        sam = run_command(args, 'muse_sam', 3, Path('/outputs/sam'))
        self.assertEqual(sam[sam.index('--planning-context') + 1], 'recent')
        self.assertEqual(sam[sam.index('--max-stop-continuations') + 1], '2')
        self.assertNotIn('--max-stop-continuations', baseline)
        args.max_stop_continuations = 0
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        self.assertEqual(muse[muse.index('--max-stop-continuations') + 1], '0')
        self.assertNotIn('--depth', muse)
        args.depth, args.depth_max_metres = True, 2.5
        baseline = run_command(args, 'baseline', 3, Path('/outputs/base'))
        muse = run_command(args, 'muse', 3, Path('/outputs/muse'))
        sam = run_command(args, 'muse_sam', 3, Path('/outputs/sam'))
        self.assertNotIn('--depth', baseline)
        self.assertNotIn('--depth-max-metres', baseline)
        for command in (muse, sam):
            self.assertIn('--depth', command)
            self.assertEqual(command[command.index('--depth-max-metres') + 1], '2.5')

    def test_two_condition_report_includes_paired_result(self):
        row = {'condition': 'baseline', 'status': 'completed', 'seed': 0, 'success': False,
               'original_goal': 'Arrange tea', 'steps': 20, 'wall_seconds': 10, 'api_calls': 0, 'sam_calls': 0,
               'scene_identity': {'state': 'same'}}
        report = summarize([row, {**row, 'condition': 'muse', 'success': True}], ('baseline', 'muse'))
        self.assertEqual(set(report['conditions']), {'baseline', 'muse'})
        self.assertEqual(report['paired_seeds'][0]['success'], {'baseline': False, 'muse': True})
        self.assertTrue(report['paired_seeds'][0]['scene_identities_match'])

    def test_errors_are_not_treated_as_completed_failures(self):
        row = {'condition': 'baseline', 'status': 'completed', 'seed': 0, 'success': False,
               'original_goal': 'Open the cabinet', 'steps': 600, 'wall_seconds': 100,
               'api_calls': 0, 'sam_calls': 0}
        report = summarize([row, {'condition': 'muse', 'status': 'error', 'seed': 0}])
        self.assertEqual(report['conditions']['baseline']['success_rate'], 0)
        self.assertIsNone(report['conditions']['muse']['success_rate'])
        self.assertEqual(report['conditions']['muse']['errors'], 1)
        self.assertEqual(report['paired_seeds'], [])

    def test_sam_detection_and_subtask_claims_do_not_override_task_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'result.json').write_text(json.dumps({'success': False}))
            events = [
                {'event': 'decision', 'status': 'complete'},
                {'event': 'transition_blocked'},
                {'event': 'supervisor_response', 'wall_seconds': 2, 'sam': {'boxes': [{'x1': 1}]}},
                {'event': 'supervisor_response', 'wall_seconds': 3, 'sam': {'boxes': []}},
            ]
            (path / 'trace.jsonl').write_text('\n'.join(json.dumps(event) for event in events))
            metrics = episode_metrics(path)
            self.assertFalse(metrics['success'])
            self.assertEqual(metrics['subtasks_marked_complete'], 1)
            self.assertEqual(metrics['blocked_handoffs'], 1)
            self.assertEqual(metrics['accepted_handoffs'], 0)
            self.assertEqual(metrics['sam_calls'], 2)
            self.assertEqual(metrics['sam_calls_with_boxes'], 1)
            self.assertEqual(metrics['supervisor_wall_seconds'], 5)

    def test_failed_supervisor_requests_contribute_to_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'result.json').write_text(json.dumps({'success': False}))
            events = [
                {'event': 'supervisor_request_failed', 'wall_seconds': 7},
                {'event': 'supervisor_response', 'wall_seconds': 3, 'audit': [
                    {'role': 'planner'}, {'role': 'selective_verification'}]},
            ]
            (path / 'trace.jsonl').write_text('\n'.join(json.dumps(event) for event in events))
            metrics = episode_metrics(path)
            self.assertEqual(metrics['supervisor_wall_seconds'], 10)
            self.assertEqual(metrics['supervisor_failed_requests'], 1)
            self.assertEqual(metrics['planner_calls'], 1)
            self.assertEqual(metrics['verification_calls'], 1)
