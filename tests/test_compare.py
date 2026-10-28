"""Comparison checks without a simulator, GPU, or live API requests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sim.compare import episode_metrics, run_command, summarize


class CompareTests(unittest.TestCase):
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
        self.assertNotIn('--plan', baseline)
        self.assertNotIn('--reference-scene', baseline)
        self.assertIn('--reference-scene', muse)
        self.assertIn('--reference-scene', sam)
        self.assertEqual(sam[sam.index('--plan') + 1], '/outputs/muse/episode-000/plan.json')
        for flag in ('--seed', '--max-steps', '--action-steps', '--decision-steps'):
            self.assertEqual(baseline[baseline.index(flag) + 1], sam[sam.index(flag) + 1])

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
