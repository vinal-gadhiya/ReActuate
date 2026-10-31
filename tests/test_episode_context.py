"""Episode notebook checks; run in the supervisor container without live APIs."""

import json
import unittest

import numpy as np

from core.episode_context import EpisodeContext
from core.policy_http import RetryableSupervisorError
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


class EpisodeContextTests(unittest.TestCase):
    def journal(self):
        journal = EpisodeContext()
        journal.observe({'event': 'initial_plan', 'goal': 'Arrange tea',
                         'steps': [step('access', 'Open cabinet'), step('mug', 'Transfer mug')]})
        return journal

    def observe_decision(self, journal, proposal=None, at=0, subtask='mug'):
        journal.observe({'event': 'decision', 'step': at, 'subtask': subtask,
                         **(proposal or decision())})

    def test_completions_and_actual_attempts_survive_recent_window(self):
        journal = self.journal()
        journal.observe({'event': 'subtask_transition', 'step': 64,
                         'completed_step': step('access', 'Open cabinet'),
                         'completion_evidence': 'Shelf exposed in right camera.'})
        for i in range(20):
            journal.observe({'event': 'tool', 'subtask': 'mug', 'tool': 'run_groot',
                             'arguments': {'instruction': 'Place mug on tray', 'steps': 16},
                             'step_before': 64 + 16 * i, 'step_after': 80 + 16 * i})
        context = journal.context(384)
        self.assertEqual(len(context['recent_events']), 12)
        self.assertEqual(context['completed_claims'][0]['id'], 'access')
        self.assertIn('not environment proof', context['completed_claims'][0]['source'])
        self.assertEqual([s['id'] for s in context['unfinished_plan']], ['mug'])
        self.assertEqual(context['totals']['executed_steps'], 320)
        self.assertEqual(context['goal_attempts'][0]['bursts'], 20)
        self.assertEqual(context['goal_attempts'][0]['last_execution_step'], 384)

    def test_replan_changes_only_after_acceptance_and_does_not_count_motion(self):
        journal = self.journal()
        replacement = [step('place', 'Place held mug on tray')]
        self.observe_decision(journal, decision('replan', '', replacement))
        self.assertEqual(journal.context()['plan_revision'], 0)
        self.assertEqual(len(journal.context()['unfinished_plan']), 2)
        journal.observe({'event': 'plan_replaced', 'step': 70, 'steps': replacement,
                         'reason': 'Mug already held.'})
        replacement[0]['instruction'] = 'mutated'
        context = journal.context(70)
        self.assertEqual(context['plan_revision'], 1)
        self.assertEqual(context['unfinished_plan'][0]['instruction'], 'Place held mug on tray')
        self.assertEqual(context['totals']['executed_steps'], 0)

    def test_object_corrections_preserve_uncertainty_time_and_old_location(self):
        journal = self.journal()
        proposal = decision()
        proposal['memory_updates'] = [{'name': 'kettle', 'kind': 'object', 'location': 'left counter',
                                       'status': 'uncertain', 'evidence': 'Cropped dark shape.'}]
        self.observe_decision(journal, proposal, 0)
        self.observe_decision(journal, proposal, 16)
        self.assertEqual(len(journal.context()['object_location_changes']), 1)
        proposal['memory_updates'][0].update(location='right of tray', status='visible',
                                             evidence='White kettle handle in right camera.')
        self.observe_decision(journal, proposal, 32)
        change = journal.context()['object_location_changes'][-1]
        self.assertEqual(change['previous_location'], 'left counter')
        self.assertEqual(change['previous_status'], 'uncertain')
        self.assertEqual(change['location'], 'right of tray')
        self.assertEqual(change['step'], 32)

    def test_invalid_proposals_and_api_errors_add_feedback_without_claims_or_motion(self):
        journal = self.journal()
        journal.observe({'event': 'invalid_decision', 'step': 0, 'reason': 'Unavailable tool',
                         'proposal': {'memory_updates': [{'name': 'invented'}]}})
        journal.observe({'event': 'supervisor_unavailable', 'step': 0, 'error': 'connection reset'})
        context = journal.context()
        self.assertEqual(context['totals']['recovery_events'], 2)
        self.assertEqual(context['totals']['executed_steps'], 0)
        self.assertEqual(context['object_location_changes'], [])
        self.assertEqual(context['completed_claims'], [])
        self.assertEqual(context['recovery_feedback'][-1]['reason'], 'connection reset')

    def test_context_is_bounded_and_snapshot_cannot_mutate_journal(self):
        journal = self.journal()
        for i in range(80):
            self.observe_decision(journal, {**decision(), 'evidence': 'x' * 10000}, i)
            journal.observe({'event': 'tool', 'subtask': str(i), 'tool': 'run_groot',
                             'arguments': {'instruction': 'x' * 10000},
                             'step_before': i, 'step_after': i + 1})
            journal.observe({'event': 'subtask_transition', 'step': i,
                             'completed_step': step(str(i), 'x' * 10000),
                             'completion_evidence': 'x' * 10000})
        context = journal.context(80)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False)), journal.MAX_CHARS)
        self.assertGreater(context['omitted_summary_records'], 0)
        self.assertEqual(context['totals']['executed_steps'], 80)
        self.assertEqual(context['latest_decision']['step'], 79)
        context['unfinished_plan'].clear()
        self.assertEqual(len(journal.context()['unfinished_plan']), 2)
        self.assertEqual(EpisodeContext().context()['completed_claims'], [])
        self.assertEqual(EpisodeContext().context()['original_goal'], '')

    def test_runner_records_accepted_replan_and_preserves_physical_steps(self):
        journal, tools = self.journal(), FakeTools()
        replies = iter([decision('complete', ''), decision('replan', '', [step('place', 'Place mug')]),
                        decision(program='run_groot(instruction="Place mug", steps=4)'), decision('stop', '')])
        events = []

        def record(event):
            events.append(event)
            journal.observe(event)

        reason = run_subtasks(tools, journal.active_plan, 'agent', lambda *args: next(replies), record)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.steps, 4)
        self.assertEqual(tools.instructions, ['Place mug'])
        self.assertEqual(len([e for e in events if e['event'] == 'plan_replaced']), 1)
        context = journal.context(tools.steps)
        self.assertEqual(context['completed_claims'][0]['id'], 'access')
        self.assertEqual([s['id'] for s in context['unfinished_plan']], ['place'])
        self.assertEqual(context['totals']['executed_steps'], 4)

    def test_rejected_replan_keeps_active_plan(self):
        journal, tools = self.journal(), FakeTools()
        reason = run_subtasks(tools, journal.active_plan, 'agent',
                              lambda *args: decision('replan', '', [step('place', 'Place mug')]),
                              journal.observe, max_replans=0)
        self.assertEqual(reason, 'replan_budget')
        self.assertEqual(journal.context()['plan_revision'], 0)
        self.assertEqual([s['id'] for s in journal.context()['unfinished_plan']], ['access', 'mug'])
        self.assertEqual(tools.steps, 0)

    def test_api_outage_continuation_counts_actual_execution_without_new_visual_claims(self):
        journal, tools = self.journal(), FakeTools()
        replies = iter([decision(program='run_groot(instruction="Place mug", steps=4)'),
                        RetryableSupervisorError('connection reset'), decision('stop', '')])

        def decide(*args):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        run_subtasks(tools, [step('mug', 'Transfer mug')], 'agent', decide, journal.observe,
                     subtask_budget_mode='episode')
        context = journal.context(tools.steps)
        self.assertEqual(tools.instructions, ['Place mug', 'Place mug'])
        self.assertEqual(context['totals']['executed_steps'], 20)
        self.assertEqual(context['totals']['policy_bursts'], 2)
        self.assertEqual(context['totals']['decisions'], 2)
        self.assertEqual(context['completed_claims'], [])
        self.assertEqual(context['object_location_changes'], [])

    def test_supervisor_forwards_notebook_in_same_request_with_selected_policy_prompt(self):
        journal, requests = self.journal(), []

        class FakeAPI:
            def response(self, model, content, instructions=None, schema=None):
                requests.append((json.loads(content[0]['text']), instructions))
                return decision('stop', ''), {'model': model, 'usage': {}}

        Supervisor(FakeAPI(), 'test-model', use_sam=False).handle('decide', {
            'api_calls_left': 1, 'use_sam': False, 'step': step('mug', 'Transfer mug'),
            'decision_step_limit': 64, 'capabilities': {'set_gripper': False},
            'control_owner': 'patchwam', 'episode_context': journal.context(32),
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][0]['context']['original_goal'], 'Arrange tea')
        self.assertNotIn('episode_context', requests[0][0]['context'])
        self.assertIn('Backend: patchwam', requests[0][1])


if __name__ == '__main__':
    unittest.main()
