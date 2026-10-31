"""Task-independent evidence retention and execution review; container CPU tests."""

import json
import unittest

import numpy as np

from core.episode_context import EpisodeContext
from core.scene_memory import SceneMemory
from inference.supervisor import Supervisor
from tests.test_agent import decision, step


class OutcomeSupervisionTests(unittest.TestCase):
    def test_different_appearance_does_not_erase_prior_instance_evidence(self):
        memory = SceneMemory()
        proposal = decision()
        proposal['memory_updates'] = [{'name': 'target container', 'kind': 'object',
                                       'status': 'visible', 'location': 'beside refrigerator',
                                       'evidence': 'Transparent body with red lid in right view.'}]
        memory.update(proposal, 10, 1)
        proposal['memory_updates'][0].update(location='beside stove', evidence='Dark metal appliance in left view.')
        memory.update(proposal, 20, 2)
        for i in range(10):
            proposal['memory_updates'][0]['evidence'] = f'Dark appliance, observation {i}.'
            memory.update(proposal, 30 + i, 3 + i)
        entry = memory.context()['entries'][0]
        self.assertIn('red lid', entry['identity_anchor']['evidence'])
        self.assertEqual(entry['previous_location_claim']['location'], 'beside refrigerator')
        self.assertEqual(entry['location'], 'beside stove')
        self.assertLessEqual(len(entry['observations']), 6)
        # Retaining a conflict does not invent which claim is correct.
        self.assertEqual(entry['status'], 'visible')
        self.assertNotIn('verified_identity', entry)

    def test_uncertain_candidate_does_not_become_an_identity_anchor(self):
        memory = SceneMemory()
        proposal = decision()
        proposal['memory_updates'] = [{'name': 'target', 'kind': 'object', 'status': 'uncertain',
                                       'location': 'behind drawer', 'evidence': 'Partial shape.'}]
        memory.update(proposal, 0, 1)
        self.assertIsNone(memory.context()['entries'][0]['identity_anchor'])
        proposal['memory_updates'][0].update(status='visible', evidence='Blue bowl fully visible.')
        memory.update(proposal, 1, 2)
        self.assertEqual(memory.context()['entries'][0]['identity_anchor']['step'], 1)
        snapshot = memory.context()
        snapshot['entries'][0]['identity_anchor']['evidence'] = 'mutated'
        self.assertIn('Blue bowl', memory.context()['entries'][0]['identity_anchor']['evidence'])
        self.assertEqual(SceneMemory().context()['entries'], [])

    def test_identity_prompt_history_has_global_budget(self):
        memory = SceneMemory()
        proposal = decision()
        for i in range(48):
            for j in range(7):
                proposal['memory_updates'] = [{'name': f'object {i}', 'kind': 'object', 'status': 'visible',
                                               'location': str(j) + 'x' * 490, 'evidence': 'y' * 500}]
                memory.update(proposal, i * 7 + j, i * 7 + j)
        context = memory.context()
        self.assertEqual(len(context['entries']), 48)
        self.assertGreater(context['identity_history_records_omitted'], 0)
        self.assertEqual(len(memory.entries[('object', 'object 0')]['observations']), 6)

    def record_decision(self, journal, at, proposal=None, subtask='goal'):
        journal.observe({'event': 'decision', 'step': at, 'subtask': subtask, **(proposal or decision())})

    def record_execution(self, journal, before, after, subtask='goal', instruction='Transfer the red container'):
        journal.observe({'event': 'tool', 'subtask': subtask, 'tool': 'run_groot',
                         'arguments': {'instruction': instruction}, 'step_before': before, 'step_after': after})

    def test_moving_retries_are_reviewed_across_subtask_renaming_without_claiming_failure(self):
        journal = EpisodeContext()
        self.record_decision(journal, 0)
        for i in range(4):
            self.record_execution(journal, 16 * i, 16 * (i + 1), subtask=f'replanned-{i}')
            self.record_decision(journal, 16 * (i + 1))
        feedback = journal.execution_feedback()
        attempts = feedback['instruction_attempts'][0]
        self.assertEqual(attempts['bursts'], 4)
        self.assertEqual(attempts['executed_steps'], 64)
        self.assertEqual(attempts['same_assessment_bursts'], 4)
        self.assertFalse(feedback['recent_outcome_reviews'][-1]['assessment_changed'])
        self.assertIn('not proof of failure', feedback['interpretation'])
        self.assertNotIn('task_failed', feedback)

    def test_changed_grasp_is_reported_as_claim_and_resets_unchanged_counter(self):
        journal = EpisodeContext()
        self.record_decision(journal, 0)
        self.record_execution(journal, 0, 16)
        self.record_decision(journal, 16)
        self.record_execution(journal, 16, 32)
        proposal = decision()
        proposal['assessment'].update(phase='carry', held_object='red container', grasp='supported')
        self.record_decision(journal, 32, proposal)
        feedback = journal.execution_feedback()
        review = feedback['recent_outcome_reviews'][-1]
        self.assertTrue(review['assessment_changed'])
        self.assertEqual(review['same_assessment_bursts'], 0)
        self.assertEqual(review['assessment_after']['held_object'], 'red container')
        self.assertIn('not verified progress', feedback['interpretation'])

    def test_outage_and_inspection_do_not_create_executions_or_clear_pending_review(self):
        journal = EpisodeContext()
        self.record_decision(journal, 0)
        self.record_execution(journal, 0, 16)
        journal.observe({'event': 'supervisor_unavailable', 'step': 16, 'error': 'connection reset'})
        self.record_execution(journal, 16, 32)
        self.record_decision(journal, 32, decision('inspect', ''))
        self.record_decision(journal, 32, decision('inspect', ''))
        feedback = journal.execution_feedback()
        self.assertEqual(feedback['instruction_attempts'][0]['bursts'], 2)
        self.assertEqual(len(feedback['recent_outcome_reviews']), 2)
        self.assertEqual(feedback['executions_awaiting_review'], [])
        feedback['instruction_attempts'].clear()
        self.assertEqual(len(journal.execution_feedback()['instruction_attempts']), 1)
        self.assertEqual(EpisodeContext().execution_feedback()['instruction_attempts'], [])

    def test_legacy_feedback_stays_in_audit_without_extra_api_calls(self):
        journal, requests = EpisodeContext(), []
        self.record_decision(journal, 0)
        self.record_execution(journal, 0, 16)
        self.record_decision(journal, 16)

        class API:
            def response(self, model, content, instructions=None, schema=None):
                requests.append((json.loads(content[0]['text']), instructions))
                return decision('stop', ''), {'model': model}

        response = Supervisor(API(), 'muse', use_sam=False).handle('decide', {
            'step': step('goal', 'Transfer red container'), 'decision_step_limit': 64,
            'api_calls_left': 1, 'use_sam': False, 'capabilities': {'set_gripper': False},
            'control_owner': 'patchwam', 'execution_feedback': journal.execution_feedback(),
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertEqual(response['api_calls'], 1)
        self.assertEqual(len(requests), 1)
        self.assertNotIn('execution_feedback', requests[0][0]['context'])
        audit = journal.execution_feedback()
        self.assertEqual(audit['instruction_attempts'][0]['executed_steps'], 16)
        self.assertFalse(audit['recent_outcome_reviews'][0]['assessment_changed'])
        self.assertIn('Repeated supported goal failures', requests[0][1])
        self.assertEqual(response['result']['status'], 'stop')


if __name__ == '__main__':
    unittest.main()
