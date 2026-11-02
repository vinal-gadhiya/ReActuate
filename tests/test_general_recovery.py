"""Task-general recovery feedback for manipulation, fixtures and travel; Thor only."""

import json
import unittest

import numpy as np

from core.episode_context import EpisodeContext
from inference.supervisor import Supervisor
from tests.test_agent import decision, step


class GeneralRecoveryTests(unittest.TestCase):
    CASES = (
        ('Open the cabinet.', 'cabinet', 'Unobstructed access to the cabinet interior.',
         'Door closed.', 'Door partly open but withdrawal path blocked.'),
        ('Close the blender lid.', 'blender lid', 'Lid seated on the blender.',
         'Lid on the counter.', 'Lid is tilted rather than seated.'),
        ('Turn on the microwave.', 'microwave', 'Microwave visibly operating.',
         'Microwave off.', 'Door appears unlatched and microwave remains off.'),
        ('Turn on the stove.', 'burner', 'Required burner visibly on.',
         'Required burner off.', 'Another burner is on while the required burner remains off.'),
        ('Turn on the sink faucet.', 'sink', 'Water flowing from the faucet.',
         'No water flowing.', 'No water flowing after lever contact.'),
        ('Slide the oven rack out.', 'oven rack', 'Rack extended to the required working position.',
         'Rack inside oven.', 'Rack only partially extended.'),
        ('Navigate to the dining counter.', 'dining counter', 'At the intended counter within working reach.',
         'Intended counter distant.', 'At another counter; intended destination still distant.'),
        ('Pour the milk into the bowl.', 'bowl', 'Milk transferred into the specified bowl.',
         'Bowl empty.', 'Bowl empty and liquid visible outside it.'),
    )

    def sample(self, instruction, target, criterion, before, after):
        journal = EpisodeContext()
        active = {**step('active', instruction), 'target': target, 'success_criteria': criterion}
        journal.observe({'event': 'initial_plan', 'goal': criterion, 'steps': [active]})
        for at, evidence in ((0, before), (16, after)):
            proposal = decision('inspect', '')
            proposal['evidence'] = evidence
            proposal['assessment'].update(grasp='empty', held_object='', phase='unknown', evidence=evidence)
            proposal['memory_updates'] = [{'name': target, 'kind': 'fixture', 'status': 'visible',
                                           'location': 'current workspace', 'evidence': evidence}]
            journal.observe({'event': 'decision', 'subtask': 'active', 'step': at, **proposal})
            if at == 0:
                journal.observe({'event': 'tool', 'subtask': 'active', 'tool': 'run_groot',
                                 'arguments': {'instruction': instruction, 'steps': 16},
                                 'step_before': 0, 'step_after': 16})
        return journal, active

    def test_every_action_retains_stage_effect_evidence_even_when_grasp_does_not_change(self):
        for instruction, target, criterion, before, after in self.CASES:
            with self.subTest(instruction=instruction):
                journal, _ = self.sample(instruction, target, criterion, before, after)
                review = journal.execution_feedback()['recent_outcome_reviews'][0]
                self.assertEqual(review['planned_target'], target)
                self.assertEqual(review['planned_success_criteria'], criterion)
                self.assertEqual(review['instruction'], instruction)
                self.assertEqual(review['evidence_before'], before)
                self.assertEqual(review['evidence_after'], after)
                self.assertEqual(review['reported_scene_updates'][0]['evidence'], after)
                self.assertFalse(review['assessment_changed'])
                self.assertEqual((review['step_before'], review['step_after'], review['review_step']), (0, 16, 16))
                self.assertNotIn('task_failed', review)
                self.assertNotIn('recovery_instruction', review)

    def test_general_feedback_reaches_the_existing_single_planner_without_extra_calls(self):
        calls = []

        class API:
            def response(self, model, content, instructions=None, schema=None):
                calls.append(json.loads(content[0]['text']))
                return decision('inspect', ''), {'model': model}

        supervisor = Supervisor(API(), 'muse', use_sam=False)
        for case in self.CASES:
            journal, active = self.sample(*case)
            before_calls = len(calls)
            reply = supervisor.handle('decide', {
                'step': active, 'api_calls_left': 1, 'decision_step_limit': 32,
                'capabilities': {'set_gripper': False},
                'execution_feedback': journal.execution_feedback(),
            }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
            self.assertEqual(len(calls), before_calls + 1)
            self.assertEqual(reply['api_calls'], 1)
            self.assertIsNone(reply['validation_error'])
            sent = calls[-1]['context']['execution_feedback']['recent_outcome_reviews'][0]
            self.assertEqual(sent['planned_success_criteria'], case[2])

    def test_replan_does_not_relabel_the_expected_effect_of_an_already_executed_action(self):
        journal = EpisodeContext()
        old = {**step('old', 'Close the lid.'), 'target': 'lid', 'success_criteria': 'Lid seated.'}
        journal.observe({'event': 'initial_plan', 'goal': 'Prepare the appliance.', 'steps': [old]})
        journal.observe({'event': 'tool', 'subtask': 'old', 'tool': 'run_groot',
                         'arguments': {'instruction': old['instruction'], 'steps': 16},
                         'step_before': 0, 'step_after': 16})
        journal.observe({'event': 'plan_replaced', 'step': 16, 'steps': [step('new', 'Open the cabinet.')],
                         'reason': 'An unmet prerequisite was identified.'})
        journal.observe({'event': 'decision', 'subtask': 'new', 'step': 16, **decision('inspect', '')})
        review = journal.execution_feedback()['recent_outcome_reviews'][0]
        self.assertEqual(review['planned_target'], 'lid')
        self.assertEqual(review['planned_success_criteria'], 'Lid seated.')


if __name__ == '__main__':
    unittest.main()
