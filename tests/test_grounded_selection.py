"""Source-grounded instructions and dated pickup feedback; run on Thor."""

import json
import unittest

import numpy as np

from core.agent import pickup_request, validate_stage_program
from core.episode_context import EpisodeContext
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


class GroundedSelectionTests(unittest.TestCase):
    def review(self, journal, at, grasp='supported', held='blue mug', status='continue'):
        proposal = decision(status, '' if status != 'continue' else 'run_groot(instruction="Pick the red mug from the tray.", steps=4)')
        proposal['assessment'].update(grasp=grasp, held_object=held,
                                      evidence='Current external view shows the held object; compare the wrist view.')
        journal.observe({'event': 'decision', 'subtask': 'pick', 'step': at, **proposal})

    def execute(self, journal, before, after, instruction='Pick the red mug from the tray.'):
        journal.observe({'event': 'tool', 'subtask': 'pick', 'tool': 'run_groot',
                         'arguments': {'instruction': instruction, 'steps': after - before},
                         'step_before': before, 'step_after': after})

    def test_source_qualified_forms_are_accepted_without_rewriting(self):
        for instruction, target, source in (
                ('Pick up the croissant from the white plate.', 'croissant', 'white plate'),
                ('Pick the mug from the blue tray.', 'mug', 'blue tray'),
                ('Pick up the kettle from the counter.', 'kettle', 'counter'),
                ('Pick the lemon wedge from the fridge shelf.', 'lemon wedge', 'fridge shelf'),
                ('Pick up the bowl.', 'bowl', '')):
            commands = [('run_groot', {'instruction': instruction, 'steps': 16})]
            with self.subTest(instruction=instruction):
                validate_stage_program(commands)
                self.assertEqual(commands[0][1]['instruction'], instruction)
                self.assertEqual(pickup_request(instruction),
                                 {'requested_object': target, 'requested_source': source})
        self.assertIsNone(pickup_request('Place the mug on the tray.'))
        self.assertIsNone(pickup_request('Pick up the mug and place it on the tray.'))

    def test_duplicated_articles_are_rejected_instead_of_silently_rewritten(self):
        for instruction in ('Pick up the the mug from the tray.', 'Pick a a bowl from the counter.'):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, 'duplicated articles'):
                validate_stage_program([('run_groot', {'instruction': instruction, 'steps': 16})])

    def test_continuous_hold_occlusion_and_identity_rename_are_not_new_acquisitions(self):
        journal = EpisodeContext()
        self.execute(journal, 0, 16)
        self.review(journal, 16)
        self.execute(journal, 16, 32)
        self.review(journal, 32)
        self.execute(journal, 32, 48)
        self.review(journal, 48, 'uncertain', '')
        self.execute(journal, 48, 64)
        self.review(journal, 64, held='small blue cup')
        self.review(journal, 64, 'empty', '', 'inspect')
        self.review(journal, 64, held='blue mug')
        feedback = journal.execution_feedback()
        self.assertEqual(len(feedback['pickup_observations']), 4)
        self.assertEqual({record['held_claim_epoch'] for record in feedback['pickup_observations']}, {1})
        self.assertEqual([record['step_after'] for record in feedback['pickup_observations']], [16, 32, 48, 64])
        self.assertNotIn('wrong_object_count', feedback)

    def test_release_then_another_pickup_has_a_new_epoch_with_dated_source_evidence(self):
        journal = EpisodeContext()
        self.execute(journal, 0, 16)
        self.review(journal, 16)
        self.execute(journal, 16, 32, 'Place the blue mug on the counter.')
        self.review(journal, 32, 'empty', '')
        self.execute(journal, 32, 48)
        self.review(journal, 48)
        records = journal.execution_feedback()['pickup_observations']
        self.assertEqual([record['held_claim_epoch'] for record in records], [1, 2])
        self.assertTrue(all(record['requested_object'] == 'red mug' for record in records))
        self.assertTrue(all(record['requested_source'] == 'tray' for record in records))
        self.assertTrue(all(record['reported_held_object'] == 'blue mug' for record in records))
        records.clear()
        self.assertEqual(len(journal.execution_feedback()['pickup_observations']), 2)

    def test_feedback_and_derived_forms_reach_single_muse_call_with_no_new_tools(self):
        journal, calls = EpisodeContext(), []
        self.execute(journal, 0, 16)
        self.review(journal, 16)
        result = decision(program='run_groot(instruction="Pick the red mug from the white plate.", steps=4)')

        class API:
            def response(self, model, content, instructions=None, schema=None):
                calls.append((json.loads(content[0]['text']), instructions, schema))
                return result, {'model': model}

        reply = Supervisor(API(), 'muse', use_sam=False).handle('decide', {
            'api_calls_left': 1, 'use_sam': False, 'step': step('pick', 'Pick up the red mug.'),
            'decision_step_limit': 4, 'capabilities': {'set_gripper': False},
            'instruction_style': 'single_stage', 'execution_feedback': journal.execution_feedback(),
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertEqual(len(calls), 1)
        self.assertEqual(reply['api_calls'], 1)
        self.assertIsNone(reply['validation_error'])
        self.assertEqual(reply['result']['program'], result['program'])
        self.assertEqual(calls[0][0]['context']['execution_feedback']['pickup_observations'][0]['step_after'], 16)
        self.assertIn('Pick the {object} from the {source}.', calls[0][0]['derived_pickup_forms']['templates'])

    def test_explicit_staging_replan_preserves_target_work_and_observation_boundaries(self):
        tools, events = FakeTools(), []
        replacement = [step('stage', 'Place the blue mug on the counter.'),
                       step('retry', 'Pick the red mug from the white plate.'),
                       step('finish', 'Place the red mug on the tray.')]
        replies = [decision('replan', '', replacement)]
        for stage in replacement:
            replies.extend([decision(program=f'run_groot(instruction={stage["instruction"]!r}, steps=4)'),
                            decision('complete', '')])
        iterator = iter(replies)
        reason = run_subtasks(tools, [step('pick', 'Pick the red mug from the white plate.')], 'agent',
                              lambda *args: next(iterator), events.append, concise_instructions=True)
        self.assertEqual(reason, 'plan_exhausted_without_task_success')
        self.assertEqual(tools.steps, 12)
        self.assertEqual(tools.instructions, [stage['instruction'] for stage in replacement])
        self.assertEqual(len([event for event in events if event['event'] == 'subtask_transition']), 3)
        self.assertFalse(tools.success)


if __name__ == '__main__':
    unittest.main()
