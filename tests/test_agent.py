"""Run in the supervisor container; no live API calls or simulator required."""

import json
from types import SimpleNamespace
import unittest

import numpy as np

from core.agent import parse_program, validate_decision, validate_plan
from inference.meta_api import output_text
from inference.supervisor import Supervisor, sam_boxes
from sim.evaluate import make_policy_observation
from sim.run_agent import EpisodeTools, run_subtasks


def step(identifier, instruction):
    return {'id': identifier, 'instruction': instruction, 'target': 'cabinet',
            'success_criteria': 'Door is visibly open', 'max_steps': 10}


def decision(status='continue', program='run_groot(instruction="open cabinet", steps=4)', replacement=None):
    return {'status': status, 'evidence': 'Door remains closed in the current image.',
            'program': program, 'replacement_steps': replacement or []}


class FakeTools:
    def __init__(self):
        self.steps, self.max_steps = 0, 30
        self.allow_gripper, self.success, self.ended = False, False, False
        self.instructions = []

    @property
    def done(self):
        return self.success or self.steps >= self.max_steps

    def run_groot(self, instruction, steps):
        self.instructions.append(instruction)
        self.steps += min(steps, self.max_steps - self.steps)


class AgentTests(unittest.TestCase):
    def test_program_validation_rejects_arbitrary_code_and_budget_overruns(self):
        for code in (
            'import os', 'env.reset()', 'while True: pass', 'x = 1',
            'run_groot(instruction="x", steps=100)',
            'run_groot(instruction="x", steps=True)',
            'run_groot(instruction="x", steps=4 + 4)',
            'set_gripper(closed=False, steps=5)',
        ):
            with self.subTest(code=code), self.assertRaises((ValueError, SyntaxError)):
                parse_program(code, max_steps=40)
        commands = parse_program('set_gripper(closed=False, steps=5)\nrun_groot(instruction="regrasp", steps=8)', 20, True)
        self.assertEqual(len(commands), 2)

    def test_completion_cannot_include_hidden_motion(self):
        with self.assertRaises(ValueError):
            validate_decision(decision(status='complete'))

    def test_instruction_override_preserves_original_observation(self):
        original = {'annotation.task': 'original composite goal'}
        metadata = {'modalities': {'language': {'modality_keys': ['annotation.task']}}}
        converted = make_policy_observation(original, metadata, instruction='open cabinet')
        self.assertEqual(converted['annotation.task'].tolist(), ['open cabinet'])
        self.assertEqual(original['annotation.task'], 'original composite goal')

    def test_instruction_changes_discard_unused_action_chunks(self):
        observation = {
            'video.camera': np.zeros((4, 6, 3), dtype=np.uint8),
            'state.arm': np.zeros(2), 'annotation.task': 'original task',
        }
        metadata = {
            'camera_height': 4, 'camera_width': 6, 'action_horizon': 4,
            'modalities': {
                'video': {'modality_keys': ['video.camera']},
                'state': {'modality_keys': ['state.arm']},
                'language': {'modality_keys': ['annotation.task']},
                'action': {'modality_keys': ['action.arm']},
            },
        }
        applied, instructions = [], []

        class Env:
            action_space = SimpleNamespace(spaces={'action.arm': SimpleNamespace(shape=(2,))})

            def step(self, action):
                applied.append(action)
                return dict(observation), 0, False, False, {'success': False}

        class Policy:
            def predict(self, inputs):
                instructions.append(inputs['annotation.task'].item())
                return {'action.arm': np.zeros((4, 2)), '__request_id': inputs['__request_id']}

        tools = EpisodeTools(Env(), observation, Policy(), metadata,
                             SimpleNamespace(append_data=lambda frame: None), 10, 4)
        tools.run_groot('open cabinet', 3)
        tools.run_groot('place mug inside', 2)
        self.assertEqual(len(applied), 5)
        self.assertEqual(instructions, ['open cabinet', 'place mug inside'])
        self.assertEqual(observation['annotation.task'], 'original task')

    def test_sequence_switches_in_place_and_does_not_claim_task_success(self):
        tools, events = FakeTools(), []
        replies = iter([
            decision(program='run_groot(instruction="open cabinet", steps=4)'),
            decision('complete', ''),
            decision(program='run_groot(instruction="place mug inside", steps=3)'),
            decision('complete', ''),
        ])
        reason = run_subtasks(tools, [step('a', 'open cabinet'), step('b', 'place mug inside')],
                              'agent', lambda *args: next(replies), events.append)
        self.assertEqual(tools.instructions, ['open cabinet', 'place mug inside'])
        self.assertEqual(tools.steps, 7)
        self.assertEqual(reason, 'plan_exhausted_without_task_success')
        self.assertFalse(tools.success)

    def test_decision_and_replan_budgets(self):
        tools = FakeTools()
        reason = run_subtasks(tools, [step('a', 'open cabinet')], 'agent',
                              lambda *args: decision(), lambda event: None, max_decisions=1)
        self.assertEqual(reason, 'decision_budget')
        tools = FakeTools()
        reason = run_subtasks(tools, [step('a', 'open cabinet')], 'agent',
                              lambda *args: decision('replan', '', [step('b', 'retry')]),
                              lambda event: None, max_replans=0)
        self.assertEqual(reason, 'replan_budget')

    def test_duplicate_plan_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan({'steps': [step('a', 'open'), step('a', 'close')]})

    def test_meta_output_requires_completed_text(self):
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': '{"steps":[]}'}]}]}
        self.assertEqual(output_text(response), '{"steps":[]}')
        with self.assertRaises(RuntimeError):
            output_text({**response, 'status': 'incomplete'})

    def test_sam_boxes_validate_image_dimensions_and_coordinates(self):
        text = '<0f>0<|box;x1=1;y1=2;x2=5;y2=6;w=10;h=10|><|mask;x=0;y=0;data=...|>'
        self.assertEqual(len(sam_boxes(text, 10, 10)), 1)
        self.assertEqual(sam_boxes(text, 20, 20), [])

    def test_supervisor_uses_schema_and_respects_low_call_budget(self):
        calls = []

        class FakeAPI:
            def response(self, model, content, instructions=None, schema=None):
                calls.append((model, schema))
                return {'steps': [step('a', 'open cabinet')]}, {'usage': {}, 'output': 'fake'}

        supervisor = Supervisor(FakeAPI(), 'test-model', use_sam=True)
        reply = supervisor.handle('plan', {'api_calls_left': 1}, {'current:camera': np.zeros((4, 6, 3), dtype=np.uint8)})
        self.assertEqual(reply['api_calls'], 1)
        self.assertEqual(calls[0][0], 'test-model')


if __name__ == '__main__':
    unittest.main()
