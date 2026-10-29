"""Run in the supervisor container; no live API calls or simulator required."""

import json
from types import SimpleNamespace
import unittest

import numpy as np

from core.agent import parse_program, validate_decision, validate_plan
from inference.meta_api import output_text
from inference.supervisor import Supervisor, sam_boxes
from sim.evaluate import make_policy_observation
from sim.run_agent import EpisodeTools, handoff_context, run_subtasks


def step(identifier, instruction):
    return {'id': identifier, 'instruction': instruction, 'target': 'cabinet',
            'success_criteria': 'Door is visibly open', 'max_steps': 10}


def decision(status='continue', program='run_groot(instruction="open cabinet", steps=4)', replacement=None):
    return {'status': status, 'evidence': 'Door remains closed in the current image.',
            'program': program, 'replacement_steps': replacement or [],
            'handoff': {'readiness': 'ready' if status == 'complete' else 'not_applicable',
                        'evidence': 'Shelf access is visible.' if status == 'complete' else 'No transition requested.'},
            'assessment': {'phase': 'unknown', 'held_object': '', 'grasp': 'uncertain',
                           'destination': '', 'information_needed': '', 'evidence': 'No supported grasp claim.'},
            'memory_updates': [], 'perception': {'target': '', 'camera': '', 'mode': 'detect'}}


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

    def navigate(self, fixture, steps):
        self.run_groot(f'Navigate to the {fixture}.', steps)


class AgentTests(unittest.TestCase):
    def test_reorder_preserves_useful_grasp_completed_prefix_and_all_remaining_goals(self):
        for deferred, held, destination, finish in (
            ('kettle', 'mug', 'tray', 'close cabinet'),
            ('apple', 'banana', 'fruit bowl', 'close drawer'),
            ('spatula', 'spoon', 'utensil holder', 'finish organizing'),
        ):
            with self.subTest(deferred=deferred, held=held):
                self._check_reorder(deferred, held, destination, finish)

    def _check_reorder(self, deferred, held, destination, finish):
        from core.transport_control import TransportControl

        tools, events, observed_plans = FakeTools(), [], []
        tools.transport_control = TransportControl()
        original = [step('access', 'open access'), step('deferred', f'transfer {deferred}'),
                    step('held', f'transfer {held}'), step('finish', finish)]
        placement = f'Place the held {held} at the {destination}'
        replacement = [step('held-first', placement),
                       step('deferred-later', f'transfer {deferred}'), step('finish-last', finish)]
        reorder = decision('replan', '', replacement)
        reorder['assessment'] = {'phase': 'carry', 'grasp': 'supported', 'held_object': held,
                                 'destination': destination, 'information_needed': '',
                                 'evidence': 'Required object lifted while another target remains unlocalized.'}
        place_held = decision(program=f'run_groot(instruction={placement!r}, steps=4)')
        place_held['assessment'] = dict(reorder['assessment'])
        replies = iter([decision('complete', ''), reorder, place_held, decision('stop', '')])

        def decide(plan, index, *args):
            observed_plans.append(([item['id'] for item in plan], index, tools.steps))
            return next(replies)

        reason = run_subtasks(tools, original, 'agent', decide, events.append)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(observed_plans[2], (['access', 'held-first', 'deferred-later', 'finish-last'], 1, 0))
        self.assertEqual(tools.steps, 4)
        self.assertEqual(tools.instructions, [placement])
        self.assertEqual(tools.transport_control.held_object, held)
        self.assertEqual([event['subtask'] for event in events if event['event'] == 'subtask_transition'], ['access'])

    def test_invalid_model_program_returns_feedback_without_motion_then_recovers(self):
        tools, events = FakeTools(), []
        invalid = decision(program='env.step()')
        replies = iter([invalid, decision(), decision(status='stop', program='')])
        reason = run_subtasks(tools, [step('1', 'open cabinet')], 'agent',
                              lambda *args: next(replies), events.append)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.steps, 4)
        self.assertEqual(len([event for event in events if event['event'] == 'invalid_decision']), 1)

    def test_repeated_invalid_programs_stop_with_no_motion(self):
        tools, events = FakeTools(), []
        reason = run_subtasks(tools, [step('1', 'open cabinet')], 'agent',
                              lambda *args: decision(program='env.step()'), events.append)
        self.assertEqual(reason, 'invalid_decision_limit')
        self.assertEqual(tools.steps, 0)
        self.assertEqual(len(events), 3)

    def test_invalid_supervisor_proposal_preserves_api_usage_and_audit(self):
        invalid = decision(program='env.step()')

        class FakeAPI:
            def response(self, *args, **kwargs):
                return invalid, {'model': 'test-model', 'usage': {'output_tokens': 10}}

        reply = Supervisor(FakeAPI(), 'test-model').handle('decide', {
            'api_calls_left': 1, 'decision_step_limit': 40, 'capabilities': {'set_gripper': False},
            'step': step('1', 'open cabinet'),
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertIs(reply['result'], invalid)
        self.assertTrue(reply['validation_error'])
        self.assertEqual(reply['api_calls'], 1)
        self.assertEqual(reply['audit'][0]['usage']['output_tokens'], 10)

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

    def test_handoff_blocks_advancement_until_access_is_ready(self):
        tools, events = FakeTools(), []
        blocked = decision('complete', '')
        blocked['handoff'] = {'readiness': 'blocked', 'evidence': 'One panel is open but the target shelf is obstructed.'}
        replies = iter([
            blocked,
            decision(program='run_groot(instruction="open enough to access the shelf", steps=4)'),
            decision('complete', ''),
            decision(program='run_groot(instruction="pick lemon", steps=4)'),
            decision('stop', ''),
        ])
        reason = run_subtasks(tools, [step('access', 'open fridge'), step('pick', 'pick lemon')],
                              'agent', lambda *args: next(replies), events.append)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.instructions, ['open enough to access the shelf', 'pick lemon'])
        transitions = [event for event in events if event['event'] == 'subtask_transition']
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]['step'], 4)
        self.assertEqual(sum(event['event'] == 'transition_blocked' for event in events), 1)

    def test_temporary_navigation_keeps_transfer_pending(self):
        tools, events = FakeTools(), []
        plan = [step('transfer', 'Pick up the lemon wedge and place it in the glass of lemonade.')]
        navigation = decision(program='navigate(fixture="sink", steps=4)')
        navigation['assessment']['information_needed'] = 'Find the destination glass near the observed sink.'
        navigation['memory_updates'] = [{'name': 'sink', 'kind': 'fixture', 'location': 'left camera',
                                         'status': 'visible', 'evidence': 'Basin and faucet visible.'}]
        replies = iter([
            navigation,
            decision(program='run_groot(instruction="Pick up the lemon wedge and place it in the glass of lemonade.", steps=4)'),
            decision('stop', ''),
        ])
        reason = run_subtasks(tools, plan, 'agent', lambda *args: next(replies), events.append)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.instructions, ['Navigate to the sink.', plan[0]['instruction']])
        self.assertEqual(plan[0]['id'], 'transfer')
        self.assertFalse(any(event['event'] == 'subtask_transition' for event in events))
        self.assertFalse(tools.success)

    def test_handoff_context_retains_old_evidence_after_recent_history_rolls_off(self):
        accepted = {'event': 'subtask_transition', 'subtask': 'access', 'step': 20,
                    'completion_evidence': 'Shelf exposed.',
                    'handoff': {'readiness': 'ready', 'evidence': 'Arm clear of door.'}}
        history = [accepted] + [{'event': 'tool', 'step_after': index} for index in range(50)]
        plan = [step('access', 'open fridge'), step('pick', 'pick lemon'), step('place', 'place lemon')]
        context = handoff_context(plan, 1, history)
        self.assertEqual(context['accepted_handoffs'], [accepted])
        self.assertEqual(context['next_step']['id'], 'place')
        self.assertIn('not proof', context['handoff_rule'])

    def test_duplicate_plan_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_plan({'steps': [step('a', 'open'), step('a', 'close')]})

    def test_episode_budget_allows_recovery_past_planned_subtask_ceiling(self):
        tools, limits = FakeTools(), []

        def decide(plan, index, spent, limit):
            limits.append(limit)
            return decision(program=f'run_groot(instruction="open cabinet", steps={limit})')

        reason = run_subtasks(tools, [step('a', 'open cabinet')], 'agent', decide,
                              lambda event: None, decision_steps=8, subtask_budget_mode='episode')
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 30)
        self.assertEqual(limits, [8, 8, 8, 6])

    def test_plan_budget_preserves_zero_motion_completion_or_replan_checkpoint(self):
        tools, limits = FakeTools(), []

        def decide(plan, index, spent, limit):
            limits.append(limit)
            return decision('stop', '') if limit == 0 else decision(
                program=f'run_groot(instruction="open cabinet", steps={limit})')

        reason = run_subtasks(tools, [step('a', 'open cabinet')], 'agent', decide,
                              lambda event: None, decision_steps=8)
        self.assertEqual(reason, 'supervisor_stop')
        self.assertEqual(tools.steps, 10)
        self.assertEqual(limits, [8, 2, 0])

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

    def test_catalog_is_sent_to_planning_and_execution_without_mutating_client_context(self):
        requests = []

        class FakeAPI:
            def response(self, model, content, instructions=None, schema=None):
                requests.append((json.loads(content[0]['text']), instructions))
                result = {'steps': [step('a', 'open fridge')]} if len(requests) == 1 else decision()
                return result, {'model': model}

        supervisor = Supervisor(FakeAPI(), 'test-model')
        context = {'api_calls_left': 2, 'simulator_context': {'policy_reference': {
            'instruction_catalog': {'client_copy': True}, 'registered_atomic_task_names': ['OpenFridge']}},
            'step': step('a', 'open fridge'), 'decision_step_limit': 40,
            'capabilities': {'set_gripper': False}}
        images = {'current:camera': np.zeros((4, 6, 3), dtype=np.uint8)}
        supervisor.handle('plan', context, images)
        supervisor.handle('decide', context, images)
        for payload, prompt in requests:
            tasks = payload['atomic_instruction_catalog']['tasks']
            self.assertEqual(len(tasks), 65)
            self.assertTrue(any(task['name'] == 'PickPlaceCounterToCabinet' for task in tasks))
            self.assertNotIn('instruction_catalog', payload['context']['simulator_context']['policy_reference'])
            self.assertIn('Do not decompose these into separate reach', prompt)
            self.assertIn('Reorder independent unfinished subgoals', prompt)
            self.assertIn('replacement_steps replaces the entire UNFINISHED suffix', prompt)
            self.assertIn('If a held object contributes to ANY unfinished goal', prompt)
            self.assertIn('Do not infer that objectives commute', prompt)
        self.assertEqual(context['simulator_context']['policy_reference']['instruction_catalog'], {'client_copy': True})

    def test_sam_override_controls_calls_and_forwards_boxes_to_muse(self):
        calls = []

        class FakeAPI:
            def response(self, model, content, instructions=None, schema=None):
                calls.append((model, content))
                if model == 'sam-3.1':
                    return '<|box;x1=1;y1=1;x2=4;y2=3;w=6;h=4|>', {'model': model}
                return decision(), {'model': model}

        supervisor = Supervisor(FakeAPI(), 'test-model', use_sam=False)
        context = {'api_calls_left': 3, 'use_sam': True, 'step': step('a', 'open cabinet'),
                   'decision_step_limit': 40, 'capabilities': {'set_gripper': False}}
        images = {'current:camera': np.ones((4, 6, 3), dtype=np.uint8),
                  'before:camera': np.zeros((4, 6, 3), dtype=np.uint8)}
        reply = supervisor.handle('decide', context, images)
        self.assertEqual(reply['api_calls'], 2)
        self.assertEqual([call[0] for call in calls], ['sam-3.1', 'test-model'])
        sent = json.loads(calls[-1][1][0]['text'])
        self.assertEqual(sent['sam']['boxes'][0]['x1'], 1)
        self.assertEqual(sum(part['type'] == 'input_image' for part in calls[-1][1]), 2)
        calls.clear()
        reply = supervisor.handle('decide', {**context, 'use_sam': False}, images)
        self.assertEqual(reply['api_calls'], 1)
        self.assertIsNone(reply['sam'])
        self.assertEqual([call[0] for call in calls], ['test-model'])
        calls.clear()
        reply = supervisor.handle('decide', {**context, 'api_calls_left': 1}, images)
        self.assertIsNone(reply['sam'])
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
