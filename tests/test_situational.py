"""Direct Awomo controller contract tests; execute in Thor containers only."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

import numpy as np

from core.policy_http import RetryableSupervisorError, SupervisorClient
from core.scene_memory import SceneMemory
from core.situational import run_situational, validate_situational, SITUATIONAL_SCHEMA
from inference.supervisor import Supervisor


def context(step=0, horizon=64):
    return {'control_owner': 'awomo-0.5i', 'instruction_mode': 'atomic',
            'original_goal': 'Gather the required tableware on the counter.',
            'observation_step': step, 'episode_steps_left': horizon - step,
            'decision_step_limit': min(32, horizon - step), 'policy_step_quantum': 16,
            'release_step_limit': min(16, horizon - step), 'api_calls_left': 200,
            'boundary_review_enabled': True,
            'observation_refs': [{'camera': 'current:video.robot0_agentview_left', 'step': step}]}


def action(instruction='Pick the mug from the cabinet.', steps=32, step=0):
    return {'instruction': instruction, 'steps': steps, 'evidence': 'Current labelled image supports this approach.',
            'observation_refs': context(step)['observation_refs'],
            'perception': {'target': '', 'camera': '', 'mode': 'detect'}}


class Tools:
    def __init__(self, horizon=64, success_after=None, end_after=None):
        self.steps = 0
        self.max_steps = horizon
        self.success = self.ended = False
        self.success_after, self.end_after = success_after, end_after
        self.last_diagnostics = {}
        self.instructions = []

    @property
    def done(self):
        return self.success or self.ended or self.steps >= self.max_steps

    def run_groot(self, instruction, steps):
        self.instructions.append(instruction)
        self.steps += steps
        self.success = self.success_after is not None and self.steps >= self.success_after
        self.ended = self.end_after is not None and self.steps >= self.end_after


class SituationalTests(unittest.TestCase):
    def test_http_client_routes_direct_action_operation(self):
        client = SupervisorClient('http://supervisor:8001')
        client._request = Mock(return_value=json.dumps({'result': action(), 'api_calls': 1}))
        response = client.ask('act', context(), {
            'current:video.robot0_agentview_left': np.zeros((8, 8, 3), dtype=np.uint8)})
        self.assertEqual(client._request.call_args.args[0], '/act')
        self.assertEqual(response['result']['instruction'], action()['instruction'])

    def supervise(self, result, use_sam=False):
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append((model, content, kwargs))
                if model == 'sam-3.1':
                    return '', {'model': model}
                return deepcopy(result), {'model': model}

        response = Supervisor(API(), 'muse', use_sam=use_sam).handle('act', context(), {
            'current:video.robot0_agentview_left': np.zeros((8, 8, 3), dtype=np.uint8)})
        return response, calls

    def test_single_controller_call_without_planning_or_any_verifier(self):
        reply, calls = self.supervise(action())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2]['schema'], SITUATIONAL_SCHEMA)
        self.assertNotIn('STAGE COMPLETION RELEASES COMMITMENT', calls[0][2]['instructions'])
        self.assertNotIn('success_criteria', json.dumps(calls[0][2]['schema']))
        self.assertIsNone(reply['validation_error'])
        for field in ('boundary_review', 'completion_confirmation', 'release_confirmation'):
            self.assertIsNone(reply[field])
        self.assertEqual(calls[0][1][1]['text'], 'current:video.robot0_agentview_left')

    def test_release_uses_no_separate_readiness_verifier(self):
        reply, calls = self.supervise(action('Open the gripper.', steps=16))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(reply['validation_error'])
        self.assertIsNone(reply['release_confirmation'])
        with self.assertRaises(ValueError):
            validate_situational(action('Open the gripper.', steps=32), context())

    def test_truncated_action_retry_keeps_schema_and_has_no_verification_calls(self):
        from inference.meta_api import MetaAPIError
        calls = []

        class API:
            max_output_tokens = 4096

            def response(self, model, content, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    error = MetaAPIError('max_output_tokens', retryable=True, kind='output_limit',
                                         allow_policy_continuation=False)
                    error.audit = [{'model': model}]
                    raise error
                return action(), {'model': model}

        reply = Supervisor(API(), 'muse').handle('act', context(), {
            'current:video.robot0_agentview_left': np.zeros((8, 8, 3), dtype=np.uint8)})
        self.assertEqual(reply['api_calls'], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['schema'], calls[1]['schema'])
        self.assertEqual(calls[1]['max_output_tokens'], 8192)
        self.assertEqual(reply['audit'][-1]['role'], 'situational_controller')
        self.assertIsNone(reply['completion_confirmation'])

    def test_optional_sam_is_evidence_and_not_a_boundary_review(self):
        reply, calls = self.supervise(action(), use_sam=True)
        self.assertEqual([call[0] for call in calls], ['sam-3.1', 'muse'])
        self.assertIsNotNone(reply['sam'])
        self.assertIsNone(reply['boundary_review'])

    def test_rejects_non_awomo_and_preserves_language_and_cadence(self):
        for change in ({'instruction': 'Do whatever is necessary.'}, {'steps': 17},
                       {'steps': 0}, {'steps': 48}, {'status': 'complete'},
                       {'observation_refs': [{'camera': 'current:invented', 'step': 0}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_situational({**action(), **change}, context())
        with self.assertRaises(ValueError):
            validate_situational(action(), {**context(), 'control_owner': 'patchwam'})
        validate_situational(action(steps=8, step=32), context(32, 40))
        validate_situational(action('Pick the mug from the cabinet and place it on the counter.'), context())

    def test_can_change_instruction_immediately_without_completion_or_recovery(self):
        tools, events, contexts = Tools(), [], []
        instructions = ['Pick the mug from the cabinet.', 'Open the cabinet doors.']

        def ask(operation, payload):
            self.assertEqual(operation, 'act')
            payload.update(context(tools.steps, tools.max_steps))
            contexts.append(deepcopy(payload))
            return action(instructions[len(contexts) - 1], step=tools.steps)

        reason = run_situational(tools, ask, events.append, decision_steps=32,
                                 history=events, memory=SceneMemory())
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.instructions, instructions)
        self.assertNotIn('step', contexts[0])
        self.assertFalse(any(event['event'] in ('subtask_transition', 'recovery_required') for event in events))
        self.assertEqual(contexts[1]['last_instruction'], instructions[0])

    def test_bad_response_or_connection_failure_never_replays_previous_motion(self):
        for failure in ('connection', 'invalid', 'budget'):
            tools, events = Tools(), []

            def ask(operation, payload):
                payload.update(context(tools.steps, tools.max_steps))
                if tools.steps == 0:
                    return action()
                if failure == 'connection':
                    raise RetryableSupervisorError('connection failed', 1)
                return None if failure == 'budget' else action(steps=17, step=tools.steps)

            reason = run_situational(tools, ask, events.append, decision_steps=32,
                                     history=events, memory=SceneMemory(), max_failures=2)
            self.assertEqual(tools.steps, 32)
            self.assertEqual(len(tools.instructions), 1)
            self.assertEqual(reason, 'api_budget' if failure == 'budget' else 'supervisor_failure_limit')

    def test_environment_success_and_end_stop_execution(self):
        for options, expected in (({'success_after': 32}, 'success'), ({'end_after': 32}, 'environment_end')):
            tools = Tools(**options)

            def ask(operation, payload):
                payload.update(context(tools.steps, tools.max_steps))
                return action(step=tools.steps)

            reason = run_situational(tools, ask, lambda event: None, decision_steps=32,
                                     history=[], memory=SceneMemory())
            self.assertEqual(reason, expected)
            self.assertEqual(tools.steps, 32)
