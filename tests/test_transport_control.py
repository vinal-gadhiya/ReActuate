"""Action arbitration checks: execute in the supervisor container, no simulator required."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.agent import parse_program
from core.transport_control import ACTION_SHAPES, TransportControl
from sim.run_agent import EpisodeTools, run_subtasks
from tests.test_agent import FakeTools, decision, step


def assessment(phase='carry', grasp='supported'):
    return {'phase': phase, 'grasp': grasp, 'held_object': 'kettle', 'destination': 'tray',
            'information_needed': '', 'evidence': 'Kettle lifted and transported with gripper.'}


def tray():
    return {'name': 'tray', 'kind': 'object', 'status': 'visible', 'location': 'counter',
            'evidence': 'Tray below kettle in current views.'}


class TransportTests(unittest.TestCase):
    def test_default_episode_tools_passes_all_gr00t_actions_unchanged(self):
        recorded = []
        observation = {'video.camera': np.zeros((4, 4, 3), np.uint8)}
        def execute(action):
            recorded.append({key: value.copy() for key, value in action.items()})
            return observation.copy(), 0, False, False, {}
        tools = EpisodeTools(SimpleNamespace(step=execute), observation, None,
                             {'modalities': {'video': {'modality_keys': ['video.camera']}}},
                             SimpleNamespace(append_data=lambda frame: None), 50, 8)
        proposed = {key: np.full(shape, .7, np.float32) for key, shape in ACTION_SHAPES.items()}
        proposed['action.gripper_close'][:] = 0
        proposed['action.control_mode'][:] = 1
        tools.step(proposed)
        for key in proposed:
            np.testing.assert_array_equal(recorded[0][key], proposed[key])
        self.assertIsNone(tools.transport_control)
        self.assertFalse(tools.allow_gripper)
        self.assertFalse(tools.allow_vertical)

    def test_latch_blocks_open_and_survives_uncertainty_without_mutating_prediction(self):
        control = TransportControl()
        control.update(assessment(), [])
        command = {key: np.zeros(shape, np.float32) for key, shape in ACTION_SHAPES.items()}
        self.assertEqual(control.filter(command)['action.gripper_close'][0], 1)
        self.assertEqual(command['action.gripper_close'][0], 0)
        control.update(assessment(grasp='uncertain'), [])
        self.assertEqual(control.filter(command)['action.gripper_close'][0], 1)
        with self.assertRaisesRegex(ValueError, 'Release blocked'):
            control.authorize('set_gripper', {'closed': False})

    def test_release_needs_current_destination_and_place_phase(self):
        control = TransportControl()
        control.update(assessment(), [tray()])
        with self.assertRaises(ValueError):
            control.authorize('set_gripper', {'closed': False})
        control.update(assessment(phase='place'), [])
        with self.assertRaises(ValueError):
            control.authorize('move_vertical', {})
        control.update(assessment(phase='place'), [tray()])
        control.authorize('set_gripper', {'closed': False})
        action = {key: np.ones(shape, np.float32) for key, shape in ACTION_SHAPES.items()}
        action['action.gripper_close'][:] = 0
        result = control.filter(action, authorized_release=True)
        self.assertEqual(result['action.gripper_close'][0], 0)
        self.assertFalse(result['action.base_motion'].any())
        self.assertEqual(result['action.control_mode'][0], 0)
        self.assertFalse(result['action.end_effector_rotation'].any())
        control.released()
        self.assertFalse(control.context()['held_object_latch'])

    def test_loss_stops_before_more_motion(self):
        tools, events = FakeTools(), []
        tools.transport_control = TransportControl()
        tools.transport_control.update(assessment(), [])
        reply = decision()
        reply['assessment'] = assessment(grasp='lost')
        reason = run_subtasks(tools, [step('1', 'place kettle')], 'agent',
                              lambda *args: reply, events.append)
        self.assertEqual(reason, 'object_loss_reported')
        self.assertEqual(tools.steps, 0)

    def test_runner_rejects_transport_release_before_any_action(self):
        tools, events = FakeTools(), []
        tools.allow_gripper = True
        tools.transport_control = TransportControl()
        reply = decision(program='set_gripper(closed=False, steps=5)')
        reply['assessment'] = assessment()
        reply['memory_updates'] = [tray()]
        reason = run_subtasks(tools, [step('1', 'place kettle')], 'agent',
                              lambda *args: reply, events.append)
        self.assertEqual(reason, 'recovery_unavailable')
        self.assertEqual(tools.steps, 0)
        self.assertTrue(any(event['event'] == 'recovery_required' for event in events))

    def test_no_hidden_vertical_or_release_motion_in_program(self):
        self.assertEqual(parse_program('move_vertical(distance=-0.01, steps=5)', 8, True, True)[0][0], 'move_vertical')
        for source in ('move_vertical(distance=-0.03, steps=5)',
                       'move_vertical(distance=True, steps=5)',
                       'move_vertical(distance=-0.01, steps=5)\nset_gripper(closed=False, steps=3)'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_program(source, 10, True, True)
        with self.assertRaises(ValueError):
            parse_program('move_vertical(distance=-0.01, steps=5)', 8, True, False)

    def test_acquisition_open_is_blocked_even_when_model_claims_empty(self):
        commands = parse_program('set_gripper(closed=False, steps=3)\nrun_groot(instruction="pick kettle", steps=5)',
                                 8, True, True)
        control = TransportControl()
        control.update(assessment(phase='acquire', grasp='empty'), [])
        with self.assertRaisesRegex(ValueError, 'direct opening during acquisition'):
            control.authorize_program(commands)
        control.update(assessment(phase='place'), [tray()])
        with self.assertRaisesRegex(ValueError, 'only command'):
            control.authorize_program(commands)

    def test_wrong_target_grasp_is_still_protected(self):
        control = TransportControl()
        held = {**assessment(phase='acquire'), 'held_object': 'mug'}
        control.update(held, [])
        self.assertEqual(control.held_object, 'mug')
        with self.assertRaisesRegex(ValueError, 'Release blocked'):
            control.authorize('set_gripper', {'closed': False})

    def test_unlatched_handle_release_remains_available_during_access(self):
        control = TransportControl()
        control.update(assessment(phase='access'), [])
        control.authorize('set_gripper', {'closed': False})

    def test_vertical_units_base_stop_and_actual_gripper_release(self):
        recorded = []
        controller = SimpleNamespace(input_ref_frame='base', output_max=np.array([.05]*3), input_max=np.ones(3))
        kitchen = SimpleNamespace(robots=[SimpleNamespace(composite_controller=SimpleNamespace(
            part_controllers={'right': controller}))])
        spaces = {key: SimpleNamespace(shape=shape, high=np.ones(shape)) for key, shape in ACTION_SHAPES.items()}
        observation = {'video.camera': np.zeros((4, 4, 3), np.uint8)}

        def execute(action):
            recorded.append(action)
            return observation.copy(), 0, False, False, {}

        env = SimpleNamespace(unwrapped=SimpleNamespace(env=kitchen), action_space=SimpleNamespace(spaces=spaces), step=execute)
        metadata = {'modalities': {'video': {'modality_keys': ['video.camera']}}}
        with patch('sim.run_agent.gripper_supported', return_value=True):
            tools = EpisodeTools(env, observation, None, metadata, SimpleNamespace(append_data=lambda frame: None),
                                 50, 8, hybrid_control=True)
        tools.transport_control.update(assessment(phase='place'), [tray()])
        proposed = {key: np.ones(shape, np.float32) for key, shape in ACTION_SHAPES.items()}
        proposed['action.gripper_close'][:] = 0
        tools.step(proposed)
        np.testing.assert_allclose(recorded[-1]['action.end_effector_position'], [.04, .04, .04])
        self.assertFalse(recorded[-1]['action.base_motion'].any())
        self.assertEqual(recorded[-1]['action.control_mode'][0], 0)
        self.assertEqual(recorded[-1]['action.gripper_close'][0], 1)
        self.assertEqual(proposed['action.gripper_close'][0], 0)
        recorded.clear()
        tools.move_vertical(-.01, 5)
        self.assertEqual(len(recorded), 5)
        for action in recorded:
            np.testing.assert_allclose(action['action.end_effector_position'], [0, 0, -.04])
            self.assertEqual(action['action.gripper_close'][0], 1)
            self.assertFalse(action['action.base_motion'].any())
        with self.assertRaises(ValueError):
            tools.move_vertical(-.02, 2)
        tools.set_gripper(False, 5)
        self.assertEqual(recorded[-1]['action.gripper_close'][0], 0)
        self.assertFalse(tools.transport_control.held_object)

    def test_handles_are_not_automatically_latched(self):
        control = TransportControl()
        control.update(assessment(phase='access'), [])
        self.assertFalse(control.held_object)
