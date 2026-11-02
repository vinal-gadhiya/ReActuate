"""Awomo wire/session regressions. Run inside Thor containers; no model weights needed."""

from copy import deepcopy
from http.server import ThreadingHTTPServer
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from core.awomo import AwomoAdapter, AwomoHistory
from core.patchwam import CAMERAS, ACTION_LAYOUT, LANGUAGE
from core.instructions import PolicyInstructionAdapter
from core.policy_config import resolve_policy_execution, validate_policy_interval
from core.policy_http import PolicyClient
from inference.serve import make_handler
from sim.evaluate import make_policy_observation
from sim.run_agent import EpisodeTools
from tests.test_patchwam import observation


class OfficialPolicyDouble:
    """Exercise the upstream episode-instruction latch that our bridge adapts."""

    def __init__(self):
        self.episode = None
        self.prompt = SimpleNamespace(_cache={'old': 'tokens'})
        self.calls = []
        self.instructions = []
        self.output = np.arange(16 * 12, dtype=np.float32).reshape(16, 12) / 10

    def predict(self, request):
        self.calls.append(deepcopy(request))
        if request['replan_index'] == 0:
            self.episode = {'id': request['session_id'], 'instruction': request['instruction'], 'frames': {}}
        for at, frame in zip(request['observation_steps'], request['camera_frames']):
            self.episode['frames'][at] = frame.copy()
        self.instructions.append(self.episode['instruction'])
        return {'actions': self.output.copy()}


def inputs(engine, step=0, episode='episode-a', instruction='Open the cabinet.'):
    result = make_policy_observation(observation(), engine.metadata, instruction)
    steps = [0] if step == 0 else list(range(step - 12, step + 1, 4))
    result.update({'__episode_id': np.asarray(episode), '__control_step': np.asarray(step),
                   '__episode_seed': np.asarray(7), '__frame_steps': np.asarray(steps),
                   '__camera_frames': np.stack([np.stack([observation()[f'video.{camera}'] for camera in CAMERAS])
                                                for _ in steps])})
    return result


class AwomoTests(unittest.TestCase):
    def test_official_actions_are_only_rearranged_and_state_is_in_dataset_order(self):
        official = OfficialPolicyDouble()
        engine = AwomoAdapter(official)
        actions = engine.predict(inputs(engine))
        for name, indices in (('base_motion', [7, 8, 9, 10]), ('control_mode', [11]),
                              ('end_effector_position', [0, 1, 2]),
                              ('end_effector_rotation', [3, 4, 5]), ('gripper_close', [6])):
            np.testing.assert_array_equal(actions[f'action.{name}'], official.output[:, indices])
        np.testing.assert_array_equal(official.calls[0]['state'], np.arange(16))
        self.assertEqual(official.calls[0]['seed'], 70000)

    def test_muse_instruction_change_takes_effect_without_losing_visual_history(self):
        official = OfficialPolicyDouble()
        engine = AwomoAdapter(official)
        engine.predict(inputs(engine))
        text = 'Pick the mug from the counter and place it on the tray.'
        engine.predict(inputs(engine, 16, instruction=text))
        self.assertEqual(official.instructions, ['Open the cabinet.', text])
        self.assertEqual(official.calls[1]['replan_index'], 1)
        self.assertEqual(sorted(official.episode['frames']), [0, 4, 8, 12, 16])
        self.assertEqual(official.calls[1]['seed'], 70001)
        action = {'intent': 'pick and place', 'object': 'mug', 'destination': 'tray', 'instruction': text}
        self.assertEqual(PolicyInstructionAdapter('awomo-0.5i').render(action), text)

    def test_new_episode_clears_visual_and_prompt_state_and_old_sessions_are_rejected(self):
        official = OfficialPolicyDouble()
        engine = AwomoAdapter(official)
        engine.predict(inputs(engine))
        engine.predict(inputs(engine, 16))
        official.prompt._cache['episode-a'] = 'cached'
        engine.predict(inputs(engine, episode='episode-b', instruction='Close the cabinet.'))
        self.assertEqual(official.calls[-1]['replan_index'], 0)
        self.assertEqual(sorted(official.episode['frames']), [0])
        self.assertEqual(official.prompt._cache, {})
        with self.assertRaises(ValueError):
            engine.predict(inputs(engine, 32, episode='episode-a'))

    def test_gaps_stale_predictions_and_mismatched_pixels_fail_before_inference(self):
        for change in ('gap', 'pixels', 'duplicate'):
            official = OfficialPolicyDouble()
            engine = AwomoAdapter(official)
            engine.predict(inputs(engine))
            candidate = inputs(engine, 16)
            if change == 'gap':
                candidate['__frame_steps'] = np.asarray([4, 12, 16])
            elif change == 'pixels':
                candidate['__camera_frames'][-1, 0, 0, 0] = 255
            else:
                candidate = inputs(engine)
            with self.subTest(change=change), self.assertRaises(ValueError):
                engine.predict(candidate)
            self.assertEqual(len(official.calls), 1)

    def test_history_uses_actual_four_step_sensor_samples_and_can_refresh_same_pose(self):
        history = AwomoHistory('episode', 7)
        sensors = observation()
        history.observe(0, sensors)
        history.predicted(0)
        for at in range(1, 17):
            sensors = {**observation(), **{f'video.{camera}': np.full((256, 256, 3), at, np.uint8) for camera in CAMERAS}}
            history.observe(at, sensors)
        payload = history.payload(16)
        self.assertEqual(payload['__frame_steps'].tolist(), [4, 8, 12, 16])
        self.assertEqual(payload['__camera_frames'][:, 0, 0, 0, 0].tolist(), [4, 8, 12, 16])
        sensors['video.robot0_agentview_left'][:] = 99
        history.refresh(16, sensors)
        self.assertEqual(history.payload(16)['__camera_frames'][-1, 0, 0, 0, 0], 99)
        self.assertEqual(history.payload(16)['__frame_steps'].tolist(), [4, 8, 12, 16])

    def test_chunk_cadence_is_enforced_but_final_partial_episode_is_allowed(self):
        metadata = AwomoAdapter(OfficialPolicyDouble()).metadata
        self.assertEqual(resolve_policy_execution(metadata, expected_policy='awomo-0.5i'), 16)
        with self.assertRaises(ValueError):
            resolve_policy_execution(metadata, 8)
        with self.assertRaises(ValueError):
            validate_policy_interval(metadata, 0, 8, 4000)
        validate_policy_interval(metadata, 0, 32, 4000)
        validate_policy_interval(metadata, 16, 5, 21)

    def test_real_episode_tools_supply_temporal_wire_context_across_instruction_changes(self):
        official = OfficialPolicyDouble()
        engine = AwomoAdapter(official)
        applied, requests = [], []

        class Env:
            action_space = SimpleNamespace(spaces={f'action.{name}': SimpleNamespace(shape=(width,))
                                                   for name, width in ACTION_LAYOUT})

            def step(self, action):
                applied.append(action)
                return observation(), 0, False, False, {'success': False}

        class Client:
            def predict(self, request):
                requests.append(deepcopy(request))
                return {**engine.predict(request), '__request_id': request['__request_id']}

        tools = EpisodeTools(Env(), observation(), Client(), engine.metadata,
                             SimpleNamespace(append_data=lambda frame: None), 21, 16, policy_seed=7)
        tools.run_groot('Open the cabinet.', 16)
        tools.run_groot('Pick up the mug.', 5)
        self.assertEqual(tools.steps, 21)
        self.assertEqual([request['__frame_steps'].tolist() for request in requests], [[0], [4, 8, 12, 16]])
        self.assertEqual(official.instructions, ['Open the cabinet.', 'Pick up the mug.'])
        np.testing.assert_array_equal(applied[16]['action.end_effector_position'], official.output[0, :3])

    def test_http_server_accepts_declared_history_keys_and_returns_matching_request(self):
        engine = AwomoAdapter(OfficialPolicyDouble())
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = inputs(engine)
            request['__request_id'] = np.asarray('episode-a:0')
            response = PolicyClient(f'http://127.0.0.1:{server.server_port}').predict(request)
            self.assertEqual(response.pop('__request_id').item(), 'episode-a:0')
            self.assertEqual(response['action.base_motion'].shape, (16, 4))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
