"""PatchWAM adapter tests: supervisor container, no GPU, weights or Meta calls."""

from http.server import ThreadingHTTPServer
import threading
import unittest

import numpy as np

from core.patchwam import CAMERAS, LANGUAGE, PatchWAMAdapter
from core.policy_http import PolicyClient
from core.simulator_context import build_simulator_context
from inference.serve import make_handler
from sim.evaluate import make_policy_observation


class FakeAPPolicy:
    action_horizon, state_dim, image_keys = 16, 16, list(CAMERAS)

    def __init__(self):
        self.requests = []
        self.output = np.arange(16 * 12, dtype=np.float32).reshape(16, 12) / 10

    def act(self, images, state, instruction):
        self.requests.append((images, state.copy(), instruction))
        return self.output.copy()


def observation():
    # Deliberately use a different dict order than the dataset's packed layout.
    return {
        'state.gripper_qpos': np.array([14, 15], np.float32),
        'state.end_effector_rotation_relative': np.arange(10, 14, dtype=np.float32),
        'state.end_effector_position_relative': np.array([7, 8, 9], np.float32),
        'state.base_rotation': np.arange(3, 7, dtype=np.float32),
        'state.base_position': np.array([0, 1, 2], np.float32),
        **{f'video.{key}': np.full((256, 256, 3), i * 60, np.uint8) for i, key in enumerate(CAMERAS)},
        LANGUAGE: 'Place the mug on the tray.',
    }


class PatchWAMTests(unittest.TestCase):
    def test_official_dataset_order_and_raw_named_actions(self):
        policy = FakeAPPolicy()
        engine = PatchWAMAdapter(policy)
        sensors = observation()
        result = engine.predict(make_policy_observation(sensors, engine.metadata))
        images, state, instruction = policy.requests[-1]
        np.testing.assert_array_equal(state, np.arange(16, dtype=np.float32))
        self.assertEqual(instruction, sensors[LANGUAGE])
        for key in CAMERAS:
            np.testing.assert_array_equal(images[key], sensors[f'video.{key}'])
        np.testing.assert_array_equal(result['action.base_motion'], policy.output[:, :4])
        np.testing.assert_array_equal(result['action.control_mode'], policy.output[:, 4:5])
        np.testing.assert_array_equal(result['action.end_effector_position'], policy.output[:, 5:8])
        np.testing.assert_array_equal(result['action.end_effector_rotation'], policy.output[:, 8:11])
        np.testing.assert_array_equal(result['action.gripper_close'], policy.output[:, 11:12])
        # Values above one are preserved; the adapter never applies a control clamp.
        self.assertGreater(result['action.end_effector_position'].max(), 1)
        self.assertEqual(engine.metadata['recommended_action_steps'], 16)

    def test_invalid_sensors_and_predictions_are_rejected(self):
        policy, sensors = FakeAPPolicy(), observation()
        engine = PatchWAMAdapter(policy)
        inputs = make_policy_observation(sensors, engine.metadata)
        for key, value in (
            ('state.base_position', np.zeros((1, 2))),
            ('state.base_rotation', np.full((1, 4), np.nan)),
            (f'video.{CAMERAS[0]}', np.zeros((1, 256, 256, 3), np.float32)),
            (LANGUAGE, np.array([''])),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                engine.predict({**inputs, key: value})
        self.assertEqual(policy.requests, [])
        for output in (np.zeros((8, 12)), np.zeros((16, 11)), np.full((16, 12), np.nan)):
            policy.output = output
            with self.assertRaises(ValueError):
                engine.predict(inputs)

    def test_http_protocol_and_instruction_switch_use_current_observation(self):
        policy = FakeAPPolicy()
        engine = PatchWAMAdapter(policy)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = PolicyClient(f'http://127.0.0.1:{server.server_port}', timeout=2)
            self.assertEqual(client.metadata()['policy'], 'patchwam')
            sensors = observation()
            for i, instruction in enumerate(('Pick the mug from the cabinet.', 'Place the mug on the tray.')):
                sensors['state.base_position'] = np.full(3, i, np.float32)
                inputs = make_policy_observation(sensors, engine.metadata, instruction)
                reply = client.predict({**inputs, '__request_id': np.array(f'episode:step:{i}')})
                self.assertEqual(reply.pop('__request_id').item(), f'episode:step:{i}')
                self.assertEqual(reply['action.base_motion'].shape, (16, 4))
            self.assertEqual([r[2] for r in policy.requests],
                             ['Pick the mug from the cabinet.', 'Place the mug on the tray.'])
            np.testing.assert_array_equal(policy.requests[-1][1][:3], np.ones(3))
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                client.predict({'__request_id': np.array('invalid')})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_patchwam_context_does_not_change_gr00t_context(self):
        metadata = PatchWAMAdapter(FakeAPPolicy()).metadata
        patch = build_simulator_context('ArrangeTea', True, metadata, [], False, {})
        gr00t = build_simulator_context('ArrangeTea', True, {**metadata, 'policy': 'gr00t_n1.5'}, [], False, {})
        self.assertIn('Frozen PatchWAM', patch['guide']['robot']['executor'])
        self.assertIn('Frozen GR00T N1.5', gr00t['guide']['robot']['executor'])
        self.assertFalse(patch['tools']['set_gripper']['available'])


if __name__ == '__main__':
    unittest.main()
