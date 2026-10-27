"""Run inside the inference container: python -m unittest discover -s tests -v."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from core.policy_http import PolicyClient, decode_arrays, encode_arrays
from inference.prepare_upstream import prepare
from inference.serve import make_handler
from sim.evaluate import make_policy_observation, validate_actions


def metadata():
    return {
        'protocol_version': 1, 'camera_height': 4, 'camera_width': 6,
        'modalities': {
            'video': {'modality_keys': ['video.camera'], 'delta_indices': [0]},
            'state': {'modality_keys': ['state.arm'], 'delta_indices': [0]},
            'language': {'modality_keys': ['annotation.task'], 'delta_indices': [0]},
            'action': {'modality_keys': ['action.arm'], 'delta_indices': [0, 1]},
        },
    }


class FakeEngine:
    metadata = metadata()

    def predict(self, observation):
        if observation['video.camera'].shape != (1, 4, 6, 3):
            raise ValueError('Incorrect temporal image shape')
        return {'action.arm': np.array([[0.25, 0.5], [0.75, 1]], dtype=np.float32)}


class PolicyTests(unittest.TestCase):
    def test_numpy_roundtrip_and_object_rejection(self):
        source = {'image': np.arange(72, dtype=np.uint8).reshape(4, 6, 3), 'text': np.array(['open cabinet'])}
        received = decode_arrays(encode_arrays(source))
        for key in source:
            np.testing.assert_array_equal(source[key], received[key])
        with self.assertRaises(ValueError):
            encode_arrays({'invalid': np.array([object()])})

    def test_observations_preserve_pixels_and_temporal_axis(self):
        image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        observation = {'video.camera': image, 'state.arm': np.zeros(2), 'annotation.task': 'open cabinet'}
        converted = make_policy_observation(observation, metadata())
        np.testing.assert_array_equal(converted['video.camera'][0], image)
        self.assertEqual(converted['state.arm'].shape, (1, 2))
        self.assertEqual(converted['annotation.task'].tolist(), ['open cabinet'])
        with self.assertRaises(ValueError):
            make_policy_observation({**observation, 'annotation.task': ''}, metadata())

    def test_action_validation_rejects_missing_keys_nan_and_bad_horizons(self):
        space = SimpleNamespace(spaces={'action.arm': SimpleNamespace(shape=(2,))})
        validate_actions({'action.arm': np.zeros((2, 2))}, space, 2)
        for bad in ({}, {'action.arm': np.zeros((1, 2))}, {'action.arm': np.full((2, 2), np.nan)}):
            with self.assertRaises(ValueError):
                validate_actions(bad, space, 2)

    def test_http_inference_and_error_response(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(FakeEngine()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = PolicyClient(f'http://127.0.0.1:{server.server_port}', timeout=2)
            self.assertEqual(client.metadata(), metadata())
            inputs = {
                'video.camera': np.zeros((1, 4, 6, 3), dtype=np.uint8),
                'state.arm': np.zeros((1, 2)), 'annotation.task': np.array(['open cabinet']),
                '__request_id': np.array('episode:0:step:0'),
            }
            result = client.predict(inputs)
            self.assertEqual(result.pop('__request_id').item(), 'episode:0:step:0')
            np.testing.assert_array_equal(result['action.arm'], FakeEngine().predict(inputs)['action.arm'])
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                client.predict({'__request_id': np.array('invalid')})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_inference_packaging_and_attention_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'pyproject.toml'
            project.write_text('[project]\nname = "gr00t"\ndependencies = [\n"training-only"\n]\n')
            requirements = root / 'requirements.txt'
            requirements.write_text('# comment\nnumpy==1.26.4\n')
            config = root / 'gr00t/model/backbone/eagle2_hg_model/config.json'
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({'text_config': {}, 'vision_config': {}}))
            model = config.parent / 'modeling_eagle2_5_vl.py'
            model.write_text('config.vision_config._attn_implementation = "flash_attention_2"\n')
            prepare(root, requirements)
            self.assertNotIn('training-only', project.read_text())
            self.assertIn('torch==2.9.1', project.read_text())
            self.assertNotIn('flash_attention_2', model.read_text())
            patched = json.loads(config.read_text())
            for section in (patched, patched['text_config'], patched['vision_config']):
                self.assertEqual(section['_attn_implementation'], 'eager')


if __name__ == '__main__':
    unittest.main()
