"""Scene comparison and image integrity regressions, without hardware or live APIs."""

import base64
from io import BytesIO
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from core.observations import array_digest, check_rgb, rgb_diagnostics
from inference.meta_api import image_part
from inference.supervisor import Supervisor
from sim.scene_checks import choose_layout_style, scene_fingerprint, seed_process, verify_scene
from sim.run_agent import EpisodeTools


class SceneCheckTests(unittest.TestCase):
    def test_fixed_scene_selection_is_independent_of_global_rng_and_respects_exclusions(self):
        pairs = [(1, 1), (11, 14), (15, 14), (40, 34)]
        selected = choose_layout_style(pairs, [1], [34], 3)
        np.random.random(100)
        self.assertEqual(selected, choose_layout_style(pairs, [1], [34], 3))
        self.assertIn(selected, [(11, 14), (15, 14)])
        seed_process(5)
        first = np.random.random()
        seed_process(5)
        self.assertEqual(np.random.random(), first)

    def test_scene_mismatch_is_not_hidden_by_matching_goal(self):
        state = SimpleNamespace(flatten=lambda: np.array([0., 1., 2.]))
        kitchen = SimpleNamespace(get_ep_meta=lambda: {'layout_id': 11, 'style_id': 14, 'pose': np.zeros(3)},
                                  sim=SimpleNamespace(model=SimpleNamespace(get_xml=lambda: '<mujoco/>'),
                                                      get_state=lambda: state))
        env = SimpleNamespace(unwrapped=SimpleNamespace(env=kitchen))
        obs = {'annotation.human.task_description': 'Place pitcher and cup.',
               'video.camera': np.zeros((8, 8, 3), dtype=np.uint8)}
        actual = scene_fingerprint(env, obs, ['video.camera'])
        verify_scene(actual, actual)
        changed = {**actual, 'identity': {**actual['identity'], 'layout_id': 40}}
        with self.assertRaisesRegex(ValueError, 'layout_id'):
            verify_scene(actual, changed)

    def test_png_preserves_exact_pixels_and_does_not_mutate_source(self):
        original = np.random.default_rng(2).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        expected = original.copy()
        part = image_part(original)
        data = base64.b64decode(part['image_url'].split(',', 1)[1])
        decoded = np.asarray(Image.open(BytesIO(data)))
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(array_digest(decoded), array_digest(original))

    def test_noise_guard_preserves_source_evidence(self):
        noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.assertTrue(rgb_diagnostics({'camera': noise})['camera']['suspected_static_noise'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'static noise'):
                check_rgb({'camera': noise}, Path(directory), 40)
            self.assertTrue((Path(directory) / 'suspected-noise-step-40-camera-0.png').exists())
        # A noisy wrist view alone does not reject all otherwise valid cameras.
        check_rgb({'wrist': noise, 'external': np.zeros_like(noise)})

    def test_supervisor_checks_rgb_hash_before_any_api_request(self):
        api = SimpleNamespace(response=lambda *args, **kwargs: self.fail('API must not be called on hash mismatch'))
        supervisor = Supervisor(api, 'test-model')
        context = {'api_calls_left': 1, 'image_diagnostics': {'current:camera': {'sha256': 'incorrect'}}}
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            supervisor.handle('plan', context, {'current:camera': np.zeros((8, 8, 3), dtype=np.uint8)})

    def test_inspection_refreshes_pixels_without_advancing_physics(self):
        original = {'video.camera': np.zeros((8, 8, 3), dtype=np.uint8)}
        refreshed = {'video.camera': np.ones((8, 8, 3), dtype=np.uint8)}
        calls = []

        def observe(force_update=False):
            calls.append(force_update)
            return refreshed

        wrapper = SimpleNamespace(env=SimpleNamespace(_get_observations=observe), get_observation=lambda raw: raw)
        tools = EpisodeTools(SimpleNamespace(unwrapped=wrapper), original, None,
                             {'modalities': {'video': {'modality_keys': ['video.camera']}}},
                             SimpleNamespace(append_data=lambda frame: None), 10, 8)
        tools.refresh_observation()
        steps, clips = tools.perception_clip()
        self.assertEqual(calls, [True])
        self.assertEqual(tools.steps, 0)
        self.assertEqual(steps, [0])
        np.testing.assert_array_equal(clips['clip:video.camera'][-1], refreshed['video.camera'])
