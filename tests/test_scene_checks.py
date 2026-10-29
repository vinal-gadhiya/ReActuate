"""Scene comparison and image integrity regressions, without hardware or live APIs."""

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from core.observations import array_digest, check_rgb, rgb_diagnostics
from inference.meta_api import image_part
from inference.supervisor import Supervisor
from sim.scene_checks import (canonical_metadata, choose_layout_style, ensure_hash_seed, metadata_digest,
                              reset_scene, scene_fingerprint, seed_process, verify_scene)
from sim.run_agent import EpisodeTools


class SceneCheckTests(unittest.TestCase):
    def test_standalone_runner_reexecutes_with_comparison_hash_seed_and_preserves_arguments(self):
        with patch.dict('os.environ', {'PYTHONHASHSEED': 'random'}), \
                patch('sys.argv', ['run_agent.py', '--seed', '3', '--output', '/outputs/new run']), \
                patch('os.execve') as execute:
            ensure_hash_seed(3)
        arguments = execute.call_args.args
        self.assertEqual(arguments[1][1:], ['-m', 'sim.run_agent', '--seed', '3', '--output', '/outputs/new run'])
        self.assertEqual(arguments[2]['PYTHONHASHSEED'], '3')
        with patch.dict('os.environ', {'PYTHONHASHSEED': '3'}), patch('os.execve') as execute:
            ensure_hash_seed(3)
            execute.assert_not_called()

    def test_only_membership_order_is_ignored_in_old_scene_references(self):
        metadata = {'object_cfgs': [{'name': 'kettle', 'placement': {'pos': [0, 1]}, 'info': {
            'cat': 'kettle', 'mjcf_path': '/assets/kettle.xml',
            'groups_containing_sampled_obj': ['cookware', 'all', 'kettle']}}]}
        changed = deepcopy(metadata)
        changed['object_cfgs'][0]['info']['groups_containing_sampled_obj'].reverse()
        self.assertEqual(canonical_metadata(metadata), canonical_metadata(changed))
        self.assertEqual(metadata['object_cfgs'][0]['info']['groups_containing_sampled_obj'][0], 'cookware')
        expected = {'identity': {'metadata_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
                                 'model_xml_sha256': 'same', 'initial_state_sha256': 'same'},
                    'initial_metadata': metadata, 'initial_state': [0., 1.]}
        actual = {'identity': {**expected['identity'], 'metadata_sha256': metadata_digest(changed)},
                  'initial_metadata': changed, 'initial_state': [0., 1.]}
        self.assertTrue(verify_scene(actual, expected)['metadata_membership_order_normalized'])
        for field in ('model_xml_sha256', 'initial_state_sha256'):
            mismatch = {**actual, 'identity': {**actual['identity'], field: 'different'}}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                verify_scene(mismatch, expected)
        for key, value in (('mjcf_path', '/assets/other.xml'), ('cat', 'mug'),
                           ('groups_containing_sampled_obj', ['all', 'mug'])):
            other = deepcopy(changed)
            other['object_cfgs'][0]['info'][key] = value
            mismatch = {**actual, 'initial_metadata': other,
                        'identity': {**actual['identity'], 'metadata_sha256': metadata_digest(other)}}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'metadata_sha256'):
                verify_scene(mismatch, expected)
        other = deepcopy(changed)
        other['object_cfgs'][0]['placement']['pos'].reverse()
        self.assertNotEqual(metadata_digest(other), metadata_digest(metadata))

    def test_reset_replays_only_reference_spawn_through_normal_reset(self):
        calls = []
        metadata = {'init_robot_base_pos': [1., 2., 0.], 'init_robot_base_ori': [0., 0., 0.],
                    'object_cfgs': ['do not replay'], 'lang': 'do not replay'}
        kitchen = SimpleNamespace(get_ep_meta=lambda: metadata,
                                  set_ep_meta=lambda value: calls.append(('metadata', value)))
        env = SimpleNamespace(unwrapped=SimpleNamespace(env=kitchen),
                              reset=lambda seed: (calls.append(('reset', seed)) or ({'rgb': 'fresh'}, {})))
        reference = {'initial_metadata': {**metadata, 'init_robot_base_pos': [3., 4., 0.]}}
        observation, _ = reset_scene(env, 7, reference)
        self.assertEqual(calls, [('reset', 7), ('metadata', {
            'init_robot_base_pos': [3., 4., 0.], 'init_robot_base_ori': [0., 0., 0.]}), ('reset', 7)])
        self.assertEqual(observation, {'rgb': 'fresh'})
        self.assertEqual(metadata['init_robot_base_pos'], [1., 2., 0.])

    def test_baseline_uses_same_spawn_replay_reset_path(self):
        calls = []
        metadata = {'init_robot_base_pos': [1., 2., 0.], 'init_robot_base_ori': [0., 0., 0.]}
        kitchen = SimpleNamespace(get_ep_meta=lambda: metadata, set_ep_meta=lambda value: calls.append(value))
        env = SimpleNamespace(unwrapped=SimpleNamespace(env=kitchen), reset=lambda seed: ({}, {}))
        reset_scene(env, 0)
        self.assertEqual(calls, [metadata])
        with self.assertRaisesRegex(ValueError, 'regenerate the baseline'):
            reset_scene(env, 0, {'identity': {}})

    def test_mismatch_reports_metadata_fields_and_numeric_state_difference(self):
        expected = {'identity': {'metadata_sha256': 'a', 'initial_state_sha256': 'a'},
                    'initial_metadata': {'init_robot_base_pos': [1, 2, 0]}, 'initial_state': [0., 1., 2.]}
        actual = {'identity': {'metadata_sha256': 'b', 'initial_state_sha256': 'b'},
                  'initial_metadata': {'init_robot_base_pos': [2, 2, 0]}, 'initial_state': [0., 1.5, 2.]}
        with self.assertRaisesRegex(ValueError, r'metadata.init_robot_base_pos.*state_max_abs_delta=0.5'):
            verify_scene(actual, expected)

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
