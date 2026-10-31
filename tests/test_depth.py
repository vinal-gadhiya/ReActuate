"""Container-only depth conversion and supervisor routing regression checks."""

import json
import unittest

import numpy as np

from core.depth import depth_to_metres, supervisor_depth_views, visualize_depth
from inference.supervisor import Supervisor
from sim.evaluate import make_policy_observation
from tests.test_agent import decision, scene_assessment, step


class DepthTests(unittest.TestCase):
    def test_metric_conversion_uses_actual_clipping_planes(self):
        near, far = 0.1, 10.0
        expected = np.array([[near, 1.0, far]])
        normalized = (1 - near / expected) / (1 - near / far)
        actual = depth_to_metres(normalized[..., None], near, far)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertEqual(actual.shape, (1, 3))
        with self.assertRaises(ValueError):
            depth_to_metres(normalized, 10, 0.1)

    def test_depth_scale_is_fixed_and_invalid_pixels_are_identified(self):
        a = visualize_depth(np.array([[0.5, 1.0, 5.0, np.nan]]), 2.0)
        b = visualize_depth(np.array([[1.0, 1.5]]), 2.0)
        np.testing.assert_array_equal(a[0, 1], b[0, 0])
        np.testing.assert_array_equal(a[0, 2], [0, 0, 0])
        np.testing.assert_array_equal(a[0, 3], [255, 0, 255])
        self.assertEqual(a.dtype, np.uint8)
        metric = depth_to_metres(np.array([[0.0, np.nan, 2.0]]), 0.1, 10)
        self.assertTrue(np.isnan(metric[0, 1:]).all())
        with self.assertRaises(ValueError):
            depth_to_metres(np.full((2, 2), np.nan), 0.1, 10)

    def test_depth_views_preserve_alignment_and_do_not_enter_policy_inputs(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.linspace(0, 1, 24).reshape(4, 6, 1)
        observation = {'video.camera': rgb, 'video.camera_depth': depth}
        views, metres, context = supervisor_depth_views(observation, ['video.camera'], 0.1, 10, 2)
        self.assertEqual(views['current_depth:video.camera'].shape, rgb.shape)
        self.assertEqual(float(metres['video.camera_depth'][0, 0]), np.float32(0.1))
        self.assertEqual(float(metres['video.camera_depth'][-1, -1]), 10)
        self.assertEqual(context['display_range_metres'], [0, 2])
        np.testing.assert_array_equal(observation['video.camera_depth'], depth)
        metadata = {'camera_width': 6, 'camera_height': 4,
                    'modalities': {'video': {'modality_keys': ['video.camera']}}}
        policy = make_policy_observation(observation, metadata)
        self.assertEqual(set(policy), {'video.camera'})
        np.testing.assert_array_equal(policy['video.camera'][0], rgb)
        with self.assertRaisesRegex(ValueError, 'missing'):
            supervisor_depth_views({'video.camera': rgb}, ['video.camera'], 0.1, 10)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            supervisor_depth_views({'video.camera': rgb, 'video.camera_depth': depth[:2]},
                                   ['video.camera'], 0.1, 10)

    def test_muse_receives_paired_depth_but_sam_and_camera_schema_use_rgb(self):
        calls = []

        class API:
            def response(self, model, content, instructions=None, schema=None):
                calls.append((model, content, instructions, schema))
                if model == 'sam-3.1':
                    return '', {'model': model}
                if 'scene_assessment' in schema['properties']:
                    return {'steps': [step('a', 'Open fridge.')], 'scene_assessment': scene_assessment()}, {'model': model}
                return decision(), {'model': model}

        images = {f'{phase}:{camera}': np.zeros((4, 6, 3), np.uint8)
                  for phase in ('current', 'before', 'current_depth', 'before_depth')
                  for camera in ('camera', 'right', 'wrist')}
        context = {'api_calls_left': 3, 'depth_observations': {'enabled': True},
                   'step': step('a', 'Open fridge.'), 'decision_step_limit': 8,
                   'capabilities': {'set_gripper': False}}
        supervisor = Supervisor(API(), 'muse', use_sam=True)
        supervisor.handle('plan', context, images)
        enum = calls[0][3]['properties']['scene_assessment']['properties']['entities']['items']['properties']['cameras']['items']['enum']
        self.assertEqual(enum, ['current:camera', 'current:right', 'current:wrist'])
        supervisor.handle('decide', context, images)
        self.assertEqual([call[0] for call in calls], ['muse', 'sam-3.1', 'muse'])
        self.assertEqual(sum(part['type'] == 'input_image' for part in calls[1][1]), 1)
        for call in (calls[0], calls[2]):
            self.assertEqual(sum(part['type'] == 'input_image' for part in call[1]), 12)
            payload = json.loads(call[1][0]['text'])
            self.assertTrue(payload['context']['depth_observations']['enabled'])
            self.assertIn('contact or gripper-object distance', call[2])
        orphan = {'current:camera': images['current:camera'], 'current_depth:other': images['current:camera']}
        with self.assertRaisesRegex(ValueError, 'match'):
            supervisor.handle('plan', context, orphan)
        with self.assertRaisesRegex(ValueError, 'depth context'):
            supervisor.handle('plan', {**context, 'depth_observations': {'enabled': False}}, images)


if __name__ == '__main__':
    unittest.main()
