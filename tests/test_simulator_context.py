"""Container-only checks for public context and operator-note validation."""

import json
from pathlib import Path
import tempfile
import unittest

from core.simulator_context import build_simulator_context, load_operator_context


class SimulatorContextTests(unittest.TestCase):
    def test_context_separates_capabilities_from_reference_tasks(self):
        metadata = {'camera_width': 256, 'camera_height': 256, 'modalities': {
            'video': {'modality_keys': ['video.robot0_agentview_left', 'video.robot0_eye_in_hand']}}}
        context = build_simulator_context('MakeIceLemonade', True, metadata,
                                          ['OpenFridge', 'OpenCabinet'], False, {'scene_notes': 'A cup is visible.'})
        self.assertFalse(context['tools']['set_gripper']['available'])
        self.assertTrue(context['tools']['run_groot']['available'])
        self.assertEqual(context['task']['kind'], 'composite')
        self.assertEqual(context['policy_reference']['registered_atomic_task_names'], ['OpenCabinet', 'OpenFridge'])
        self.assertEqual(context['cameras'][1]['role'], 'wrist camera')
        self.assertNotIn('objects', context)
        self.assertNotIn('object_poses', context)
        self.assertIn('not independently verified', context['operator_context']['source'])
        json.dumps(context)

    def test_operator_context_validation(self):
        self.assertEqual(load_operator_context(None), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'context.json'
            expected = {'task_notes': 'Release objects after placing them.', 'constraints': ['Do not move the bowl.']}
            path.write_text(json.dumps(expected))
            self.assertEqual(load_operator_context(path), expected)
            path.write_text(json.dumps({'tools': ['teleport']}))
            with self.assertRaises(ValueError):
                load_operator_context(path)
