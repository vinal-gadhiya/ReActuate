"""Container-only checks for public context and operator-note validation."""

import json
from pathlib import Path
import tempfile
import unittest

from core.simulator_context import (build_simulator_context, load_atomic_task_catalog,
                                    load_operator_context, task_success_requirements)


class SimulatorContextTests(unittest.TestCase):
    def test_success_rules_are_public_task_specific_and_fresh_per_request(self):
        rules = task_success_requirements('GarnishPancake')
        self.assertEqual(len(rules['requirements']), 4)
        self.assertIn('0.25 metres', rules['requirements'][-1])
        self.assertEqual(rules['upstream_ref'], '4f8a2980def75a55dff96b990745b83540425f09')
        self.assertNotIn('requirements', task_success_requirements('OtherTask'))
        rules['requirements'].clear()
        self.assertEqual(len(task_success_requirements('GarnishPancake')['requirements']), 4)

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
        self.assertEqual(len(context['policy_reference']['instruction_catalog']['tasks']), 65)
        json.dumps(context)

    def test_pinned_catalog_covers_all_atomic_descriptions_and_rejects_unknown_task(self):
        catalog = load_atomic_task_catalog()
        tasks = {task['name']: task for task in catalog['tasks']}
        self.assertEqual(len(tasks), 65)
        self.assertEqual(catalog['upstream_ref'], '4f8a2980def75a55dff96b990745b83540425f09')
        self.assertTrue(all(task['description'].strip() for task in tasks.values()))
        self.assertEqual(tasks['MakeIcedCoffee']['description'],
                         'Pick up an ice cube and place it in the glass of coffee.')
        self.assertEqual(tasks['OpenFridge']['description'], 'Open the fridge door(s).')
        self.assertIn('MIT License', catalog['upstream_license'])
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            load_atomic_task_catalog(['FutureTaskWithoutDescription'])

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
