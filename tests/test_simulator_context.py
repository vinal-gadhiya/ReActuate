"""Container-only checks for public context and operator-note validation."""

import json
from pathlib import Path
import tempfile
import unittest

from core.simulator_context import (build_simulator_context, load_atomic_task_catalog,
                                    load_composite_task_catalog, composite_task_reference,
                                    load_operator_context, task_success_requirements)


class SimulatorContextTests(unittest.TestCase):
    def test_composite_catalog_covers_all_tasks_and_official_skill_rules(self):
        catalog = load_composite_task_catalog()
        tasks = {task['name']: task for task in catalog['tasks']}
        self.assertEqual(len(catalog['tasks']), 300)
        self.assertEqual(len(tasks), 300)
        self.assertEqual(catalog['upstream_ref'], '4f8a2980def75a55dff96b990745b83540425f09')
        self.assertTrue(all(task['description'].strip() for task in tasks.values()))
        self.assertEqual(tasks['GarnishPancake']['skills_required'],
                         ['Navigation', 'Open Door', 'Pick & Place'])
        self.assertEqual(tasks['ArrangeTea']['skills_required'], ['Close Door', 'Pick & Place'])
        self.assertEqual(tasks['RinseCuttingBoard']['skills_required'], ['Turn Lever'])
        self.assertEqual(tasks['PrepareCoffee']['skills_required'], ['Press Button', 'Pick & Place'])
        self.assertFalse(set(tasks) & {task['name'] for task in load_atomic_task_catalog()['tasks']})
        self.assertIn('MIT License', catalog['upstream_license'])
        with self.assertRaisesRegex(ValueError, 'does not cover registered tasks'):
            load_composite_task_catalog(['FutureCompositeWithoutDocumentation'])
        with self.assertRaisesRegex(ValueError, 'does not cover current task'):
            composite_task_reference('FutureCompositeWithoutDocumentation')

    def test_composite_context_selects_current_task_and_keeps_atomic_episodes_separate(self):
        metadata = {'camera_width': 256, 'camera_height': 256,
                    'modalities': {'video': {'modality_keys': ['video.robot0_agentview_left']}}}
        context = build_simulator_context('GarnishPancake', True, metadata, [], False, {},
                                          composite_tasks=['GarnishPancake', 'ArrangeTea'])
        reference = context['task']['composite_reference']
        self.assertEqual(reference['name'], 'GarnishPancake')
        self.assertIn('dining counter', reference['description'])
        self.assertEqual(reference['skills_required'], ['Navigation', 'Open Door', 'Pick & Place'])
        self.assertIn('unordered', reference['usage'])
        self.assertNotIn('tasks', reference)
        reference['skills_required'].clear()
        self.assertEqual(len(composite_task_reference('GarnishPancake')['skills_required']), 3)
        atomic = build_simulator_context('OpenFridge', False, metadata, [], False, {})
        self.assertIsNone(atomic['task']['composite_reference'])
        with self.assertRaisesRegex(ValueError, 'does not cover registered tasks'):
            build_simulator_context('GarnishPancake', True, metadata, [], False, {},
                                    composite_tasks=['UndocumentedComposite'])

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
