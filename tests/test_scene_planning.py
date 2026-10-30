"""Scene-grounded plan contracts and execution; no live model or simulator."""

from copy import deepcopy
import json
import unittest

import numpy as np

from core.scene_planning import GROUNDED_PLAN_SCHEMA, inventory_updates, load_plan, validate_grounded_plan
from core.agent import validate_placement_destination
from core.scene_memory import SceneMemory
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, scene_assessment, step


def tea_plan():
    survey = scene_assessment()
    survey['entities'] = [
        {'name': name, 'kind': kind, 'visibility': visibility, 'location': location,
         'access': access, 'cameras': [camera], 'evidence': evidence}
        for name, kind, visibility, location, access, camera, evidence in (
            ('mug', 'object', 'visible', 'open cabinet in front', 'apparently_reachable',
             'current:wrist', 'Mug exposed between open doors, near fingers.'),
            ('tray', 'object', 'visible', 'counter to right', 'needs_approach',
             'current:right', 'Tray farther right on continuous counter.'),
            ('kettle', 'object', 'visible', 'beside tray', 'needs_approach',
             'current:right', 'White kettle beside tray, farther from robot than mug.'),
            ('cabinet', 'fixture', 'visible', 'in front', 'apparently_reachable',
             'current:left', 'Cabinet doors already open.'),
        )]
    survey.update(
        ordering_constraints='Both objects must be on tray; no required order. Retrieve mug before closing cabinet.',
        first_action_rationale='Mug is exposed near gripper; kettle needs approach, so take mug first.',
        route_rationale='Carry mug to tray, handle adjacent kettle, return to close cabinet.')
    instructions = ('Pick up the mug.', 'Place the mug on the tray.',
                    'Pick up the kettle.', 'Place the kettle on the tray.', 'Close the cabinet.')
    return {'scene_assessment': survey,
            'steps': [step(str(i), instruction) for i, instruction in enumerate(instructions)]}


class ScenePlanningTests(unittest.TestCase):
    cameras = {'current:left', 'current:right', 'current:wrist'}

    def test_plan_survey_uses_all_current_views_and_is_serializable_for_replay(self):
        captured = []
        expected = tea_plan()

        class API:
            def response(self, model, content, instructions=None, schema=None):
                captured.append((content, instructions, schema))
                return deepcopy(expected), {'model': model}

        images = {key: np.full((4, 6, 3), i, np.uint8) for i, key in enumerate(sorted(self.cameras))}
        reply = Supervisor(API(), 'muse').handle('plan', {
            'api_calls_left': 1, 'original_goal': 'Put kettle and mug on tray, then close cabinet.',
        }, images)
        self.assertEqual(reply['api_calls'], 1)
        self.assertEqual(reply['result'], expected)
        content, prompt, schema = captured[0]
        self.assertEqual(schema, GROUNDED_PLAN_SCHEMA)
        self.assertEqual([item['text'] for item in content[1:] if item['type'] == 'input_text'], sorted(images))
        self.assertEqual(sum(item['type'] == 'input_image' for item in content), 3)
        self.assertIn('SCENE FIRST, ORDER SECOND', prompt)
        steps, survey = load_plan(json.loads(json.dumps(reply['result'])))
        self.assertEqual(steps[0]['instruction'], 'Pick up the mug.')
        self.assertEqual(survey, expected['scene_assessment'])
        survey['entities'].clear()
        self.assertEqual(len(reply['result']['scene_assessment']['entities']), 4)

    def test_generated_plan_requires_survey_and_real_camera_references(self):
        cases = []
        value = tea_plan()
        del value['scene_assessment']
        cases.append(value)
        value = tea_plan()
        value['scene_assessment']['entities'][0]['cameras'] = ['current:invented']
        cases.append(value)
        value = tea_plan()
        value['scene_assessment']['entities'][0]['cameras'] = []
        cases.append(value)
        value = tea_plan()
        value['scene_assessment']['entities'][0]['visibility'] = 'uncertain'
        cases.append(value)
        value = tea_plan()
        value['scene_assessment']['first_action_rationale'] = ''
        cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_grounded_plan(value, self.cameras)
        validate_grounded_plan(tea_plan(), self.cameras)

    def test_legacy_operator_plans_remain_supported_but_extra_fields_rejected(self):
        steps = [step('one', 'Pick up the mug.')]
        self.assertEqual(load_plan({'steps': steps}), (steps, None))
        with self.assertRaises(ValueError):
            load_plan({'steps': steps, 'unexpected': True})
        with self.assertRaises(ValueError):
            load_plan({'steps': steps, 'scene_assessment': {}})

    def test_survey_seeds_entities_without_fabricating_grasp_or_motion(self):
        memory = SceneMemory()
        survey = tea_plan()['scene_assessment']
        memory.update_entries(inventory_updates(survey), 0, 0)
        entries = {entry['name']: entry for entry in memory.context()['entries']}
        self.assertEqual(entries['mug']['location'], 'open cabinet in front')
        self.assertEqual(entries['kettle']['location'], 'beside tray')
        self.assertEqual(entries['cabinet']['status'], 'visible')
        self.assertIsNone(memory.assessment)
        self.assertEqual(memory.navigation_bursts, 0)
        survey['entities'][0]['evidence'] = 'mutated'
        self.assertIn('Mug exposed', memory.context()['entries'][0]['evidence'])

    def test_placement_cannot_silently_change_support_or_containment(self):
        cases = [('Place the cup on the tray.', 'Place the cup on the counter.'),
                 ('Put the apple in the bowl.', 'Put the apple on the bowl.'),
                 ('Place the pan on the dining table.', 'Place the pan on the table.')]
        for planned, proposed in cases:
            with self.subTest(proposed=proposed), self.assertRaises(ValueError):
                validate_placement_destination([('run_groot', {'instruction': proposed})], planned)
        for planned, proposed in [('Place the cup on the tray.', 'Place the cup onto the tray.'),
                                  ('Put the apple in the bowl.', 'Place the apple into the bowl.'),
                                  ('Place the cup on the tray.', 'Place the cup on the brown tray.')]:
            validate_placement_destination([('run_groot', {'instruction': proposed})], planned)

    def test_runner_rejects_temporary_counter_setdown_then_accepts_required_destination(self):
        tools, events = FakeTools(), []
        replies = iter([decision(program='run_groot(instruction="Place the cup on the counter.", steps=4)'),
                        decision(program='run_groot(instruction="Place the cup on the tray.", steps=4)'),
                        decision('stop', '')])
        run_subtasks(tools, [step('place', 'Place the cup on the tray.')], 'agent',
                     lambda *args: next(replies), events.append, concise_instructions=True)
        self.assertEqual(tools.instructions, ['Place the cup on the tray.'])
        self.assertEqual(tools.steps, 4)
        self.assertTrue(any(e['event'] == 'recovery_required' and 'destination differs' in e['reason'] for e in events))

    def test_mug_first_plan_completes_placement_before_kettle_without_temporary_setdown(self):
        plan, _ = load_plan(tea_plan())
        tools, events = FakeTools(), []
        proposals = []
        for item in plan:
            proposals.extend([decision(program=f'run_groot(instruction={item["instruction"]!r}, steps=4)'),
                              decision('complete', '')])
        replies = iter(proposals)
        reason = run_subtasks(tools, plan, 'agent', lambda *args: next(replies), events.append,
                              concise_instructions=True)
        self.assertEqual(reason, 'plan_exhausted_without_task_success')
        self.assertEqual(tools.instructions, [item['instruction'] for item in plan])
        placements = [e for e in events if e['event'] == 'subtask_transition']
        self.assertEqual(placements[1]['completed_step']['instruction'], 'Place the mug on the tray.')
        self.assertEqual(placements[1]['step'], 8)
        self.assertEqual(next(e['step_before'] for e in events if e['event'] == 'tool'
                              and e['arguments']['instruction'] == 'Pick up the kettle.'), 8)

    def test_held_useful_mug_can_finish_its_goal_without_repeating_pickup(self):
        original = [step('kettle', 'Pick up the kettle.'), step('kettle-place', 'Place the kettle on the tray.'),
                    step('mug', 'Pick up the mug.'), step('mug-place', 'Place the mug on the tray.'),
                    step('close', 'Close the cabinet.')]
        replacement = [original[3], original[0], original[1], original[4]]
        reorder = decision('replan', '', replacement)
        reorder['assessment'].update(held_object='mug', grasp='supported', phase='carry', destination='tray')
        place = decision(program='run_groot(instruction="Place the mug on the tray.", steps=4)')
        place['assessment'] = dict(reorder['assessment'])
        done = decision('complete', '')
        done['assessment'].update(grasp='empty', phase='place')
        replies = iter([reorder, place, done, decision('stop', '')])
        tools, events = FakeTools(), []
        run_subtasks(tools, original, 'agent', lambda *args: next(replies), events.append,
                     concise_instructions=True)
        self.assertEqual(tools.instructions, ['Place the mug on the tray.'])
        accepted = next(e for e in events if e['event'] == 'plan_replaced')['steps']
        self.assertEqual([item['id'] for item in accepted], ['mug-place', 'kettle', 'kettle-place', 'close'])
        self.assertEqual(next(e for e in events if e['event'] == 'subtask_transition')['subtask'], 'mug-place')


if __name__ == '__main__':
    unittest.main()
