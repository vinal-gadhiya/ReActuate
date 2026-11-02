"""Catalog-grounded language contracts. Run in Thor containers, not on the host."""

from copy import deepcopy
import json
import re
import unittest

import numpy as np

from core.atomic_language import AtomicInstructionGrounder
from core.simulator_context import load_atomic_task_catalog, composite_task_reference
from core.working_context import task_references
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, scene_assessment, step


class AtomicLanguageTests(unittest.TestCase):
    def setUp(self):
        self.grounder = AtomicInstructionGrounder()

    def test_split_transfer_navigation_and_fixture_formats_keep_entity_names(self):
        for instruction, form in (
            ('Pick up the white kettle.', 'pickup'),
            ('Pick the blue mug from the open cabinet.', 'pickup_from'),
            ('Place the mug on the tray.', 'place_on'),
            ('Place the apple in the bowl.', 'place_in'),
            ('Place the mug under the coffee machine dispenser.', 'place_under'),
            ('Open the cabinet.', 'open'), ('Close the fridge doors.', 'close'),
            ('Navigate to the dining counter.', 'navigate'),
            ('Turn on the sink faucet.', 'turn_on'),
            ('Pick up the mug and place it on the tray.', 'transfer_on'),
            ('Pick the croissant from the white plate and place it on the tray.', 'transfer_from_on'),
            ('Pick the mug from the counter and place it in the cabinet.', 'transfer_from_in'),
            ('Pick the mug from the tray and place it under the coffee machine dispenser.', 'transfer_from_under'),
        ):
            with self.subTest(instruction=instruction):
                match = self.grounder.validate(instruction)
                self.assertEqual(match['form'], form)
                self.assertTrue(match['source_tasks'])

    def test_all_65_official_descriptions_remain_available_including_specialized_actions(self):
        catalog = load_atomic_task_catalog()
        self.assertEqual(len(catalog['tasks']), 65)
        for task in catalog['tasks']:
            instruction = re.sub(r'\{([^{}]+)\}|\[([^\[\]]+)\]',
                                 lambda match: (match[1] or match[2]).split('/')[0]
                                 if '/' in (match[1] or match[2]) else 'visible item', task['description'])
            instruction = instruction.replace('(s)', 's')
            with self.subTest(task=task['name']):
                self.assertTrue(self.grounder.validate(instruction)['source_tasks'])
                self.assertTrue(self.grounder.validate_generated(instruction)['source_tasks'])
        for instruction in ('Press the start button on the microwave.',
                            'Fully slide the oven rack out.',
                            'Increase the toaster oven temperature.',
                            'Press down the lever to turn on the electric kettle.'):
            self.grounder.validate(instruction)

    def test_free_form_synonyms_unfilled_slots_and_appended_actions_are_rejected(self):
        for instruction in (
            'Grab the mug.', 'Set down the mug on the counter.',
            'Head over to the dining counter.', 'Align the gripper with the handle.',
            'Pick up the {object}.', 'Pick up the mug. Then close the cabinet.',
            'Place the mug on the tray and open the cabinet.',
            'Pick up the mug and lift the gripper.',
            'Pick the mug from the counter and place it on the tray and close the cabinet.',
        ):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                self.grounder.validate(instruction)

    def test_invalid_form_provenance_is_rejected(self):
        catalog = deepcopy(load_atomic_task_catalog())
        catalog['instruction_forms'][0]['source_tasks'] = ['InventedAtomicTask']
        with self.assertRaises(ValueError):
            AtomicInstructionGrounder(catalog)

    def test_generated_commands_accept_documented_panel_vocabulary_without_rewriting(self):
        for fixture in ('fridge', 'cabinet'):
            for verb in ('Open', 'Close'):
                whole, part = f'{verb} the {fixture}.', f'{verb} the {fixture} door.'
                self.grounder.validate_generated(whole)
                self.grounder.validate_generated(part)
                self.grounder.validate_generated(f'{verb} the {fixture} doors.')
                self.grounder.validate_generated(part, committed_instruction=part)
        # The documented format supports additional scene fixtures.
        catalog = deepcopy(load_atomic_task_catalog())
        for task in catalog['tasks']:
            if task['description'] == 'Open the cabinet door(s).':
                task['description'] = 'Open the pantry door(s).'
        AtomicInstructionGrounder(catalog).validate_generated('Open the pantry door.')

    def test_runner_preserves_catalog_grounded_instruction_text(self):
        for committed in ('Open the fridge.', 'Open the fridge door.', 'Open the cabinet doors.'):
            replies = [decision(program=f'run_groot(instruction={committed!r}, steps=4)')]
            replies.append(decision('stop', ''))
            responses, tools, events = iter(replies), FakeTools(), []
            with self.subTest(committed=committed):
                run_subtasks(tools, [step('open', committed)], 'agent', lambda *args: next(responses),
                             events.append, instruction_grounder=self.grounder)
                self.assertEqual(tools.instructions, [committed])
                self.assertEqual(tools.steps, 4)
                self.assertFalse(any(event['event'] == 'invalid_decision' for event in events))

    def test_composite_skill_hints_reserve_atomic_references_and_current_atomic_is_retained(self):
        catalog = load_atomic_task_catalog()
        catalog = {**catalog, 'source': catalog['description_source']}
        context = {'original_goal': 'Arrange the required objects.', 'simulator_context': {'task': {
            'composite_reference': composite_task_reference('GarnishPancake')}}}
        selected = task_references(catalog, context)['tasks']
        names = {item['name'] for item in selected}
        for skill in ('nav', 'door_open', 'PickPlace'):
            sources = {name for form in catalog['instruction_forms'] if skill in form['skill_keys']
                       for name in form['source_tasks']}
            self.assertTrue(names & sources)
        context['simulator_context']['task'] = {'name': 'AdjustWaterTemperature'}
        self.assertIn('AdjustWaterTemperature', {item['name'] for item in task_references(catalog, context)['tasks']})
        context['simulator_context']['task'] = {'composite_reference': {
            'skill_keys': ['rack_slide', 'start_toaster', 'blender_lid_close'], 'skills_required': []}}
        names = {item['name'] for item in task_references(catalog, context)['tasks']}
        for skill in ('rack_slide', 'start_toaster', 'blender_lid_close'):
            self.assertTrue(names & set(catalog['skill_reference_tasks'][skill]))

    def test_runner_rejects_ungrounded_proposal_before_motion_or_memory_updates(self):
        invalid = decision(program='run_groot(instruction="Grab the mug.", steps=4)')
        invalid['memory_updates'] = [{'name': 'invented', 'kind': 'object', 'location': 'counter',
                                      'status': 'visible', 'evidence': 'Invalid proposal claim.'}]
        valid = decision(program='run_groot(instruction="Pick up the mug.", steps=4)')
        replies = iter([invalid, valid, decision('stop', '')])
        tools, events = FakeTools(), []
        run_subtasks(tools, [step('pick', 'Pick up the mug.')], 'agent', lambda *args: next(replies),
                     events.append, instruction_grounder=self.grounder)
        self.assertEqual(tools.instructions, ['Pick up the mug.'])
        self.assertEqual(tools.steps, 4)
        self.assertEqual(sum(event['event'] == 'invalid_decision' for event in events), 1)
        memory_events = [event for event in events if event['event'] == 'scene_memory']
        self.assertFalse(any(entry['name'] == 'invented' for event in memory_events
                             for entry in event['memory']['entries']))
        executed = next(event for event in events if event['event'] == 'tool')
        self.assertEqual(executed['instruction_grounding']['form'], 'pickup')

    def test_replan_language_is_validated_and_baseline_preserves_explicit_text(self):
        replacement = [step('bad', 'Reposition near the mug.')]
        replies = iter([decision('replan', '', replacement), decision('stop', '')])
        tools, events = FakeTools(), []
        run_subtasks(tools, [step('pick', 'Pick up the mug.')], 'agent', lambda *args: next(replies),
                     events.append, instruction_grounder=self.grounder)
        self.assertEqual(tools.steps, 0)
        self.assertFalse(any(event['event'] == 'plan_replaced' for event in events))
        original = 'Take the mug and arrange everything for serving.'
        tools = FakeTools()
        run_subtasks(tools, [step('original', original)], 'baseline', None, lambda event: None,
                     instruction_grounder=self.grounder)
        self.assertEqual(tools.instructions, [original])

    def test_supervisor_returns_budgeted_language_feedback_and_supplies_forms(self):
        requests = []

        class API:
            def response(self, model, content, instructions=None, schema=None):
                requests.append(json.loads(content[0]['text']))
                return decision(program='run_groot(instruction="Grab the mug.", steps=4)'), {'model': model}

        response = Supervisor(API(), 'muse').handle('decide', {
            'working_context_version': 2, 'api_calls_left': 1, 'decision_step_limit': 16,
            'capabilities': {'set_gripper': False}, 'step': step('pick', 'Pick up the mug.'),
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertIn('not grounded', response['validation_error'])
        self.assertEqual(response['api_calls'], 1)
        self.assertEqual(len(response['audit']), 1)
        self.assertTrue(requests[0]['atomic_instruction_forms'])

    def test_initial_plan_and_replan_return_language_errors_without_losing_usage(self):
        requirements = [{'id': 'serve', 'description': 'Place the mug on the tray.', 'dependencies': []}]
        initial = {'requirements': requirements, 'scene_assessment': scene_assessment(),
                   'steps': [{**step('pick', 'Grab the mug.'), 'goal_ids': ['serve']}]}
        replacement = [{**step('retry', 'Set down the mug on the tray.'), 'goal_ids': ['serve']}]
        results = iter([initial, decision('replan', '', replacement)])

        class API:
            def response(self, model, content, instructions=None, schema=None):
                return next(results), {'model': model, 'usage': {'output_tokens': 10}}

        supervisor = Supervisor(API(), 'muse')
        context = {'working_context_version': 2, 'api_calls_left': 1, 'decision_step_limit': 16,
                   'step': step('pick', 'Pick up the mug.'), 'capabilities': {'set_gripper': False}}
        for operation in ('plan', 'decide'):
            response = supervisor.handle(operation, context, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
            self.assertIn('not grounded', response['validation_error'])
            self.assertEqual(response['api_calls'], 1)
            self.assertEqual(response['audit'][0]['usage']['output_tokens'], 10)


if __name__ == '__main__':
    unittest.main()
