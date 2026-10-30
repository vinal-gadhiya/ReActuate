"""Concise language stages without changing policy predictions; container only."""

import unittest

import numpy as np

from core.agent import parse_program, validate_stage_program
from core.scene_memory import SceneMemory
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


class StagedInstructionTests(unittest.TestCase):
    def test_simple_actions_for_different_tasks_are_accepted(self):
        for instruction in ('Pick up the mug.', 'Place the bowl on the table.',
                            'Close the cabinet.', 'Pour the milk into the bowl.',
                            'Turn on the stove.', 'Wash the apple.',
                            'Add salt and pepper.'):
            with self.subTest(instruction=instruction):
                validate_stage_program([('run_groot', {'instruction': instruction, 'steps': 16})])
        validate_stage_program(parse_program('navigate(fixture="counter", steps=16)'))

    def test_combined_or_verbose_stages_are_rejected_without_rewriting_them(self):
        for instruction in ('Pick up the cup and place it on the tray.',
                            'Pick up the cup. Place it on the tray.',
                            'Pick up the cup; place it on the tray.',
                            'Pick up the cup, place it on the tray.',
                            'Pick up the cup then carry it to the table.',
                            'Pick up the small white cup from the shelf above the sink beside the cabinet.'):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                validate_stage_program([('run_groot', {'instruction': instruction, 'steps': 16})])
        with self.assertRaises(ValueError):
            validate_stage_program(parse_program(
                'run_groot(instruction="Pick up the cup.", steps=4)\n'
                'run_groot(instruction="Place the cup on the tray.", steps=4)'))
        with self.assertRaises(ValueError):
            validate_stage_program(parse_program('navigate(fixture="the counter to the right of the large cabinet", steps=16)'))

    def test_runner_observes_between_pickup_and_placement(self):
        tools, events, observed = FakeTools(), [], []
        pickup = decision(program='run_groot(instruction="Pick up the cup.", steps=4)')
        acquired = decision('complete', '')
        acquired['assessment'].update(phase='carry', held_object='cup', grasp='supported')
        placing = decision(program='run_groot(instruction="Place the cup on the table.", steps=4)')
        placing['assessment'].update(phase='place', held_object='cup', grasp='supported')
        placed = decision('complete', '')
        placed['assessment'].update(phase='place', held_object='', grasp='empty')
        replies = iter([pickup, acquired, placing, placed])

        def decide(*args):
            observed.append(tools.steps)
            return next(replies)

        run_subtasks(tools, [step('pick', 'Pick up the cup.'), step('place', 'Place the cup on the table.')],
                     'agent', decide, events.append, concise_instructions=True)
        self.assertEqual(observed, [0, 4, 4, 8])
        self.assertEqual(tools.instructions, ['Pick up the cup.', 'Place the cup on the table.'])
        self.assertEqual([e['subtask'] for e in events if e['event'] == 'subtask_transition'], ['pick', 'place'])

    def test_rejected_combined_command_does_not_execute(self):
        tools, events = FakeTools(), []
        replies = iter([decision(program='run_groot(instruction="Pick up the cup and place it on the table.", steps=4)'),
                        decision(program='run_groot(instruction="Pick up the cup.", steps=4)'),
                        decision('stop', '')])
        run_subtasks(tools, [step('pick', 'Pick up the cup.')], 'agent', lambda *args: next(replies),
                     events.append, concise_instructions=True)
        self.assertEqual(tools.steps, 4)
        self.assertEqual(tools.instructions, ['Pick up the cup.'])
        self.assertEqual(sum(e['event'] == 'recovery_required' for e in events), 1)

    def test_stop_continuation_is_bounded_even_while_robot_is_moving(self):
        tools, events = FakeTools(), []
        tools.max_steps = 100
        replies = iter([decision(program='run_groot(instruction="Pick up the cup.", steps=4)')]
                       + [decision('stop', '') for _ in range(3)])
        reason = run_subtasks(tools, [step('pick', 'Pick up the cup.')], 'agent', lambda *args: next(replies),
                              events.append, concise_instructions=True, subtask_budget_mode='episode',
                              supervisor_stop_policy='continue', max_stop_continuations=2)
        self.assertEqual(reason, 'supervisor_stop_retry_limit')
        self.assertEqual(tools.steps, 36)
        self.assertEqual(len(tools.instructions), 3)
        self.assertEqual(events[-1]['event'], 'supervisor_stop_retry_limit')

    def test_named_fixture_navigation_allows_unknown_locations_but_not_object_targets(self):
        memory = SceneMemory()
        proposal = decision()
        proposal['assessment']['information_needed'] = 'Approach the target fixture.'
        proposal['memory_updates'] = [{'name': 'counter', 'kind': 'fixture', 'location': 'beside oven',
                                       'status': 'visible', 'evidence': 'Worktop visible in right camera.'},
                                      {'name': 'cabinet', 'kind': 'fixture', 'location': 'above counter',
                                       'status': 'visible', 'evidence': 'Door and handle visible.'}]
        memory.update(proposal, 0, 1)
        for name in ('counter', 'cabinet'):
            memory.authorize_navigation(name, 6)
        memory.authorize_navigation('table', 6)
        self.assertEqual(memory.navigation_source('table'), 'policy_navigation_request')
        with self.assertRaises(ValueError):
            memory.authorize_navigation('cup', 6)

    def test_panel_specific_commands_are_rejected_but_whole_fixture_actions_remain_valid(self):
        for instruction in ('Open the left fridge door.', 'Open the right door of the cabinet.',
                            'Close the fridge door on the left.', 'Open both fridge doors.'):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(ValueError, 'whole-fixture'):
                validate_stage_program([('run_groot', {'instruction': instruction})])
        for instruction in ('Open the fridge.', 'Close the cabinet.', 'Open the left drawer.'):
            validate_stage_program([('run_groot', {'instruction': instruction})])

    def test_task_named_destination_can_be_approached_without_inventing_visual_memory(self):
        memory = SceneMemory('Take the fruit from the fridge and place it on the pastry on the dining counter.')
        proposal = decision()
        proposal['assessment'].update(phase='locate_destination', held_object='fruit', grasp='supported',
                                      information_needed='Locate pastry at the task-named destination.')
        proposal['memory_updates'] = [{'name': 'dining counter', 'kind': 'fixture', 'location': 'not yet localized',
                                       'status': 'uncertain', 'evidence': 'Named by original task, not visible.'}]
        memory.update(proposal, 0, 1)
        memory.authorize_navigation('dining counter', 6)
        self.assertEqual(memory.navigation_source('dining counter'), 'original_task_description')
        self.assertIsNone(memory.context()['entries'][0]['last_seen_step'])
        self.assertEqual(memory.context()['entries'][0]['status'], 'uncertain')
        with self.assertRaisesRegex(ValueError, 'task-specific destination'):
            memory.authorize_navigation('counter', 6)
        memory.authorize_navigation('breakfast table', 6)
        self.assertEqual(memory.navigation_source('breakfast table'), 'policy_navigation_request')
        with self.assertRaises(ValueError):
            memory.authorize_navigation('pastry', 6)
        with self.assertRaises(ValueError):
            memory.authorize_navigation('dining counter', 0)

    def test_runner_executes_unseen_fixture_navigation_without_memory_or_task_mention(self):
        memory, tools, events = SceneMemory(), FakeTools(), []
        proposal = decision(program='navigate(fixture="dining counter", steps=4)')
        replies = iter([proposal, decision('stop', '')])
        run_subtasks(tools, [step('travel', 'Navigate to the dining counter.')], 'agent',
                     lambda *args: next(replies), events.append, scene_memory=memory,
                     concise_instructions=True)
        self.assertEqual(tools.instructions, ['Navigate to the dining counter.'])
        self.assertEqual(tools.steps, 4)
        self.assertEqual(memory.context()['entries'], [])
        executed = next(e for e in events if e['event'] == 'tool')
        self.assertEqual(executed['navigation_source'], 'policy_navigation_request')

    def test_unlocalized_destination_requires_navigation_before_placement(self):
        memory = SceneMemory('Place fruit on pastry at the dining counter.')
        proposal = decision()
        proposal['assessment'].update(phase='locate_destination', held_object='fruit', grasp='supported',
                                      information_needed='Need destination view.')
        memory.update(proposal, 0, 1)
        with self.assertRaisesRegex(ValueError, 'Destination is still unlocalized'):
            memory.authorize_placement_stage([('run_groot', {'instruction': 'Place the fruit on the pastry.'})])
        memory.authorize_placement_stage([('navigate', {'fixture': 'dining counter'})])
        memory.authorize_placement_stage([('run_groot', {'instruction': 'Pick up the fruit.'})])
        proposal['assessment'].update(phase='place', information_needed='')
        memory.update(proposal, 16, 2)
        memory.authorize_placement_stage([('run_groot', {'instruction': 'Place the fruit on the pastry.'})])

    def test_runner_keeps_specific_destination_navigation_as_a_separate_stage(self):
        memory, tools, events = SceneMemory('Place fruit on pastry at the dining counter.'), FakeTools(), []
        misplaced = decision(program='run_groot(instruction="Place the fruit on the pastry.", steps=4)')
        misplaced['assessment'].update(phase='locate_destination', held_object='fruit', grasp='supported',
                                       information_needed='Find pastry at dining counter.')
        navigation = decision(program='navigate(fixture="dining counter", steps=4)')
        navigation['assessment'] = dict(misplaced['assessment'])
        placed = decision(program='run_groot(instruction="Place the fruit on the pastry.", steps=4)')
        placed['assessment'].update(phase='place', held_object='fruit', grasp='supported')
        replies = iter([misplaced, navigation, placed, decision('stop', '')])
        run_subtasks(tools, [step('place', 'Place the fruit on the pastry.')], 'agent',
                     lambda *args: next(replies), events.append, scene_memory=memory,
                     concise_instructions=True)
        self.assertEqual(tools.instructions, ['Navigate to the dining counter.', 'Place the fruit on the pastry.'])
        navigated = next(e for e in events if e['event'] == 'tool' and e['tool'] == 'navigate')
        self.assertEqual(navigated['navigation_source'], 'original_task_description')

    def test_baseline_keeps_original_multi_action_instruction(self):
        tools = FakeTools()
        original = 'Pick up the cup and place it on the table.'
        run_subtasks(tools, [step('original', original)], 'baseline', None, lambda event: None,
                     concise_instructions=True)
        self.assertEqual(tools.instructions, [original])

    def test_supervisor_returns_actionable_feedback_for_combined_stage(self):
        class API:
            def response(self, *args, **kwargs):
                return decision(program='run_groot(instruction="Pick up the cup and place it on the tray.", steps=16)'), {}

        response = Supervisor(API(), 'muse', use_sam=False).handle('decide', {
            'step': step('pick', 'Pick up the cup.'), 'decision_step_limit': 32,
            'api_calls_left': 1, 'use_sam': False, 'capabilities': {'set_gripper': False},
            'instruction_style': 'single_stage',
        }, {'current:camera': np.zeros((4, 6, 3), np.uint8)})
        self.assertEqual(response['api_calls'], 1)
        self.assertIn('distinct stages', response['validation_error'])
