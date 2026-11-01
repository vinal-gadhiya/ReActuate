"""Episode-only termination for the 244b087 experiment. Run in Thor containers."""

import unittest

from core.policy_http import RetryableSupervisorError
from core.scene_memory import SceneMemory
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision, step


class SimulatorTerminationTests(unittest.TestCase):
    def run_episode(self, decide, instruction='Open the cabinet.', max_steps=96, **options):
        tools, events = FakeTools(), []
        tools.max_steps = max_steps
        memory = SceneMemory('Move the mug from the cabinet to the tray.')
        plan = [step('active', instruction)]
        reason = run_subtasks(tools, plan, 'agent', lambda *args: decide(tools, *args), events.append,
                              execution_policy='episode', scene_memory=memory,
                              decision_steps=32, **options)
        return reason, tools, events

    def test_stop_before_first_motion_cannot_end_episode_across_task_families(self):
        for instruction in ('Open the cabinet.', 'Pick up the bowl.', 'Turn on the sink faucet.',
                            'Place the cup on the tray.', 'Navigate to the fridge.'):
            with self.subTest(instruction=instruction):
                reason, tools, events = self.run_episode(lambda *args: decision('stop', ''), instruction,
                                                        max_decisions=1, max_stop_continuations=0)
                self.assertEqual(reason, 'episode_step_budget')
                self.assertEqual(tools.steps, 96)
                self.assertEqual(tools.instructions, [instruction] * 3)
                self.assertTrue(all(event['reason'] == 'supervisor_stop' for event in events
                                    if event['event'] == 'policy_continuation'))

    def test_stop_after_valid_action_preserves_explicit_policy_instruction(self):
        calls = []

        def decide(tools, *args):
            calls.append(tools.steps)
            if len(calls) == 1:
                return decision(program='run_groot(instruction="Pick up the mug.", steps=32)')
            return decision('stop', '')

        reason, tools, _ = self.run_episode(decide)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(calls, [0, 32, 64])
        self.assertEqual(tools.instructions, ['Pick up the mug.'] * 3)

    def test_final_completion_cannot_set_success_or_replay_completed_stage(self):
        reason, tools, events = self.run_episode(lambda *args: decision('complete', ''), max_steps=70)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 70)
        self.assertFalse(tools.success)
        self.assertEqual(tools.instructions, ['Move the mug from the cabinet to the tray.'] * 3)
        self.assertTrue(any(event['event'] == 'planning_window_exhausted' for event in events))
        bursts = [event['arguments']['steps'] for event in events if event['event'] == 'tool']
        self.assertEqual(bursts, [32, 32, 6])

    def test_replan_after_final_completion_resumes_new_stage(self):
        calls = []

        def decide(tools, plan, index, spent, limit):
            calls.append(index)
            if len(calls) == 1:
                return decision('complete', '')
            if len(calls) == 2:
                return decision('replan', '', [step('remaining', 'Place the mug on the tray.')])
            return decision(program=f'run_groot(instruction="Place the mug on the tray.", steps={limit})')

        reason, tools, _ = self.run_episode(decide, max_replans=0)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.instructions, ['Move the mug from the cabinet to the tray.',
                                              'Place the mug on the tray.', 'Place the mug on the tray.'])
        self.assertEqual(calls, [0, 0, 1, 1])

    def test_invalid_decisions_and_inspections_cannot_end_or_spin_forever(self):
        for proposal in (decision('inspect', ''), {**decision(), 'program': 'invented_tool()'}):
            with self.subTest(status=proposal['status']):
                reason, tools, events = self.run_episode(lambda *args: proposal)
                self.assertEqual(reason, 'episode_step_budget')
                self.assertEqual(tools.steps, 96)
                self.assertTrue(any(event['event'] == 'policy_continuation' for event in events))

    def test_transient_failures_continue_and_fresh_decisions_are_retried(self):
        def unavailable(*args):
            raise RetryableSupervisorError('temporary outage')

        reason, tools, events = self.run_episode(unavailable, max_supervisor_failures=0)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 96)
        self.assertEqual(len([event for event in events if event['event'] == 'supervisor_unavailable']), 3)

    def test_missing_decision_keeps_physics_running_without_advancing_stages(self):
        reason, tools, _ = self.run_episode(lambda *args: None)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.instructions, ['Open the cabinet.'] * 3)

    def test_simulator_success_ends_immediately(self):
        def decide(tools, *args):
            original = tools.run_groot

            def successful_policy(instruction, steps):
                original(instruction, steps)
                tools.success = True

            tools.run_groot = successful_policy
            return decision('stop', '')

        reason, tools, _ = self.run_episode(decide)
        self.assertEqual(reason, 'task_success')
        self.assertEqual(tools.steps, 32)

    def test_episode_mode_keeps_evidence_checks_but_removes_navigation_caps(self):
        def decide(tools, plan, index, spent, limit):
            result = decision(program=f'navigate(fixture="cabinet", steps={limit})')
            result['assessment']['information_needed'] = 'Approach the task-named cabinet.'
            return result

        reason, tools, events = self.run_episode(decide, instruction='Navigate to the cabinet.',
                                                max_navigation_bursts=0)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.instructions, ['Navigate to the cabinet.'] * 3)
        self.assertFalse(any(event['event'] == 'recovery_required' for event in events))

    def test_permanent_policy_failure_is_not_hidden_or_converted_to_success(self):
        tools, events = FakeTools(), []

        def broken_policy(**kwargs):
            raise RuntimeError('CUDA failure')

        tools.run_groot = broken_policy
        with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
            run_subtasks(tools, [step('active', 'Open the cabinet.')], 'agent',
                         lambda *args: decision('stop', ''), events.append, execution_policy='episode')
        self.assertFalse(tools.success)

    def test_4000_steps_keep_supervision_even_with_small_legacy_caps(self):
        requests = []

        def decide(tools, plan, index, spent, limit):
            requests.append(tools.steps)
            return decision(program=f'run_groot(instruction="Open the cabinet.", steps={limit})')

        reason, tools, events = self.run_episode(decide, max_steps=4000, max_decisions=1,
                                                max_navigation_bursts=0, stall_bursts=1)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 4000)
        self.assertEqual(len(requests), 125)
        self.assertFalse(any(event['event'] == 'policy_continuation' for event in events))


if __name__ == '__main__':
    unittest.main()
