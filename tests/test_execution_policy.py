"""Task-general stage commitment and physical-budget regressions; Thor only."""

import unittest

from core.agent import validate_decision
from core.atomic_language import AtomicInstructionGrounder
from core.execution import validate_stage_switch
from core.goals import AttemptHistory, GoalLedger
from core.policy_http import RetryableSupervisorError
from sim.run_agent import run_subtasks
from tests.test_agent import FakeTools, decision
from tests.test_goals import definitions, proposal, refs, stage, update


class EpisodeExecutionTests(unittest.TestCase):
    def test_output_limit_failures_retry_without_motion_then_end_cleanly(self):
        ledger, tools, events, plan = self.setup_episode('Pick up the bowl.')

        def decide(*args):
            raise RetryableSupervisorError('output limit', allow_policy_continuation=False)

        reason = self.run_episode(ledger, tools, events, plan, decide,
                                  max_decisions=10, max_supervisor_failures=2)
        self.assertEqual(reason, 'supervisor_failure_limit')
        self.assertEqual(tools.steps, 0)
        self.assertEqual(tools.instructions, [])
        self.assertEqual(len([event for event in events if event['event'] == 'supervisor_unavailable']), 3)

    def test_output_limit_cannot_fall_through_to_motion_when_decision_budget_expires(self):
        ledger, tools, events, plan = self.setup_episode('Pick up the bowl.')

        def decide(*args):
            raise RetryableSupervisorError('output limit', allow_policy_continuation=False)

        reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=1)
        self.assertEqual(reason, 'decision_budget')
        self.assertEqual(tools.steps, 0)
        self.assertFalse(any(event['event'] == 'policy_continuation' for event in events))

    def test_fresh_valid_decision_can_resume_after_output_limit_without_blind_motion(self):
        ledger, tools, events, plan = self.setup_episode('Pick up the bowl.', 32)
        observed_steps = []

        def decide(*args):
            observed_steps.append(tools.steps)
            if len(observed_steps) == 1:
                raise RetryableSupervisorError('output limit', allow_policy_continuation=False)
            result = proposal('continue', tools.steps, 'Pick up the bowl.')
            result['program'] = 'run_groot(instruction="Pick up the bowl.", steps=32)'
            return result

        reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=5)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(observed_steps, [0, 0])
        self.assertEqual(tools.instructions, ['Pick up the bowl.'])
        self.assertFalse(any(event['event'] == 'policy_continuation' for event in events))

    def setup_episode(self, instruction, max_steps=2000):
        ledger = GoalLedger('Required outcomes', [definitions()[0]])
        tools, events = FakeTools(), []
        tools.max_steps = max_steps
        plan = [stage('active', 'first', instruction)]
        return ledger, tools, events, plan

    def run_episode(self, ledger, tools, events, plan, decide, **options):
        def observe(*args):
            ledger.observe(tools.steps, ['current:camera'])
            return decide(*args)

        return run_subtasks(tools, plan, 'agent', observe, events.append, goal_ledger=ledger,
                            execution_policy='episode', decision_steps=32, **options)

    def test_repeated_no_progress_does_not_end_or_switch_any_task_family_at_160_steps(self):
        for instruction in ('Open the cabinet.', 'Turn on the sink faucet.',
                            'Pick up the bowl.', 'Navigate to the dining counter.'):
            with self.subTest(instruction=instruction):
                ledger, tools, events, plan = self.setup_episode(instruction)
                attempts = AttemptHistory(ledger, failure_limit=1, recovery_limit=0)
                limits = []

                def decide(active_plan, index, spent, limit):
                    limits.append(limit)
                    result = proposal('continue', tools.steps, instruction)
                    result['program'] = f'run_groot(instruction={instruction!r}, steps={limit})'
                    if attempts.attempts:
                        last = attempts.attempts[-1]
                        result['attempt_reports'] = [{
                            'attempt_id': last['attempt_id'], 'outcome': 'no_reported_progress',
                            'evidence': 'Before and after views show no supported outcome change.',
                            'observation_refs': refs(last['step_before']) + refs(tools.steps)}]
                    return result

                reason = self.run_episode(ledger, tools, events, plan, decide,
                                          max_decisions=100, subtask_budget_mode='plan', attempt_history=attempts)
                self.assertEqual(reason, 'episode_step_budget')
                self.assertEqual(tools.steps, 2000)
                self.assertEqual(set(tools.instructions), {instruction})
                self.assertEqual([event['reason'] for event in events if event['event'] == 'invalid_decision'], [])
                self.assertEqual(limits, [32] * 62 + [16])
                executed = [event for event in events if event['event'] == 'tool']
                self.assertEqual(executed[-1]['arguments']['steps'], 16)
                self.assertFalse(any(event['event'] == 'policy_continuation' for event in events))

    def test_oversized_final_burst_is_rejected_and_continuation_respects_remaining_steps(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 48)

        def decide(*args):
            result = proposal('continue', tools.steps, 'Open the cabinet.')
            # Deliberately violate the final 16-step request limit. The runner
            # must reject this proposal, then continue only the committed stage.
            result['program'] = 'run_groot(instruction="Open the cabinet.", steps=32)'
            return result

        reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=10)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 48)
        rejected = [event for event in events if event['event'] == 'invalid_decision']
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['step'], 32)
        self.assertEqual(rejected[0]['reason'], 'Invalid tool step budget')
        executed = [event for event in events if event['event'] == 'tool']
        self.assertEqual([event['step_after'] - event['step_before'] for event in executed], [32, 16])
        self.assertEqual(executed[-1]['arguments']['steps'], 16)
        self.assertEqual(set(tools.instructions), {'Open the cabinet.'})

    def test_model_stop_and_zero_stop_allowance_still_use_remaining_physical_budget(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 256)
        reason = self.run_episode(ledger, tools, events, plan,
                                  lambda *args: proposal('stop', tools.steps),
                                  max_decisions=100, max_stop_continuations=0)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(tools.steps, 256)
        self.assertEqual(set(tools.instructions), {'Open the cabinet.'})

    def test_decision_and_api_limits_bound_supervision_without_ending_motion(self):
        for response in ('valid', 'api_exhausted'):
            with self.subTest(response=response):
                ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 2000)
                calls = []

                def decide(*args):
                    calls.append(tools.steps)
                    if response == 'api_exhausted':
                        return None
                    return proposal('continue', tools.steps, 'Open the cabinet.')

                reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=1)
                self.assertEqual(reason, 'episode_step_budget')
                self.assertEqual(tools.steps, 2000)
                self.assertEqual(len(calls), 1)
                self.assertEqual(set(tools.instructions), {'Open the cabinet.'})

    def test_failed_bursts_alone_do_not_authorize_instruction_switch(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 128)

        def decide(*args):
            instruction = 'Open the cabinet.' if tools.steps == 0 else 'Navigate to the dining counter.'
            result = proposal('continue', tools.steps, instruction)
            result['recovery'] = {'change_type': 'strategy', 'description': 'A few bursts did not work; reset the pose.',
                                  'observation_refs': refs(tools.steps), 'necessity': 'not_needed'}
            return result

        self.run_episode(ledger, tools, events, plan, decide, max_decisions=10)
        self.assertEqual(tools.steps, 128)
        self.assertEqual(set(tools.instructions), {'Open the cabinet.'})
        self.assertTrue(any('necessary intervention' in event.get('reason', '') for event in events))

    def test_invalid_replan_or_control_fields_cannot_end_or_switch_active_execution(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 256)

                def decide(*args):
                    if tools.steps == 0:
                        result = proposal('continue', 0, 'Open the cabinet.')
                        result['program'] = 'run_groot(instruction="Open the cabinet.", steps=32)'
                        return result
                    replacement = [stage('different', 'first', 'Navigate to the counter.')]
                    result = proposal('replan', tools.steps, replacement=replacement)
                    result['recovery'] = {'change_type': 'strategy', 'description': 'A few bursts did not work.',
                                          'observation_refs': refs(tools.steps), 'necessity': 'not_needed'}
                    if malformed:
                        result['program'] = 'navigate(fixture="counter", steps=32)'
                    return result

                reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=20)
                self.assertEqual(reason, 'episode_step_budget')
                self.assertEqual(tools.steps, 256)
                self.assertEqual(set(tools.instructions), {'Open the cabinet.'})
                self.assertFalse(any(event['event'] == 'plan_replaced' for event in events))

    def test_supported_temporary_intervention_does_not_replace_committed_stage_on_stop(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 128)
        counter = 0

        def decide(*args):
            nonlocal counter
            counter += 1
            if counter == 1:
                result = proposal('continue', tools.steps, 'Open the cabinet.')
                result['program'] = 'run_groot(instruction="Open the cabinet.", steps=32)'
                return result
            if counter == 2:
                result = proposal('continue', tools.steps, 'Navigate to the counter.')
                result['program'] = 'run_groot(instruction="Navigate to the counter.", steps=32)'
                result['recovery'] = {'change_type': 'prerequisite', 'description': 'Current view shows obstructed access.',
                                      'observation_refs': refs(tools.steps), 'necessity': 'blocked_access'}
                return result
            return proposal('stop', tools.steps)

        self.run_episode(ledger, tools, events, plan, decide, max_decisions=3)
        self.assertEqual(tools.instructions[:2], ['Open the cabinet.', 'Navigate to the counter.'])
        self.assertEqual(set(tools.instructions[2:]), {'Open the cabinet.'})

    def test_verified_completion_advances_to_next_stage(self):
        ledger, tools, events = GoalLedger('Two outcomes', definitions()), FakeTools(), []
        tools.max_steps = 64
        plan = [stage('access', 'first', 'Open the cabinet.'),
                stage('acquire', 'second', 'Pick up the bowl.')]
        replies = iter([
            proposal('continue', 0, 'Open the cabinet.'),
            proposal('complete', 1, goal_updates=[update('first', 'satisfied', 1)]),
            proposal('continue', 1, 'Pick up the bowl.'),
        ])
        self.run_episode(ledger, tools, events, plan, lambda *args: next(replies), max_decisions=3)
        self.assertEqual(tools.instructions[0], 'Open the cabinet.')
        self.assertEqual(set(tools.instructions[1:]), {'Pick up the bowl.'})
        self.assertEqual(next(event for event in events if event['event'] == 'subtask_transition')['subtask'], 'access')

    def test_satisfied_continue_advances_immediately_across_task_families(self):
        for first, following in (
            ('Open the cabinet.', 'Pick up the bowl.'),
            ('Navigate to the counter.', 'Place the bowl on the tray.'),
            ('Turn on the sink faucet.', 'Place the bowl in the sink.'),
            ('Place the bowl on the tray.', 'Close the cabinet.'),
        ):
            for proposed_instruction in (first, following):
                with self.subTest(first=first, proposed_instruction=proposed_instruction):
                    ledger, tools, events, _ = self.setup_episode(first, 4)
                    plan = [stage('first-stage', 'first', first), stage('next-stage', 'first', following)]
                    observed = []

                    def decide(active, index, spent, limit):
                        observed.append((active[index]['id'], tools.steps))
                        if len(observed) == 1:
                            return proposal('continue', tools.steps, first)
                        if len(observed) == 2:
                            result = proposal('continue', tools.steps, proposed_instruction)
                            result['assessment']['outcome'] = 'satisfied'
                            # Completion must not require the final requirement,
                            # downstream grasp, next motion, or handoff to be done.
                            result['handoff']['readiness'] = 'not_applicable'
                            return result
                        return proposal('continue', tools.steps, following)

                    reason = self.run_episode(ledger, tools, events, plan, decide, max_decisions=3)
                    self.assertEqual(reason, 'episode_step_budget')
                    self.assertEqual(observed, [('first-stage', 0), ('first-stage', 1), ('next-stage', 1)])
                    self.assertEqual(tools.instructions[0], first)
                    self.assertEqual(set(tools.instructions[1:]), {following})
                    self.assertEqual(ledger.pending(), ['first'])
                    transition = next(event for event in events if event['event'] == 'subtask_transition')
                    self.assertEqual(transition['step'], 1)
                    self.assertEqual(transition['subtask'], 'first-stage')
                    self.assertFalse(any(event['event'] == 'invalid_decision' for event in events))

    def test_satisfied_replan_completes_stage_before_applying_next_window(self):
        ledger, tools, events, _ = self.setup_episode('Open the cabinet.', 4)
        plan = [stage('access', 'first', 'Open the cabinet.'),
                stage('old-next', 'first', 'Pick up the bowl.')]
        replacement = [stage('reordered-next', 'first', 'Navigate to the counter.')]
        observed = []

        def decide(active, index, spent, limit):
            observed.append((active[index]['id'], tools.steps))
            if len(observed) == 1:
                return proposal('continue', tools.steps, 'Open the cabinet.')
            if len(observed) == 2:
                result = proposal('replan', tools.steps, replacement=replacement)
                result['assessment']['outcome'] = 'satisfied'
                return result
            return proposal('continue', tools.steps, 'Navigate to the counter.')

        self.run_episode(ledger, tools, events, plan, decide, max_decisions=3, max_replans=0)
        self.assertEqual(observed[-1], ('reordered-next', 1))
        self.assertEqual(tools.instructions[0], 'Open the cabinet.')
        self.assertEqual(set(tools.instructions[1:]), {'Navigate to the counter.'})
        self.assertEqual(next(event for event in events if event['event'] == 'plan_replaced')['kind'], 'completion_replan')
        self.assertEqual(ledger.pending(), ['first'])

    def test_metadata_correction_after_completion_does_not_replay_finished_stage(self):
        ledger, tools, events, _ = self.setup_episode('Open the cabinet.', 4)
        plan = [stage('access', 'first', 'Open the cabinet.'), stage('pick', 'first', 'Pick up the bowl.')]
        observed = []

        def decide(active, index, spent, limit):
            observed.append((active[index]['id'], tools.steps))
            if len(observed) == 1:
                return proposal('continue', tools.steps, 'Open the cabinet.')
            if len(observed) == 2:
                result = proposal('continue', tools.steps, 'Pick up the bowl.')
                result['assessment']['outcome'] = 'satisfied'
                result['goal_updates'] = [{'invalid': 'metadata'}]
                return result
            if len(observed) == 3:
                return proposal('complete', tools.steps)
            return proposal('continue', tools.steps, 'Pick up the bowl.')

        self.run_episode(ledger, tools, events, plan, decide, max_decisions=4)
        self.assertEqual(observed, [('access', 0), ('access', 1), ('access', 1), ('pick', 1)])
        self.assertEqual(tools.instructions[0], 'Open the cabinet.')
        self.assertEqual(set(tools.instructions[1:]), {'Pick up the bowl.'})
        invalid = next(event for event in events if event['event'] == 'invalid_decision')
        self.assertIn('finished stage will not be replayed', invalid['feedback'])

    def test_completion_never_replays_old_stage_when_supervision_ends_at_boundary(self):
        ledger, tools, events, _ = self.setup_episode('Open the cabinet.', 2000)
        plan = [stage('access', 'first', 'Open the cabinet.'), stage('pick', 'first', 'Pick up the bowl.')]
        replies = iter([proposal('continue', 0, 'Open the cabinet.'), proposal('complete', 1)])
        reason = self.run_episode(ledger, tools, events, plan, lambda *args: next(replies), max_decisions=2)
        self.assertEqual(reason, 'decision_budget')
        self.assertEqual(tools.steps, 1)
        self.assertEqual(tools.instructions, ['Open the cabinet.'])
        self.assertEqual(len([event for event in events if event['event'] == 'subtask_transition']), 1)

    def test_last_stage_completion_requests_more_planning_without_repeating_it(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.')
        replies = iter([proposal('continue', 0, 'Open the cabinet.'), proposal('complete', 1)])
        self.run_episode(ledger, tools, events, plan, lambda *args: next(replies), max_decisions=2)
        self.assertEqual(tools.instructions, ['Open the cabinet.'])
        self.assertTrue(any(event['event'] == 'planning_window_exhausted' for event in events))

    def test_stale_completion_cannot_advance_stage(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.', 64)
        plan.append(stage('later', 'first', 'Pick up the bowl.'))
        ledger.observe(0, ['current:camera'])

        def decide(*args):
            result = proposal('complete', 0)
            if tools.steps == 0:
                result = proposal('continue', 0, 'Open the cabinet.')
                result['program'] = 'run_groot(instruction="Open the cabinet.", steps=32)'
            return result

        self.run_episode(ledger, tools, events, plan, decide, max_decisions=10)
        self.assertEqual(set(tools.instructions), {'Open the cabinet.'})
        self.assertFalse(any(event['event'] == 'subtask_transition' for event in events))

    def test_actual_task_success_still_ends_episode_early(self):
        ledger, tools, events, plan = self.setup_episode('Open the cabinet.')
        original = tools.run_groot

        def succeed(instruction, steps):
            original(instruction, steps)
            tools.success = True

        tools.run_groot = succeed
        reason = self.run_episode(ledger, tools, events, plan, lambda *args: None, max_decisions=1)
        self.assertEqual(reason, 'task_success')
        self.assertEqual(tools.steps, 32)


class EvidenceContractTests(unittest.TestCase):
    def test_before_and_after_refs_and_paraphrases_do_not_duplicate_or_erase_failures(self):
        ledger = GoalLedger('Required outcome', [definitions()[0]])
        ledger.observe(0, ['current:camera'])
        ledger.observe(32, ['current:camera'])
        history = AttemptHistory(ledger)
        attempt = history.record(['first'], 'run_groot', {'instruction': 'Open the cabinet.'}, None, 0, 32)
        report = {'attempt_id': attempt['attempt_id'], 'outcome': 'no_reported_progress',
                  'evidence': 'Before and after show the same state.', 'observation_refs': refs(0) + refs(32)}
        history.review([report], 32)
        report['evidence'] = 'Current view still shows no progress.'
        history.review([report], 32)
        self.assertEqual(len(history.failures('first')), 1)
        self.assertEqual(history.attempts[0]['previous_reviews'][0]['evidence'], 'Before and after show the same state.')
        report['outcome'] = 'progress'
        with self.assertRaises(ValueError):
            history.review([report], 32)

    def test_before_only_and_future_evidence_remain_invalid(self):
        ledger = GoalLedger('Required outcome', [definitions()[0]])
        ledger.observe(0, ['current:camera'])
        history = AttemptHistory(ledger)
        attempt = history.record(['first'], 'run_groot', {'instruction': 'Open the cabinet.'}, None, 0, 32)
        for at in (0, 33):
            with self.subTest(at=at), self.assertRaises(ValueError):
                history.review([{'attempt_id': attempt['attempt_id'], 'outcome': 'no_reported_progress',
                                 'evidence': 'Claim.', 'observation_refs': refs(at)}], 32)

    def test_catalog_literal_plural_notation_and_stop_sentinel_are_non_destructive(self):
        grounder = AtomicInstructionGrounder()
        for text in ('Open the fridge door.', 'Open the fridge doors.', 'Open the fridge door(s).'):
            grounder.validate(text)
        result = decision('stop', 'stop')
        self.assertIs(validate_decision(result), result)

    def test_switch_requires_current_refs_and_supported_opportunistic_grasp(self):
        ledger = GoalLedger('Required outcome', [definitions()[0]])
        ledger.observe(0, ['current:camera'])
        ledger.observe(32, ['current:camera'])
        result = proposal('continue', 32, 'Navigate to the counter.')
        result['recovery'] = {'change_type': 'prerequisite', 'description': 'Observed obstruction.',
                              'observation_refs': refs(0), 'necessity': 'blocked_access'}
        with self.assertRaises(ValueError):
            validate_stage_switch('Open the cabinet.', 'Navigate to the counter.', result, 32, ledger)
        result['recovery'].update(observation_refs=refs(32), necessity='useful_required_object_held')
        with self.assertRaises(ValueError):
            validate_stage_switch('Open the cabinet.', 'Navigate to the counter.', result, 32, ledger)


if __name__ == '__main__':
    unittest.main()
