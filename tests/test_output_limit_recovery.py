"""Truncated supervisor output is retryable, accounted, and authorizes no motion.

Run in Thor containers only; fake APIs and robot tools do not establish live quality.
"""

import unittest

from core.policy_http import RetryableSupervisorError
from core.goals import GoalLedger
from inference.meta_api import MetaAPIError
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests import test_boundary_review as fixtures
from tests.test_agent import FakeTools
from tests.test_goals import definitions, proposal, stage


def truncated():
    error = MetaAPIError('max_output_tokens', retryable=True, kind='output_limit',
                         allow_policy_continuation=False)
    error.audit = [{'response_id': 'truncated', 'status': 'incomplete',
                    'usage': {'output_tokens': 4096}, 'max_output_tokens': 4096}]
    return error


class OutputLimitRecoveryTests(unittest.TestCase):
    def test_confirmation_retry_keeps_panel_schema_and_failed_usage(self):
        calls = []

        class API:
            max_output_tokens = 4096

            def response(self, model, content, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return {'assessment': fixtures.assessment()}, {'model': model}
                if len(calls) == 2:
                    raise truncated()
                return fixtures.opening_confirmation(), {'model': model}

        context = fixtures.BoundaryReviewTests().context(3)
        context['step']['instruction'] = 'Open the cabinet doors.'
        reply = Supervisor(API(), 'muse').handle('decide', context, fixtures.BoundaryReviewTests().images())
        self.assertEqual(reply['api_calls'], 3)
        self.assertEqual(reply['result']['status'], 'complete')
        self.assertEqual(reply['audit'][1]['role'], 'output_limit')
        self.assertEqual(reply['audit'][1]['usage']['output_tokens'], 4096)
        self.assertEqual(calls[2]['max_output_tokens'], 8192)
        self.assertEqual(calls[1]['schema'], calls[2]['schema'])

    def test_retry_is_bounded_and_does_not_spend_reserved_planner_budget(self):
        for budget, expected_calls in ((1, 1), (2, 1), (3, 2)):
            calls = []

            class API:
                def response(self, *args, **kwargs):
                    calls.append(kwargs)
                    raise truncated()

            with self.subTest(budget=budget), self.assertRaises(MetaAPIError) as caught:
                Supervisor(API(), 'muse').handle('decide', fixtures.BoundaryReviewTests().context(budget),
                                                fixtures.BoundaryReviewTests().images())
            self.assertEqual(len(calls), expected_calls)
            self.assertEqual(caught.exception.api_calls, expected_calls)
            self.assertEqual(len(caught.exception.audit), expected_calls)
            self.assertFalse(caught.exception.allow_policy_continuation)

    def test_repeated_output_limit_ends_with_structured_failure_without_motion(self):
        ledger = GoalLedger('Original outcomes', [definitions()[0]])
        tools, events = FakeTools(), []

        def decide(*args):
            raise RetryableSupervisorError('output limit', allow_policy_continuation=False)

        reason = run_subtasks(tools, [stage('pick', 'first', 'Pick up the bowl.')], 'agent', decide,
            events.append, execution_policy='episode', goal_ledger=ledger,
            max_decisions=10, max_supervisor_failures=2)
        self.assertEqual(reason, 'supervisor_failure_limit')
        self.assertEqual(tools.steps, 0)
        self.assertEqual(tools.instructions, [])
        self.assertFalse(any(event['event'] == 'policy_continuation' for event in events))

    def test_fresh_valid_decision_can_resume_after_truncation(self):
        ledger = GoalLedger('Original outcomes', [definitions()[0]])
        tools, events, seen = FakeTools(), [], []
        tools.max_steps = 1

        def decide(*args):
            seen.append(tools.steps)
            if len(seen) == 1:
                raise RetryableSupervisorError('output limit', allow_policy_continuation=False)
            ledger.observe(tools.steps, ['current:camera'])
            return proposal('continue', tools.steps, 'Pick up the bowl.')

        reason = run_subtasks(tools, [stage('pick', 'first', 'Pick up the bowl.')], 'agent', decide,
            events.append, execution_policy='episode', goal_ledger=ledger, max_decisions=4)
        self.assertEqual(reason, 'episode_step_budget')
        self.assertEqual(seen, [0, 0])
        self.assertEqual(tools.instructions, ['Pick up the bowl.'])


if __name__ == '__main__':
    unittest.main()
