"""Decomposed planning and low-height policy release. Run in Thor containers."""

import unittest

from core.atomic_language import AtomicInstructionGrounder
from core.boundary_review import opening_stage
from core.release import RELEASE_SCHEMA, validate_release
from core.goals import GoalLedger
from inference.supervisor import Supervisor
from sim.run_agent import run_subtasks
from tests import test_boundary_review as fixtures
from tests.test_agent import FakeTools
from tests.test_goals import definitions, proposal, refs, stage


def release_decision(at=32):
    value = proposal('continue', at, 'Open the gripper.')
    value['program'] = 'run_groot(instruction="Open the gripper.", steps=16)'
    value['assessment']['outcome'] = 'unsatisfied'
    value['assessment']['transport'] = {'grasp': 'supported', 'held_object': 'mug', 'phase': 'place',
        'destination': 'tray', 'information_needed': 'release', 'evidence': 'Mug held low within tray.'}
    value['recovery'] = {'change_type': 'prerequisite', 'necessity': 'controlled_release',
        'description': 'Correct object held low inside required destination; release then recheck placement.',
        'observation_refs': refs(at)}
    return value


def review(status='observed', at=32):
    return {'verdict': 'confirmed', 'evidence': 'Current views expose the low aligned held object.',
        'observation_refs': [{'step': at, 'camera': 'current:video.wrist'}],
        **{key: {'status': status, 'evidence': 'Condition exposed in useful current view.',
                 'observation_refs': [{'step': at, 'camera': 'current:video.wrist'}]}
           for key in ('supported_target_hold', 'destination_alignment', 'low_release_height')}}


class DecomposedReleaseTests(unittest.TestCase):
    def context(self):
        value = fixtures.BoundaryReviewTests().context(2)
        value['boundary_review_enabled'] = False
        value['step']['instruction'] = 'Place the mug on the tray.'
        value['release_step_limit'] = 16
        return value

    def test_generated_plans_start_decomposed_but_explicit_transfer_text_is_preserved(self):
        grounder = AtomicInstructionGrounder()
        combined = 'Pick the mug from the counter and place it on the tray.'
        grounder.validate_plan([stage('original', 'first', combined)])
        with self.assertRaises(ValueError):
            grounder.validate_generated_plan([stage('generated', 'first', combined)])
        grounder.validate_generated_plan([stage('pick', 'first', 'Pick the mug from the counter.'),
            stage('place', 'first', 'Place the mug on the tray.')])
        grounder.validate_generated_plan([stage('pick', 'first', 'Pick the mug from the cabinet.'),
            stage('travel', 'first', 'Navigate to the dining counter.'),
            stage('place', 'first', 'Place the mug on the tray.')])

    def test_release_extension_is_not_a_fixture_or_an_upstream_atomic_annotation(self):
        grounder = AtomicInstructionGrounder()
        match = grounder.validate('Open the gripper.')
        self.assertEqual(match['source_kind'], 'operator_requested_harness_extension')
        self.assertEqual(match['source_tasks'], [])
        self.assertFalse(opening_stage({'step': {'instruction': 'Open the gripper.'}}))
        with self.assertRaises(ValueError):
            grounder.validate_generated_plan([stage('release', 'first', 'Open the gripper.')])
        with self.assertRaises(ValueError):
            grounder.validate('Open the right gripper.')

    def test_each_release_condition_needs_current_observed_evidence(self):
        context = self.context()
        for key in ('supported_target_hold', 'destination_alignment', 'low_release_height'):
            for state in ('unknown', 'contradicted'):
                candidate = review()
                candidate[key]['status'] = state
                with self.subTest(key=key, state=state), self.assertRaises(ValueError):
                    validate_release(candidate, context)
            candidate = review()
            candidate[key]['observation_refs'][0]['step'] = 0
            with self.assertRaises(ValueError):
                validate_release(candidate, context)

    def test_supervisor_checks_release_even_with_normal_boundary_review_disabled(self):
        for status, expected in (('observed', 'continue'), ('unknown', 'inspect'), ('contradicted', 'inspect')):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs)
                    return (review(status) if kwargs['schema'] == RELEASE_SCHEMA else release_decision()), {'model': model}

            reply = Supervisor(API(), 'muse').handle('decide', self.context(), fixtures.BoundaryReviewTests().images())
            self.assertEqual(reply['result']['status'], expected)
            self.assertEqual(reply['api_calls'], 2)
            if expected == 'continue':
                permit = reply['release_confirmation']
                self.assertEqual((permit['stage_id'], permit['step']), ('a', 32))
                self.assertEqual(permit['program'], reply['result']['program'])
                self.assertEqual(reply['result']['assessment']['outcome'], 'unsatisfied')
            else:
                self.assertEqual(reply['result']['program'], '')

    def test_no_review_allowance_cannot_authorize_release(self):
        class API:
            def response(self, *args, **kwargs):
                return release_decision(), {'model': 'muse'}

        context = self.context()
        context['api_calls_left'] = 1
        result = Supervisor(API(), 'muse').handle('decide', context, fixtures.BoundaryReviewTests().images())
        self.assertEqual(result['result']['status'], 'inspect')

    def test_runner_requires_fresh_permission_and_preserves_pending_placement(self):
        for authorized in (False, True):
            tools, events = FakeTools(), []
            tools.max_steps, tools.chunk_steps = 32, 16
            ledger = GoalLedger('Place mug on tray', [definitions()[0]])
            count = []

            def decide(*args):
                count.append(tools.steps)
                ledger.observe(tools.steps, ['current:camera'])
                if len(count) == 1:
                    return release_decision(tools.steps)
                value = proposal('continue', tools.steps, 'Place the mug on the tray.')
                value['program'] = 'run_groot(instruction="Place the mug on the tray.", steps=16)'
                return value

            run_subtasks(tools, [stage('place', 'first', 'Place the mug on the tray.')], 'agent', decide,
                events.append, execution_policy='episode', decision_steps=32, max_decisions=5,
                goal_ledger=ledger, instruction_grounder=AtomicInstructionGrounder(),
                release_authorizer=lambda decision, step, active: authorized)
            self.assertEqual('Open the gripper.' in tools.instructions, authorized)
            self.assertIn('first', ledger.pending())
            self.assertFalse(any(event['event'] == 'subtask_transition' for event in events))
            self.assertFalse(any(event.get('tool') == 'set_gripper' for event in events))


if __name__ == '__main__':
    unittest.main()
