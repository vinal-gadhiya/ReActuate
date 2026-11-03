"""Strict placement handoff contracts. Run only in Thor containers."""

import unittest

from core.boundary_review import placement_stage, validate_confirmation, PLACEMENT_CONFIRMATION_SCHEMA
from inference.supervisor import Supervisor
from tests import test_boundary_review as fixtures


class PlacementReleaseTests(unittest.TestCase):
    def context(self, instruction='Place the mug on the tray.'):
        public = fixtures.BoundaryReviewTests().context()
        public['step']['instruction'] = instruction
        # Deliberately old, weaker criteria: the new handoff must still apply.
        public['step']['success_criteria'] = 'Object placed at destination.'
        return public

    def test_placement_and_full_transfer_require_the_same_strict_handoff(self):
        for instruction in ('Place the mug on the tray.', 'Put the item in the bowl.',
                            'Pick the mug from the counter and place it on the tray.',
                            'Pick up the mug and place it in the cabinet.',
                            'Add the brownie to the tupperware that contains the apple.'):
            with self.subTest(instruction=instruction):
                context = self.context(instruction)
                self.assertTrue(placement_stage(context))
                self.assertEqual(validate_confirmation(fixtures.placement_confirmation(), context)['verdict'], 'confirmed')
                for index in range(3):
                    for status in ('unknown', 'contradicted'):
                        statuses = ['observed'] * 3
                        statuses[index] = status
                        with self.assertRaises(ValueError):
                            validate_confirmation(fixtures.placement_confirmation(statuses), context)

    def test_gripper_near_or_occluded_cannot_advance_even_if_support_and_release_claimed(self):
        for state in ('unknown', 'contradicted'):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        return {'assessment': fixtures.assessment()}, {'model': model}
                    return fixtures.placement_confirmation(('observed', 'observed', state)), {'model': model}

            reply = Supervisor(API(), 'muse').handle('decide', self.context(), fixtures.BoundaryReviewTests().images())
            self.assertEqual(reply['result']['status'], 'inspect')
            self.assertEqual(reply['result']['program'], '')
            self.assertIsNone(reply['result']['action'])
            self.assertIsNone(reply['result']['assessment']['transport'])
            self.assertEqual(calls[1]['schema'], PLACEMENT_CONFIRMATION_SCHEMA)

    def test_all_three_observed_with_current_refs_can_advance(self):
        class API:
            def response(self, model, content, **kwargs):
                if kwargs['schema'] == PLACEMENT_CONFIRMATION_SCHEMA:
                    return fixtures.placement_confirmation(), {'model': model}
                return {'assessment': fixtures.assessment()}, {'model': model}

        reply = Supervisor(API(), 'muse').handle('decide', self.context(), fixtures.BoundaryReviewTests().images())
        self.assertEqual(reply['result']['status'], 'complete')
        self.assertEqual(reply['api_calls'], 2)
        self.assertEqual(reply['result']['program'], '')

    def test_stale_release_or_clearance_evidence_and_missing_checks_are_rejected(self):
        for key in ('object_released', 'gripper_clear'):
            result = fixtures.placement_confirmation()
            result['placement'][key]['observation_refs'][0]['step'] = 0
            with self.assertRaises(ValueError):
                validate_confirmation(result, self.context())
        result = fixtures.placement_confirmation()
        del result['placement']['gripper_clear']
        with self.assertRaises(ValueError):
            validate_confirmation(result, self.context())

    def test_supported_grasp_conflict_cannot_be_confirmed_as_placement(self):
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append(kwargs)
                assessment = fixtures.assessment()
                assessment['transport'] = {'phase': 'place', 'held_object': 'mug', 'grasp': 'supported',
                    'destination': 'tray', 'information_needed': 'release', 'evidence': 'Mug is still between fingers.'}
                return {'assessment': assessment}, {'model': model}

        reply = Supervisor(API(), 'muse').handle('decide', self.context(), fixtures.BoundaryReviewTests().images())
        self.assertEqual(reply['result']['status'], 'inspect')
        self.assertEqual(len(calls), 1)

    def test_pickup_navigation_and_controls_are_not_placement_handoffs(self):
        for instruction in ('Pick up the mug.', 'Pick the mug from the cabinet.',
                            'Navigate to the dining counter.', 'Turn on the blender.', 'Open the cabinet.'):
            self.assertFalse(placement_stage(self.context(instruction)))


if __name__ == '__main__':
    unittest.main()
