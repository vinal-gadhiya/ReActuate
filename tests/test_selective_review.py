"""Latency and evidence contracts. Run in Thor containers; no live model calls."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from core.selective_review import review_reasons
from core.working_context import working_context
from inference.supervisor import Supervisor
from tests import test_boundary_review as fixtures
from tests.test_boundary_review import assessment, confirmation
from tests.test_goals import proposal


def context(budget=3):
    value = fixtures.BoundaryReviewTests().context(budget)
    value['verification_mode'] = 'selective'
    return value


def candidate(status='continue'):
    value = proposal(status, 32, 'Pick up the mug.' if status == 'continue' else '')
    value['assessment'] = assessment('satisfied' if status == 'complete' else 'unsatisfied')
    return value


def transport(grasp='supported', name='mug'):
    return {'phase': 'carry', 'held_object': name, 'grasp': grasp,
            'destination': 'tray', 'information_needed': 'none',
            'evidence': 'External view shows object retained by fingers.'}


class API:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def response(self, model, content, **kwargs):
        self.calls.append((deepcopy(content), kwargs))
        if not self.responses:
            raise AssertionError('Unexpected additional API call')
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value), {'model': model}


class SelectiveReviewTests(unittest.TestCase):
    def images(self):
        return fixtures.BoundaryReviewTests().images()

    def test_routine_continuation_spends_one_call_and_preserves_exact_program(self):
        planned = candidate()
        api = API(planned)
        result = Supervisor(api, 'muse').handle('decide', context(), self.images())
        self.assertEqual(result['api_calls'], 1)
        self.assertEqual(result['result'], planned)
        self.assertEqual(result['audit'][0]['role'], 'planner')
        self.assertGreaterEqual(result['audit'][0]['wall_seconds'], 0)

    def test_completion_needs_independent_check_and_rejected_claims_do_not_mutate_state(self):
        for check, accepted in ((confirmation(), True), (confirmation('uncertain', 'unknown'), False)):
            with self.subTest(accepted=accepted):
                planned = candidate('complete')
                api = API(planned, check)
                result = Supervisor(api, 'muse').handle('decide', context(), self.images())
                self.assertEqual(result['api_calls'], 2)
                self.assertEqual(result['result']['status'], 'complete' if accepted else 'inspect')
                if not accepted:
                    for field in ('goal_updates', 'attempt_reports', 'memory_updates', 'replacement_steps'):
                        self.assertEqual(result['result'][field], [])
                    self.assertIsNone(result['result']['assessment']['transport'])
                    self.assertIsNone(result['result']['action'])
                    self.assertIsNotNone(result['verification_feedback'])

    def test_budget_exhaustion_cannot_accept_unverified_completion(self):
        api = API(candidate('complete'))
        result = Supervisor(api, 'muse').handle('decide', context(1), self.images())
        self.assertEqual(result['result']['status'], 'inspect')
        self.assertEqual(result['api_calls'], 1)

    def test_output_limit_retry_preserves_selective_verification_and_accounting(self):
        truncated = fixtures.BoundaryReviewTests.output_limit_error()
        api = API(candidate('complete'), truncated, confirmation())
        result = Supervisor(api, 'muse').handle('decide', context(3), self.images())
        self.assertEqual(result['api_calls'], 3)
        self.assertEqual(result['result']['status'], 'complete')
        self.assertEqual([item['role'] for item in result['audit']],
                         ['planner', 'output_limit', 'selective_verification'])

    def test_justified_prerequisite_repair_can_proceed_without_stage_completion(self):
        planned = candidate()
        planned['recovery'] = {'necessity': 'blocked_access', 'change_type': 'prerequisite',
            'description': 'Repair observed access before retrying the pending goal.',
            'observation_refs': assessment()['observation_refs']}
        api = API(planned, confirmation())
        result = Supervisor(api, 'muse').handle('decide', context(), self.images())
        self.assertEqual(result['result'], planned)
        self.assertEqual(result['result']['assessment']['outcome'], 'unsatisfied')
        self.assertEqual(result['api_calls'], 2)
        self.assertNotIn('stage_completion', result['audit'][-1]['reasons'])

    def test_false_loss_of_cross_camera_supported_grasp_is_checked_before_recovery(self):
        public = context()
        public['scene_memory']['assessment'] = {'step': 0, 'outcome': 'satisfied', 'transport': transport()}
        planned = candidate()
        planned['assessment']['transport'] = transport('empty')
        planned['memory_updates'] = [{'name': 'mug', 'kind': 'object', 'location': 'unknown',
                                      'status': 'lost', 'evidence': 'Hidden in wrist view.'}]
        api = API(planned, confirmation('incomplete', 'contradicted'))
        result = Supervisor(api, 'muse').handle('decide', public, self.images())
        self.assertEqual(result['result']['status'], 'inspect')
        self.assertEqual(result['result']['memory_updates'], [])
        self.assertIn('grasp_loss_or_release', result['audit'][-1]['reasons'])
        self.assertIn('another occluded view', api.calls[-1][1]['instructions'])

    def test_recovery_and_goal_changes_are_checked_for_non_transport_skills(self):
        for instruction in ('Close the cabinet.', 'Turn on the sink faucet.',
                            'Navigate to the dining counter.', 'Close the blender lid.'):
            with self.subTest(instruction=instruction):
                public = context()
                public['step']['instruction'] = instruction
                planned = candidate('inspect')
                planned['goal_updates'] = [{'goal_id': 'goal', 'status': 'satisfied',
                                            'evidence': 'Effect claimed.', 'observation_refs': assessment()['observation_refs']}]
                # Pure scheduling test: universal claim analysis is independent
                # of task identity, atomic phrasing or a transport assessment.
                self.assertIn('original_goal_status_change', review_reasons(planned, public))
                planned['recovery'] = {'necessity': 'blocked_access'}
                self.assertIn('necessary_intervention', review_reasons(planned, public))

    def test_all_current_before_views_survive_and_temporal_verifier_keeps_full_evidence(self):
        public, images = context(), self.images()
        for label, value in (('current:video.left', 32), ('before:video.left', 0),
                             ('current:video.right', 32), ('before:video.right', 0),
                             ('history_8:video.wrist', 8)):
            images[label] = np.full((4, 6, 3), value, np.uint8)
        public['temporal_observations'] = [{'label': 'history_8:video.wrist',
                                           'step': 8, 'camera': 'current:video.wrist'}]
        public['observation_refs'].extend({'step': at, 'camera': camera}
            for camera in ('current:video.left', 'current:video.right') for at in (0, 32))
        original = deepcopy(public)
        api = API(candidate('complete'), confirmation())
        result = Supervisor(api, 'muse').handle('decide', public, images)
        self.assertEqual(public, original)
        self.assertEqual(result['supervision_metrics']['planner_rgb_images'], 7)
        self.assertEqual(result['supervision_metrics']['verifier_rgb_images_available'], 7)
        verifier_labels = [item['text'] for item in api.calls[1][0][1:] if item['type'] == 'input_text']
        self.assertEqual(set(verifier_labels), set(images))
        payload = json.loads(api.calls[0][0][0]['text'])['context']
        self.assertEqual(payload['temporal_observations'], public['temporal_observations'])
        self.assertTrue(any(item['step'] == 8 for item in payload['observation_refs']))

    def test_image_encoding_is_reused_between_planner_and_verifier(self):
        from inference.meta_api import image_part
        with patch('inference.supervisor.image_part', wraps=image_part) as encoder:
            Supervisor(API(candidate('complete'), confirmation()), 'muse').handle(
                'decide', context(), self.images())
        self.assertEqual(encoder.call_count, len(self.images()))

    def test_compact_context_retains_dated_identity_without_copying_whole_journal(self):
        public = context()
        public['scene_memory'] = {'assessment': {'step': 8, 'outcome': 'satisfied', 'transport': transport()},
            'entries': [{'name': 'mug', 'observation_step': 8,
                         'identity_anchor': {'step': 0, 'evidence': 'External view'},
                         'observations': [{'step': at} for at in (0, 8, 16)]}]}
        compact = working_context(public)
        self.assertEqual(compact['previous_transport_claim']['observation_step'], 8)
        self.assertEqual(compact['beliefs'][0]['observations'], [{'step': 8}, {'step': 16}])
        self.assertNotIn('scene_memory', compact)


if __name__ == '__main__':
    unittest.main()
