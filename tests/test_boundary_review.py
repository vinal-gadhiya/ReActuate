"""Temporal and independent-review contracts; execute in Thor containers."""

from copy import deepcopy
import json
import unittest

import numpy as np

from core.boundary_review import temporal_snapshots, validate_boundary_review, validate_confirmation
from inference.supervisor import Supervisor
from inference.meta_api import MetaAPIError
from tests.test_agent import step
from tests.test_execution_policy import proposal


def assessment(outcome='satisfied', observed_step=32, camera='current:video.wrist'):
    return {'outcome': outcome, 'evidence': 'The current view supports the active stage effect.',
            'observation_refs': [{'step': observed_step, 'camera': camera}], 'transport': None}


def confirmation(verdict='confirmed', status='observed'):
    return {'verdict': verdict, 'evidence': 'Current views establish the required physical effect.',
            'observation_refs': assessment()['observation_refs'],
            'checks': [{'criterion': 'Required physical effect and usable state.', 'status': status,
                        'evidence': 'Current camera exposes this condition.',
                        'observation_refs': assessment()['observation_refs']}]}


class BoundaryReviewTests(unittest.TestCase):
    def context(self, budget=3):
        active = step('a', 'Pick up the mug.')
        active['success_criteria'] = 'Mug supported by gripper and separated from source support.'
        return {'boundary_review_enabled': True, 'api_calls_left': budget,
                'observation_step': 32, 'before_observation_step': 0,
                'observation_refs': [{'step': value, 'camera': 'current:video.wrist'} for value in (0, 8, 24, 32)],
                'step': active, 'decision_step_limit': 32,
                'capabilities': {'set_gripper': False},
                'scene_memory': {'entries': [{'name': 'mug', 'evidence': 'Old failure narrative'}]},
                'attempt_history': {'old_failure': True}}

    def images(self):
        return {f'{label}:video.wrist': np.full((4, 6, 3), value, np.uint8)
                for label, value in (('before', 0), ('current', 32))}

    def test_temporal_samples_preserve_pixels_and_do_not_duplicate_before_or_current(self):
        frames = [(value, {'video.wrist': np.full((4, 6, 3), value, np.uint8)})
                  for value in (0, 8, 16, 24, 32)]
        images, mapping = temporal_snapshots(frames, 32, 0, ['video.wrist'])
        self.assertEqual([item['step'] for item in mapping], [8, 24])
        for item in mapping:
            np.testing.assert_array_equal(images[item['label']], item['step'])
        images['history_8:video.wrist'][:] = 99
        self.assertTrue(np.all(frames[1][1]['video.wrist'] == 8))
        self.assertEqual(temporal_snapshots(frames[:1], 0, None, ['video.wrist']), ({}, []))

    def test_stale_future_and_fabricated_camera_claims_cannot_complete_stage(self):
        context = self.context()
        for candidate in (assessment(observed_step=0), assessment(observed_step=40),
                          assessment(camera='current:invented')):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_boundary_review({'assessment': candidate}, context)

    def test_supported_boundary_completes_without_planning_or_motion_for_multiple_families(self):
        for instruction, criterion in (
                ('Pick up the mug.', 'Mug supported by the gripper, separated from source support.'),
                ('Open the cabinet.', 'Cabinet provides usable clearance to required workspace.'),
                ('Place the bowl on the tray.', 'Bowl supported by tray after release.'),
                ('Navigate to the dining counter.', 'Robot has arrived at the dining counter.'),
                ('Turn on the sink faucet.', 'Water is visibly flowing from the faucet.')):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append((content, kwargs))
                    if len(calls) == 2:
                        return confirmation(), {'model': model}
                    return {'assessment': assessment()}, {'model': model}

            context = self.context()
            context['step']['instruction'] = instruction
            context['step']['success_criteria'] = criterion
            with self.subTest(instruction=instruction):
                reply = Supervisor(API(), 'muse').handle('decide', context, self.images())
                self.assertEqual(reply['api_calls'], 2)
                self.assertEqual(reply['result']['status'], 'complete')
                self.assertEqual(reply['result']['program'], '')
                self.assertIsNone(reply['result']['action'])
                self.assertEqual(reply['result']['goal_updates'], [])
                self.assertEqual(len(calls), 2)
                self.assertEqual(reply['completion_confirmation']['verdict'], 'confirmed')
                reviewer_context = json.loads(calls[0][0][0]['text'])
                self.assertNotIn('scene_memory', reviewer_context)
                self.assertNotIn('attempt_history', reviewer_context)

    def test_partial_effect_or_unknown_clearance_cannot_end_stage(self):
        for verdict, status in (('incomplete', 'contradicted'), ('uncertain', 'unknown'), ('confirmed', 'unknown')):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs['schema'])
                    if len(calls) == 1:
                        return {'assessment': assessment()}, {'model': model}
                    return confirmation(verdict, status), {'model': model}

            with self.subTest(verdict=verdict, status=status):
                reply = Supervisor(API(), 'muse').handle('decide', self.context(2), self.images())
                self.assertEqual(reply['api_calls'], 2)
                self.assertEqual(reply['result']['status'], 'inspect')
                self.assertEqual(reply['result']['assessment']['outcome'], 'unknown')
                self.assertEqual(reply['result']['program'], '')
                self.assertEqual([entry['role'] for entry in reply['audit']],
                                 ['boundary_review', 'completion_confirmation'])

    def test_planner_cannot_bypass_confirmation_or_exhausted_confirmation_budget(self):
        for budget in (1, 3):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs['schema'])
                    if budget == 3 and len(calls) == 1:
                        return {'assessment': assessment('unknown')}, {'model': model}
                    if budget == 3 and len(calls) == 3:
                        return confirmation('uncertain', 'unknown'), {'model': model}
                    result = proposal('complete', 32)
                    result['assessment'] = assessment()
                    return result, {'model': model}

            with self.subTest(budget=budget):
                reply = Supervisor(API(), 'muse').handle('decide', self.context(budget), self.images())
                self.assertEqual(reply['api_calls'], budget)
                self.assertEqual(reply['result']['status'], 'inspect')
                self.assertEqual(reply['result']['program'], '')

    def test_confirmation_checks_require_current_supplied_evidence(self):
        for kind in ('empty', 'stale', 'future', 'camera'):
            result = confirmation()
            if kind == 'empty':
                result['checks'] = []
            else:
                result['checks'][0]['observation_refs'] = assessment(
                    observed_step=0 if kind == 'stale' else 99 if kind == 'future' else 32,
                    camera='invented' if kind == 'camera' else 'current:video.wrist')['observation_refs']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_confirmation(result, self.context())

    def test_confirmation_failure_preserves_review_usage_without_claiming_completion(self):
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append(kwargs['schema'])
                if len(calls) == 1:
                    return {'assessment': assessment()}, {'model': model}
                raise MetaAPIError('temporary confirmation failure', retryable=True)

        with self.assertRaises(MetaAPIError) as caught:
            Supervisor(API(), 'muse').handle('decide', self.context(2), self.images())
        self.assertEqual(caught.exception.api_calls, 2)
        self.assertEqual(len(caught.exception.audit), 1)
        self.assertEqual(caught.exception.audit[0]['role'], 'boundary_review')

    def test_uncertain_or_invalid_review_preserves_planner_and_api_accounting(self):
        for candidate in (assessment('unknown'), assessment(observed_step=999)):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(kwargs['schema'])
                    if len(calls) == 1:
                        return {'assessment': candidate}, {'model': model}
                    return proposal('continue', 32, 'Pick up the mug.'), {'model': model}

            reply = Supervisor(API(), 'muse').handle('decide', self.context(2), self.images())
            self.assertEqual(reply['api_calls'], 2)
            self.assertEqual(reply['result']['status'], 'continue')
            self.assertEqual(len(calls), 2)

    def test_low_budget_skips_reviewer_and_sam_to_preserve_one_planning_call(self):
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append(model)
                return proposal('continue', 32, 'Pick up the mug.'), {'model': model}

        reply = Supervisor(API(), 'muse', use_sam=True).handle('decide', self.context(1), self.images())
        self.assertEqual(reply['api_calls'], 1)
        self.assertEqual(calls, ['muse'])
        self.assertIsNone(reply['boundary_review'])

    def test_reviewer_sam_and_planner_share_budget_and_transient_error_keeps_usage(self):
        for fail_planner in (False, True):
            calls = []

            class API:
                def response(self, model, content, **kwargs):
                    calls.append(model)
                    if len(calls) == 1:
                        return {'assessment': assessment('unknown')}, {'model': model}
                    if model == 'sam-3.1':
                        return '', {'model': model}
                    if fail_planner:
                        raise MetaAPIError('temporary failure', retryable=True)
                    return proposal('continue', 32, 'Pick up the mug.'), {'model': model}

            with self.subTest(fail_planner=fail_planner):
                supervisor = Supervisor(API(), 'muse', use_sam=True)
                if fail_planner:
                    with self.assertRaises(MetaAPIError) as caught:
                        supervisor.handle('decide', self.context(3), self.images())
                    self.assertEqual(caught.exception.api_calls, 3)
                    self.assertEqual(len(caught.exception.audit), 2)
                    self.assertEqual(caught.exception.audit[0]['role'], 'boundary_review')
                else:
                    reply = supervisor.handle('decide', self.context(3), self.images())
                    self.assertEqual(reply['api_calls'], 3)
                self.assertEqual(calls, ['muse', 'sam-3.1', 'muse'])

    def test_historical_frames_have_validated_dates_and_reach_both_review_and_planner(self):
        calls = []

        class API:
            def response(self, model, content, **kwargs):
                calls.append(content)
                if len(calls) == 1:
                    return {'assessment': assessment('unknown')}, {'model': model}
                return proposal('continue', 32, 'Pick up the mug.'), {'model': model}

        context, images = self.context(2), self.images()
        extra, mapping = temporal_snapshots([(value, {'video.wrist': np.full((4, 6, 3), value, np.uint8)})
                                            for value in (8, 24)], 32, 0, ['video.wrist'])
        context['temporal_observations'] = mapping
        Supervisor(API(), 'muse').handle('decide', context, {**images, **extra})
        self.assertEqual(len(calls), 2)
        for content in calls:
            labels = [item.get('text') for item in content if item['type'] == 'input_text']
            self.assertIn('history_8:video.wrist', labels)
            self.assertIn('history_24:video.wrist', labels)
        invalid = deepcopy(context)
        invalid['temporal_observations'][0]['step'] = 99
        with self.assertRaises(ValueError):
            Supervisor(API(), 'muse').handle('decide', invalid, {**images, **extra})
        self.assertEqual(len(calls), 2)
