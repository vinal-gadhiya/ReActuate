"""Offline case labels must not enter supervisor inputs; no real API calls."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from sim.review_cases import evaluate_cases


class ReviewCaseTests(unittest.TestCase):
    def test_privileged_expected_outcomes_are_used_only_after_model_response(self):
        seen = []

        class Client:
            def ask(self, operation, context, images):
                seen.append(context)
                return {'api_calls': 1, 'result': {'assessment': {'outcome': 'satisfied'}}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.png'
            Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(path)
            case = {'id': 'case', 'episode_id': 'episode', 'partition': 'held_out', 'operation': 'decide',
                    'context': {'original_goal': 'General outcome', 'api_calls_left': 2},
                    'images': [{'label': 'current:camera', 'path': str(path)}],
                    'expected': {'outcome': 'unsatisfied', 'label_source': 'offline simulator check'}}
            result = evaluate_cases({'cases': [case]}, Client())
            self.assertEqual(result['summary']['held_out']['false_positive_rate'], 1)
            self.assertNotIn('expected', seen[0])
            self.assertNotIn('label_source', seen[0])

    def test_unlabelled_real_observations_do_not_spend_api_calls(self):
        class Client:
            def ask(self, *args):
                raise AssertionError('No request before annotation')

        case = {'id': 'case', 'episode_id': 'episode', 'partition': 'unassigned', 'expected': None}
        result = evaluate_cases({'cases': [case]}, Client())
        self.assertEqual(result['results'][0]['status'], 'unlabelled')

    def test_partitions_are_separated_by_episode_not_individual_frames(self):
        cases = [{'id': str(index), 'episode_id': 'same', 'partition': partition, 'expected': None}
                 for index, partition in enumerate(('development', 'held_out'))]
        with self.assertRaises(ValueError):
            evaluate_cases({'cases': cases}, None)

    def test_invalid_annotations_are_rejected_before_api_calls(self):
        class Client:
            def ask(self, *args):
                raise AssertionError('Invalid annotations must not spend API calls')

        case = {'id': 'case', 'episode_id': 'episode', 'partition': 'held_out', 'operation': 'plan',
                'expected': {'eligible_first_targets': []}}
        result = evaluate_cases({'cases': [case]}, Client())
        self.assertEqual(result['results'][0]['status'], 'error')
