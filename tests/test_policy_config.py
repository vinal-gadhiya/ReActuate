"""Backend identity and cadence contracts; run inside the Thor supervisor container."""

import unittest

from core.policy_config import resolve_policy_execution


class PolicyConfigTests(unittest.TestCase):
    def test_backend_mismatch_and_missing_identity_fail_before_execution(self):
        for reported in ('gr00t_n1.5', 'unknown', None):
            metadata = {'action_horizon': 16}
            if reported is not None:
                metadata['policy'] = reported
            with self.subTest(reported=reported), self.assertRaisesRegex(ValueError, 'Expected policy'):
                resolve_policy_execution(metadata, expected_policy='patchwam')
        self.assertEqual(resolve_policy_execution({'policy': 'gr00t_n1.5', 'action_horizon': 16},
                                                 expected_policy='gr00t'), 8)

    def test_recommendation_applies_to_any_backend_and_explicit_interval_wins(self):
        for backend in ('patchwam', 'gr00t_n1.5', 'future_policy'):
            metadata = {'policy': backend, 'action_horizon': 16, 'recommended_action_steps': 16}
            with self.subTest(backend=backend):
                self.assertEqual(resolve_policy_execution(metadata), 16)
                self.assertEqual(resolve_policy_execution(metadata, 8), 8)
                self.assertEqual(resolve_policy_execution(metadata, 8, default_steps=16), 8)
                self.assertEqual(metadata['recommended_action_steps'], 16)

    def test_invalid_cadence_is_rejected_and_legacy_defaults_remain(self):
        metadata = {'action_horizon': 16}
        self.assertEqual(resolve_policy_execution(metadata), 8)
        self.assertEqual(resolve_policy_execution(metadata, default_steps=16), 16)
        for value in (0, -1, 17, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_policy_execution(metadata, value)
            with self.subTest(recommendation=value), self.assertRaises(ValueError):
                resolve_policy_execution({**metadata, 'recommended_action_steps': value})
