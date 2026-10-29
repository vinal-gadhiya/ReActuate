"""API protocol regressions; fake HTTP only, run inside the supervisor container."""

from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from core.policy_http import PolicyHTTPError, RetryableSupervisorError, SupervisorClient

from inference.meta_api import MetaAPI, MetaAPIError, output_text


class MetaAPITests(unittest.TestCase):
    def test_connection_errors_are_retryable_without_leaking_reason_text(self):
        with patch.dict(os.environ, {'MODEL_API_KEY': 'test-key'}):
            api = MetaAPI()
        for error in (URLError('secret URL'), TimeoutError(), ConnectionResetError()):
            with self.subTest(error=type(error).__name__), patch('inference.meta_api.urlopen', side_effect=error):
                with self.assertRaises(MetaAPIError) as caught:
                    api.response('muse-spark-1.3', [])
                self.assertTrue(caught.exception.retryable)
                self.assertNotIn('secret URL', str(caught.exception))

    def test_supervisor_client_preserves_failure_call_count_and_classification(self):
        client = SupervisorClient('http://supervisor')
        body = json.dumps({'error': 'Muse offline', 'retryable': True, 'api_calls': 2,
                           'audit': [{'model': 'sam-3.1'}]})
        with patch.object(client, '_request', side_effect=PolicyHTTPError(502, body)):
            with self.assertRaises(RetryableSupervisorError) as caught:
                client.ask('decide', {}, {})
        self.assertEqual(caught.exception.api_calls, 2)
        self.assertEqual(caught.exception.audit, [{'model': 'sam-3.1'}])
        body = json.dumps({'error': 'Meta HTTP 401', 'retryable': False, 'api_calls': 1})
        with patch.object(client, '_request', side_effect=PolicyHTTPError(502, body)):
            with self.assertRaises(PolicyHTTPError):
                client.ask('decide', {}, {})
        with patch.object(client, '_request', side_effect=URLError('offline')):
            with self.assertRaises(RetryableSupervisorError) as caught:
                client.ask('decide', {'use_sam': True, 'api_calls_left': 1}, {})
        self.assertEqual(caught.exception.api_calls, 1)

    def test_completed_empty_sam_result_is_valid_but_empty_muse_is_not(self):
        empty = {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': ''}]}]}
        with patch.dict(os.environ, {'MODEL_API_KEY': 'test-key'}):
            api = MetaAPI()
        with patch('inference.meta_api.urlopen', return_value=BytesIO(json.dumps(empty).encode())) as post:
            text, audit = api.response('sam-3.1', [])
            self.assertEqual(text, '')
            self.assertEqual(audit['output'], '')
            body = json.loads(post.call_args.args[0].data)
            self.assertNotIn('max_output_tokens', body)
            self.assertNotIn('text', body)
        with self.assertRaises(RuntimeError):
            output_text(empty)
        with self.assertRaises(RuntimeError):
            output_text({'status': 'incomplete', 'output': []}, allow_empty=True)
        with self.assertRaises(RuntimeError):
            output_text({'status': 'completed', 'output': [{'type': 'message', 'content': [
                {'type': 'refusal', 'refusal': 'no'}]}]}, allow_empty=True)

    def test_upstream_error_identifies_model_and_redacts_credentials(self):
        with patch.dict(os.environ, {'MODEL_API_KEY': 'secret-test-key'}):
            api = MetaAPI()
        error = HTTPError('https://api.meta.ai/v1/responses', 400, 'Bad request', {},
                          BytesIO(b'{"error":"unsupported argument; secret-test-key Bearer abc123"}'))
        with patch('inference.meta_api.urlopen', side_effect=error):
            with self.assertRaises(MetaAPIError) as caught:
                api.response('sam-3.1', [])
        message = str(caught.exception)
        self.assertIn('sam-3.1: Meta HTTP 400', message)
        self.assertIn('unsupported argument', message)
        self.assertNotIn('secret-test-key', message)
        self.assertNotIn('abc123', message)
        self.assertFalse(caught.exception.retryable)
