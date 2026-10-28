"""API protocol regressions; fake HTTP only, run inside the supervisor container."""

from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from inference.meta_api import MetaAPI, MetaAPIError, output_text


class MetaAPITests(unittest.TestCase):
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
