"""Meta Responses API client. Credentials are read only from the environment."""

import base64
from io import BytesIO
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


def image_part(frame):
    stream = BytesIO()
    Image.fromarray(frame).save(stream, format='PNG')
    return {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}


class MetaAPIError(RuntimeError):
    """An upstream error with a bounded, credential-redacted diagnostic."""

    def __init__(self, message, retryable=False, kind=None, allow_policy_continuation=True):
        super().__init__(message)
        self.retryable = retryable
        self.api_calls = 1
        self.audit = []
        self.kind = kind
        self.allow_policy_continuation = allow_policy_continuation


def output_text(response, allow_empty=False):
    if response.get('status') != 'completed':
        reason = response.get('incomplete_details', {}).get('reason') if isinstance(response.get('incomplete_details'), dict) else None
        raise RuntimeError(f'Meta response did not complete: {response.get("status")}; reason={reason}')
    if any(part.get('type') == 'refusal' for item in response.get('output', [])
           for part in item.get('content', [])):
        raise RuntimeError('Meta returned a refusal')
    text = ''.join(
        part.get('text', '')
        for item in response.get('output', []) if item.get('type') == 'message'
        for part in item.get('content', []) if part.get('type') == 'output_text'
    )
    if not text and not allow_empty:
        raise RuntimeError('Meta returned no output text (possibly a refusal)')
    return text


class MetaAPI:
    def __init__(self, timeout=120):
        self.key = os.environ.get('MODEL_API_KEY')
        if not self.key:
            raise RuntimeError('MODEL_API_KEY is not set in the supervisor container')
        self.base_url = os.environ.get('META_BASE_URL', 'https://api.meta.ai/v1').rstrip('/')
        self.timeout = timeout
        self.sam_timeout = int(os.environ.get('META_SAM_TIMEOUT', '180'))
        if not 1 <= self.sam_timeout <= 240:
            raise ValueError('META_SAM_TIMEOUT must be 1–240 seconds')
        self.max_output_tokens = int(os.environ.get('META_MAX_OUTPUT_TOKENS', '8192'))
        if not 1024 <= self.max_output_tokens <= 16384:
            raise ValueError('META_MAX_OUTPUT_TOKENS must be 1024–16384')

    def response(self, model, content, instructions=None, schema=None, max_output_tokens=None):
        body = {
            'model': model, 'input': [{'role': 'user', 'content': content}],
            'store': False,
        }
        # SAM returns segmentation records, not generated language. Keep Muse's
        # generation cap out of segmentation requests.
        if model != 'sam-3.1':
            limit = self.max_output_tokens if max_output_tokens is None else max_output_tokens
            if type(limit) is not int or not 1024 <= limit <= 16384:
                raise ValueError('max_output_tokens must be 1024–16384')
            body['max_output_tokens'] = limit
        else:
            body.pop('store')
        if instructions is not None:
            body['instructions'] = instructions
        if schema is not None:
            body['text'] = {'format': {
                'type': 'json_schema', 'name': 'robot_decision', 'strict': True, 'schema': schema,
            }}
        request = Request(
            self.base_url + '/responses', data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
        )
        try:
            with urlopen(request, timeout=self.sam_timeout if model == 'sam-3.1' else self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except HTTPError as error:
            detail = error.read(4096).decode('utf-8', errors='replace')
            detail = detail.replace(self.key, '[REDACTED]')
            detail = re.sub(r'data:(?:image|video)/[^\s"\\]+', '[MEDIA]', detail)
            detail = re.sub(r'(?i)Bearer\s+[^\s"\\]+', 'Bearer [REDACTED]', detail)
            raise MetaAPIError(f'{model}: Meta HTTP {error.code}: {detail[:1200]}',
                               retryable=error.code in (429, 500, 502, 503, 504)) from None
        except TimeoutError:
            timeout = self.sam_timeout if model == 'sam-3.1' else self.timeout
            raise MetaAPIError(f'{model}: Meta request timed out after {timeout}s', retryable=True) from None
        except (URLError, ConnectionError) as error:
            reason = getattr(error, 'reason', error)
            # Report the category without exposing URLs, credentials, or payloads.
            raise MetaAPIError(f'{model}: Meta request connection failed ({type(reason).__name__})',
                               retryable=True) from None
        if len(raw) > 8 * 1024 * 1024:
            raise RuntimeError('Meta response exceeds size limit')
        response = json.loads(raw)
        incomplete = response.get('incomplete_details')
        if (response.get('status') == 'incomplete' and isinstance(incomplete, dict)
                and incomplete.get('reason') == 'max_output_tokens'):
            error = MetaAPIError(f'{model}: Meta response exhausted max_output_tokens', retryable=True,
                                 kind='output_limit', allow_policy_continuation=False)
            # Preserve usage, but never parse or execute the truncated response.
            error.audit = [{'response_id': response.get('id'), 'model': response.get('model', model),
                            'usage': response.get('usage', {}), 'status': 'incomplete',
                            'reason': 'max_output_tokens',
                            'max_output_tokens': body.get('max_output_tokens')}]
            raise error
        try:
            text = output_text(response, allow_empty=model == 'sam-3.1')
        except RuntimeError as error:
            raise MetaAPIError(f'{model}: {error}') from error
        audit = {
            'response_id': response.get('id'), 'model': response.get('model', model),
            'usage': response.get('usage', {}), 'output': text,
        }
        return (json.loads(text) if schema is not None else text), audit
