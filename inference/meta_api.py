"""Meta Responses API client. Credentials are read only from the environment."""

import base64
from io import BytesIO
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


def image_part(frame):
    stream = BytesIO()
    Image.fromarray(frame).save(stream, format='JPEG', quality=85)
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()}


def output_text(response):
    if response.get('status') != 'completed':
        raise RuntimeError(f'Meta response did not complete: {response.get("status")}')
    text = ''.join(
        part.get('text', '')
        for item in response.get('output', []) if item.get('type') == 'message'
        for part in item.get('content', []) if part.get('type') == 'output_text'
    )
    if not text:
        raise RuntimeError('Meta returned no output text (possibly a refusal)')
    return text


class MetaAPI:
    def __init__(self, timeout=120):
        self.key = os.environ.get('MODEL_API_KEY')
        if not self.key:
            raise RuntimeError('MODEL_API_KEY is not set in the supervisor container')
        self.base_url = os.environ.get('META_BASE_URL', 'https://api.meta.ai/v1').rstrip('/')
        self.timeout = timeout

    def response(self, model, content, instructions=None, schema=None):
        body = {
            'model': model, 'input': [{'role': 'user', 'content': content}],
            'store': False, 'max_output_tokens': 4096,
        }
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
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except HTTPError as error:
            # Do not copy request headers/credentials into traces.
            raise RuntimeError(f'Meta HTTP {error.code}; check model access, quota, and API configuration') from error
        if len(raw) > 8 * 1024 * 1024:
            raise RuntimeError('Meta response exceeds size limit')
        response = json.loads(raw)
        text = output_text(response)
        audit = {
            'response_id': response.get('id'), 'model': response.get('model', model),
            'usage': response.get('usage', {}), 'output': text,
        }
        return (json.loads(text) if schema is not None else text), audit
