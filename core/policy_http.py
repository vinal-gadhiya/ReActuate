"""Small NumPy-only wire format; no torch/pickle dependency in the simulator."""

from io import BytesIO
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

PROTOCOL_VERSION = 1
MAX_BYTES = 32 * 1024 * 1024


class PolicyHTTPError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f'Policy HTTP {status}: {body[:4096]}')
        self.status, self.body = status, body


class RetryableSupervisorError(RuntimeError):
    def __init__(self, message, api_calls=1, audit=None, allow_policy_continuation=True):
        super().__init__(message)
        self.api_calls, self.audit = api_calls, audit or []
        self.allow_policy_continuation = allow_policy_continuation


def encode_arrays(values):
    arrays = {key: np.asarray(value) for key, value in values.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Object arrays are not supported')
    output = BytesIO()
    np.savez(output, **arrays)
    payload = output.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError('Policy message exceeds size limit')
    return payload


def decode_arrays(payload):
    if len(payload) > MAX_BYTES:
        raise ValueError('Policy message exceeds size limit')
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if sum(value.nbytes for value in values.values()) > MAX_BYTES:
        raise ValueError('Decoded policy message exceeds size limit')
    return values


class PolicyClient:
    def __init__(self, url, timeout=120):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _request(self, path, payload=None):
        request = Request(
            self.url + path, data=payload,
            headers={'Content-Type': 'application/octet-stream'},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = response.read(MAX_BYTES + 1)
                if len(result) > MAX_BYTES:
                    raise RuntimeError('Policy response exceeds size limit')
                return result
        except HTTPError as error:
            raise PolicyHTTPError(error.code, error.read(MAX_BYTES).decode(errors='replace')) from error

    def metadata(self):
        metadata = json.loads(self._request('/metadata'))
        if metadata.get('protocol_version') != PROTOCOL_VERSION:
            raise ValueError('Incompatible policy protocol version')
        return metadata

    def predict(self, observations):
        return decode_arrays(self._request('/predict', encode_arrays(observations)))


class SupervisorClient(PolicyClient):
    def ask(self, operation, context, images):
        if operation not in ('plan', 'decide'):
            raise ValueError('Unknown supervisor operation')
        payload = dict(images)
        payload['__context'] = np.asarray(json.dumps(context))
        try:
            return json.loads(self._request('/' + operation, encode_arrays(payload)))
        except PolicyHTTPError as error:
            try:
                detail = json.loads(error.body)
            except (ValueError, TypeError):
                detail = {}
            if error.status in (502, 503, 504) and detail.get('retryable') is True:
                calls = detail.get('api_calls', 1)
                if type(calls) is not int or not 1 <= calls <= context.get('api_calls_left', 200):
                    raise ValueError('Invalid supervisor failure call count') from error
                allow_motion = detail.get('allow_policy_continuation', True)
                if type(allow_motion) is not bool:
                    raise ValueError('Invalid supervisor continuation permission') from error
                raise RetryableSupervisorError(str(detail.get('error', 'Supervisor unavailable')),
                                               calls, detail.get('audit', []), allow_motion) from error
            raise
        except (URLError, TimeoutError, ConnectionError) as error:
            # Review, SAM, planning, confirmation and one output-limit retry may
            # have run. Reserve a conservative bound when the response is lost.
            calls = min(context.get('api_calls_left', 1), 5)
            raise RetryableSupervisorError(f'Supervisor connection failed ({type(error).__name__})', calls) from error
