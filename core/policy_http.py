"""Small NumPy-only wire format; no torch/pickle dependency in the simulator."""

from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

PROTOCOL_VERSION = 1
MAX_BYTES = 32 * 1024 * 1024


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
            raise RuntimeError(f'Policy HTTP {error.code}: {error.read(4096).decode()}') from error

    def metadata(self):
        metadata = json.loads(self._request('/metadata'))
        if metadata.get('protocol_version') != PROTOCOL_VERSION:
            raise ValueError('Incompatible policy protocol version')
        return metadata

    def predict(self, observations):
        return decode_arrays(self._request('/predict', encode_arrays(observations)))
