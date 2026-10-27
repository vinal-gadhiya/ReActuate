"""Serve the RoboCasa GR00T policy without simulator dependencies."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import os
from pathlib import Path
import threading
import traceback

import numpy as np

from core.policy_http import MAX_BYTES, PROTOCOL_VERSION, decode_arrays, encode_arrays


class Gr00tEngine:
    def __init__(self, checkpoint, denoising_steps, seed):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; start with the NVIDIA runtime and --gpus all')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('GR00T requires BF16 support on this GPU')
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Test a real kernel; device enumeration alone is insufficient on Thor.
        x = torch.ones((16, 16), dtype=torch.bfloat16, device='cuda')
        assert (x @ x).float().mean().item() == 16
        print(f'GPU: {torch.cuda.get_device_name(0)}; torch: {torch.__version__}; CUDA: {torch.version.cuda}')

        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        checkpoint = checkpoint.resolve()
        config = json.loads((checkpoint / 'config.json').read_text())
        data_config = DATA_CONFIG_MAP['panda_omron']
        modalities = data_config.modality_config()
        horizon = len(modalities['action'].delta_indices)
        if config['action_horizon'] != horizon:
            raise ValueError('Checkpoint action horizon differs from panda_omron; refusing to recreate action head')
        for name in ('video', 'state', 'language'):
            if modalities[name].delta_indices != [0]:
                raise ValueError('This runner supports one current observation only')
        self.policy = Gr00tPolicy(
            model_path=str(checkpoint), embodiment_tag='new_embodiment',
            modality_config=modalities, modality_transform=data_config.transform(),
            denoising_steps=denoising_steps, device='cuda',
        )
        eagle = self.policy.model.backbone.eagle_model
        for component in (eagle.vision_model, eagle.language_model):
            if component.config._attn_implementation != 'eager':
                raise ValueError('Unexpected attention backend; check the eager attention patch')
        resolutions = {
            tuple(self.policy.metadata.modalities.video[key.split('.', 1)[1]].resolution)
            for key in modalities['video'].modality_keys
        }
        if len(resolutions) != 1:
            raise ValueError(f'Mismatched camera resolutions: {resolutions}')
        width, height = next(iter(resolutions))
        manifest_path = checkpoint / 'download_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        self.metadata = {
            'protocol_version': PROTOCOL_VERSION,
            'policy': 'gr00t_n1.5', 'data_config': 'panda_omron',
            'embodiment': 'PandaOmron', 'embodiment_tag': 'new_embodiment',
            'source_ref': os.environ.get('GROOT_SOURCE_REF', 'unknown'),
            'checkpoint': str(checkpoint), 'download_manifest': manifest,
            'denoising_steps': denoising_steps, 'seed': seed,
            'attention': 'eager', 'camera_width': width, 'camera_height': height,
            'action_horizon': horizon,
            'modalities': {key: value.model_dump() for key, value in modalities.items()},
            'versions': {key: version(key) for key in ('torch', 'torchvision', 'transformers', 'numpy')},
        }

    def predict(self, observations):
        return self.policy.get_action(observations)


def make_handler(engine):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/health':
                result = {'status': 'ready', 'protocol_version': PROTOCOL_VERSION}
            elif self.path == '/metadata':
                result = engine.metadata
            else:
                self.reply(404, b'{"error":"unknown endpoint"}')
                return
            self.reply(200, json.dumps(result).encode())

        def do_POST(self):
            if self.path != '/predict':
                self.reply(404, b'{"error":"unknown endpoint"}')
                return
            self.connection.settimeout(120)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError('Invalid Content-Length')
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError('Incomplete request')
                observations = decode_arrays(payload)
                request_id = observations.pop('__request_id').item()
                if not isinstance(request_id, str):
                    raise ValueError('Request ID must be a string')
                expected = {
                    key for name, modality in engine.metadata['modalities'].items()
                    if name != 'action' for key in modality['modality_keys']
                }
                if set(observations) != expected:
                    raise ValueError(f'Observation keys differ: {set(observations) ^ expected}')
                with lock:
                    actions = engine.predict(observations)
                actions = {key: np.asarray(value) for key, value in actions.items()}
                actions['__request_id'] = np.asarray(request_id)
                self.reply(200, encode_arrays(actions), 'application/octet-stream')
            except (ValueError, KeyError) as error:
                self.reply(400, json.dumps({'error': str(error)}).encode())
            except Exception as error:
                traceback.print_exc()
                self.reply(500, json.dumps({'error': str(error)}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--denoising-steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.denoising_steps <= 0:
        parser.error('--denoising-steps must be positive')
    engine = Gr00tEngine(args.checkpoint, args.denoising_steps, args.seed)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f'Policy ready at http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
