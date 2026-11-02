"""Serve pinned Awomo-0.5I on Thor through the common NumPy policy protocol."""

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys

from core.awomo import AwomoAdapter, SOURCE_REF, CHECKPOINT_REF
from inference.download_awomo import sha256, verify_checkpoint
from inference.serve import make_handler


def load_policy(source, assets):
    source, assets = Path(source).resolve(), Path(assets).resolve()
    if Path('/opt/build-info/awomo.commit').read_text().strip() != SOURCE_REF:
        raise ValueError('Awomo source revision mismatch')
    manifest = json.loads((assets / 'manifest.json').read_text())
    if manifest.get('source_ref') != SOURCE_REF or manifest.get('checkpoint_revision') != CHECKPOINT_REF:
        raise ValueError('Awomo asset provenance mismatch; run inference.download_awomo')
    verify_checkpoint(assets / 'checkpoint')
    for name, expected in manifest['files'].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Awomo asset checksum mismatch: {name}')
    base_paths = {}
    for key, name in manifest['base_paths'].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets) or not path.is_dir():
            raise ValueError(f'Invalid Awomo base asset path: {key}')
        base_paths[key] = str(path)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Awomo requires CUDA BF16 on Thor; CPU fallback is disabled')
    probe = torch.ones((16, 16), dtype=torch.bfloat16, device='cuda')
    if (probe @ probe).float().mean().item() != 16:
        raise RuntimeError('Awomo CUDA BF16 kernel check failed')
    sys.path.insert(0, str(source))
    from eval_robocasa365.awomo_05i.policy import load_policy as official_load_policy
    policy = official_load_policy(str(assets / 'checkpoint'), device='cuda',
                                 qwen_path=base_paths['qwen3_vl'], vae_path=base_paths['flux2_vae'])
    model = policy.model
    if (model.action_horizon != 16 or model.action_dim != 80
            or model.history_slots != 9 or model.history_period != 40 or not policy.compress_inputs):
        raise ValueError('Unexpected released Awomo model interface')
    for component in (model.dit, model.vae, model.encoder):
        if any(parameter.device.type != 'cuda' for parameter in component.parameters()):
            raise RuntimeError('Awomo contains CPU model parameters; refusing CPU fallback')
    provenance = {**manifest, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'input_compression': 'official H264 CRF23',
                  'scheduler': model.config['scheduler'], 'history': model.config['history'],
                  'instruction_adaptation': 'Update retained episode instruction before each official predict; retain visual history.'}
    return AwomoAdapter(policy, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/opt/Awomo'))
    parser.add_argument('--assets', type=Path, default=Path('/awomo'))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    engine = load_policy(args.source, args.assets)
    print(json.dumps({'event': 'awomo_ready', **engine.metadata}), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
