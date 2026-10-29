"""Serve the pinned PatchWAM APPolicy via the existing NumPy HTTP protocol."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer

from core.patchwam import CHECKPOINT_SHA256, SOURCE_REF, PatchWAMAdapter
from inference.serve import make_handler


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_policy_class(source):
    source = Path(source).resolve()
    for path in (source, source / 'src'):
        sys.path.insert(0, str(path))
    path = source / 'experiments/robocasa/eval/ap_policy_server.py'
    spec = importlib.util.spec_from_file_location('patchwam_upstream_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.APPolicy


def load_policy(source, assets, denoising_steps, seed):
    source, assets = Path(source).resolve(), Path(assets).resolve()
    if Path('/opt/build-info/patchwam.commit').read_text().strip() != SOURCE_REF:
        raise ValueError('PatchWAM source revision mismatch')
    manifest = json.loads((assets / 'manifest.json').read_text())
    if manifest.get('source_ref') != SOURCE_REF or manifest['checkpoint']['sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Unexpected PatchWAM checkpoint manifest')
    checkpoint = assets / 'checkpoint' / 'step_559660.pt'
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('PatchWAM checkpoint checksum mismatch')
    print('Verifying PatchWAM base asset checksums...', flush=True)
    for name, expected in manifest['files'].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'PatchWAM asset checksum mismatch: {name}')
    os.environ.update({
        'REPO_ROOT': str(source),
        'AAP_FLUX2_KLEIN': str(assets / 'flux' / 'flux-2-klein-base-4b.safetensors'),
        'AAP_FLUX2_AE': str(assets / 'vae' / 'ae.safetensors'),
        'AAP_QWEN3': str(assets / 'qwen'),
    })
    for path in (os.environ['AAP_FLUX2_KLEIN'], os.environ['AAP_FLUX2_AE'],
                 assets / 'qwen' / 'config.json'):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing PatchWAM asset: {path}; run inference.download_patchwam first')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Released PatchWAM requires a CUDA GPU with BF16 support; no CPU fallback')
    torch.manual_seed(seed)
    probe = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
    if (probe @ probe).float().mean().item() != 16:
        raise RuntimeError('CUDA BF16 kernel check failed')
    # Keep the published APPolicy preprocessing and prediction implementation.
    # Use a seeded global RNG, rather than reseeding every policy prediction.
    policy = upstream_policy_class(source)(source / 'run_config', checkpoint,
                                          source / 'run_config/dataset_stats.json', 'cuda', denoising_steps, None)
    manifest = {**manifest, 'run_config_sha256': sha256_file(source / 'run_config/config.yaml'),
                'dataset_stats_sha256': sha256_file(source / 'run_config/dataset_stats.json'),
                'denoising_steps': denoising_steps, 'seed': seed, 'rng_scope': 'server process',
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0)}
    return PatchWAMAdapter(policy, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/opt/PatchWAM'))
    parser.add_argument('--assets', type=Path, default=Path('/patchwam'))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--denoising-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.denoising_steps <= 0:
        parser.error('--denoising-steps must be positive')
    engine = load_policy(args.source, args.assets, args.denoising_steps, args.seed)
    print(json.dumps({'event': 'patchwam_ready', **engine.metadata}), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
