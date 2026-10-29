"""Download PatchWAM and its three base assets; record immutable revisions/hashes."""

import argparse
import json
from pathlib import Path

from core.patchwam import CHECKPOINT_REF, CHECKPOINT_REPO, CHECKPOINT_SHA256, SOURCE_REF
from inference.serve_patchwam import sha256_file

SPECS = {
    'checkpoint': (CHECKPOINT_REPO, ['step_559660.pt']),
    'flux': ('black-forest-labs/FLUX.2-klein-base-4B', ['flux-2-klein-base-4b.safetensors']),
    'vae': ('black-forest-labs/FLUX.2-dev', ['ae.safetensors']),
    'qwen': ('Qwen/Qwen3-4B', ['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/patchwam'))
    for key in ('flux', 'vae', 'qwen'):
        parser.add_argument(f'--{key}-revision', help='HF commit/ref; resolved SHA is saved and reused on later downloads')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {'source_ref': SOURCE_REF, 'assets': {}, 'files': {}}
    api = HfApi()
    for key, (repo, patterns) in SPECS.items():
        revision = CHECKPOINT_REF if key == 'checkpoint' else (
            getattr(args, f'{key}_revision') or old.get('assets', {}).get(key, {}).get('revision', 'main'))
        revision = api.model_info(repo, revision=revision).sha
        if key == 'checkpoint' and revision != CHECKPOINT_REF:
            raise ValueError('Checkpoint revision mismatch')
        destination = args.output / key
        print(f'Downloading {repo}@{revision} into {destination}', flush=True)
        snapshot_download(repo, revision=revision, local_dir=str(destination), allow_patterns=patterns)
        paths = sorted(path for path in destination.rglob('*') if path.is_file()
                       and '.cache' not in path.relative_to(destination).parts)
        if not paths:
            raise FileNotFoundError(f'No assets downloaded from {repo}')
        for path in paths:
            manifest['files'][str(path.relative_to(args.output))] = sha256_file(path)
        manifest['assets'][key] = {'repo': repo, 'revision': revision}
    digest = manifest['files']['checkpoint/step_559660.pt']
    if digest != CHECKPOINT_SHA256:
        raise ValueError('Released PatchWAM checkpoint SHA256 mismatch')
    required = ['flux/flux-2-klein-base-4b.safetensors', 'vae/ae.safetensors', 'qwen/config.json']
    if any(name not in manifest['files'] for name in required):
        raise FileNotFoundError('Incomplete PatchWAM base assets')
    if not any(name.startswith('qwen/') and name.endswith('.safetensors') for name in manifest['files']):
        raise FileNotFoundError('Missing Qwen text encoder weights')
    manifest['checkpoint'] = {**manifest['assets']['checkpoint'], 'sha256': digest}
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    print(f'PatchWAM assets ready: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
