"""Download pinned Awomo-0.5I RoboCasa weights and the checkpoint-pinned base assets."""

import argparse
import hashlib
import json
from pathlib import Path

from core.awomo import CHECKPOINT_REPO, CHECKPOINT_REF, CHECKSUMS, SOURCE_REF


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(directory):
    for name, expected in CHECKSUMS.items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError(f'Awomo checkpoint checksum mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/awomo'))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    root = args.output.resolve()
    checkpoint = root / 'checkpoint'
    snapshot_download(CHECKPOINT_REPO, revision=CHECKPOINT_REF, local_dir=checkpoint,
                      allow_patterns=['model.pt', 'awomo_config.json', 'SHA256SUMS', 'LICENSE.md', 'README.md'])
    verify_checkpoint(checkpoint)
    config = json.loads((checkpoint / 'awomo_config.json').read_text())
    bases = {}
    for name, spec in config['base_models'].items():
        if not spec.get('revision') or len(spec['revision']) != 40:
            raise ValueError('Awomo base models must have pinned commit revisions')
        directory = root / name
        subfolder = spec.get('subfolder')
        snapshot_download(spec['repo_id'], revision=spec['revision'], local_dir=directory,
                          allow_patterns=[f'{subfolder}/*'] if subfolder else None,
                          ignore_patterns=['*.bin', '*.pt', '*.gguf'])
        bases[name] = str((directory / subfolder if subfolder else directory).relative_to(root))
    files = {str(path.relative_to(root)): sha256(path)
             for path in root.rglob('*') if path.is_file() and '.cache' not in path.parts and path.name != 'manifest.json'}
    manifest = {'source_ref': SOURCE_REF, 'checkpoint_repo': CHECKPOINT_REPO,
                'checkpoint_revision': CHECKPOINT_REF, 'base_models': config['base_models'],
                'base_paths': bases, 'files': files}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'awomo_download_verified', 'directory': str(root),
                      'checkpoint_revision': CHECKPOINT_REF}), flush=True)


if __name__ == '__main__':
    main()
