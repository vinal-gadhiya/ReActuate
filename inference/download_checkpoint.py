"""Download inference artifacts without optimizer or trainer state."""

import argparse
import json
from pathlib import Path

REPO = 'robocasa/robocasa365_checkpoints'
SUBDIR = 'gr00t_n1-5/multitask_learning/checkpoint-120000'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/checkpoints'))
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(REPO, revision=args.revision).sha
    snapshot_download(
        REPO, revision=revision, local_dir=str(args.output),
        allow_patterns=[
            f'{SUBDIR}/config.json', f'{SUBDIR}/*.safetensors',
            f'{SUBDIR}/model.safetensors.index.json', f'{SUBDIR}/experiment_cfg/*',
        ],
    )
    checkpoint = args.output / SUBDIR
    required = ['config.json', 'model.safetensors.index.json', 'experiment_cfg/metadata.json']
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(checkpoint / shard)
    manifest = {'repo': REPO, 'revision': revision, 'subdir': SUBDIR}
    (checkpoint / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Checkpoint ready: {checkpoint}\nRevision: {revision}')


if __name__ == '__main__':
    main()
