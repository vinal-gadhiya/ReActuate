"""Select inference dependencies and eager attention in the pinned source tree."""

import argparse
import json
from pathlib import Path
import re


def prepare(root, requirements):
    path = root / 'pyproject.toml'
    text = path.read_text()
    dependencies = [
        line.strip() for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    dependencies += ['torch==2.9.1', 'torchvision==0.24.1']
    text, count = re.subn(
        r'^dependencies = \[.*?^\]',
        'dependencies = ' + json.dumps(dependencies, indent=4),
        text, count=1, flags=re.MULTILINE | re.DOTALL,
    )
    if count != 1:
        raise ValueError('Cannot find upstream dependency declaration')
    path.write_text(text)

    # The bundled Eagle configuration selects FlashAttention by default. Eager
    # attention uses the same weights without a separate CUDA extension build.
    config_path = root / 'gr00t/model/backbone/eagle2_hg_model/config.json'
    config = json.loads(config_path.read_text())
    for section in (config, config['text_config'], config['vision_config']):
        section['_attn_implementation'] = 'eager'
        section['_attn_implementation_autoset'] = True
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    # Eagle's vision constructor overrides the JSON setting. Patch that single
    # assignment too, and fail if the pinned source no longer matches.
    model_path = config_path.parent / 'modeling_eagle2_5_vl.py'
    model = model_path.read_text()
    forced = 'config.vision_config._attn_implementation = "flash_attention_2"'
    if model.count(forced) != 1:
        raise ValueError('Upstream vision attention override changed; review patch')
    model_path.write_text(model.replace(forced, 'config.vision_config._attn_implementation = "eager"'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('requirements', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.requirements)
