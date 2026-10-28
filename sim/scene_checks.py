"""Evaluation-only scene fingerprints and public RGB diagnostics."""

import hashlib
import json
import random

import numpy as np
from core.observations import array_digest, rgb_diagnostics


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)


def choose_layout_style(pairs, excluded_layouts, excluded_styles, seed):
    candidates = sorted((int(layout), int(style)) for layout, style in pairs
                        if layout not in excluded_layouts and style not in excluded_styles)
    if not candidates:
        raise ValueError('No valid scene pairs remain for this task and split')
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def scene_fingerprint(env, observation, camera_keys):
    kitchen = env.unwrapped.env
    metadata = kitchen.get_ep_meta()
    xml = kitchen.sim.model.get_xml()
    state = np.asarray(kitchen.sim.get_state().flatten(), dtype=np.float64)
    identity = {
        'layout_id': metadata['layout_id'], 'style_id': metadata['style_id'],
        'goal': str(observation['annotation.human.task_description']),
        'metadata_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True, default=numpy_json).encode()).hexdigest(),
        'model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
        'initial_state_sha256': array_digest(np.round(state, decimals=8)),
    }
    return {'identity': identity, 'initial_images': rgb_diagnostics({key: observation[key] for key in camera_keys}),
            'note': 'Privileged evaluation fingerprint only; never provided to Muse or GR00T.'}


def verify_scene(actual, expected):
    differences = [key for key, value in expected['identity'].items() if actual['identity'].get(key) != value]
    if differences:
        raise ValueError(f'Initial scene mismatch ({differences}); comparison aborted before policy execution')


def numpy_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Nonserializable scene metadata: {type(value).__name__}')
