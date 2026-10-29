"""Evaluation-only scene fingerprints and public RGB diagnostics."""

import hashlib
import json
import random
import os
import sys
from copy import deepcopy

import numpy as np
from core.observations import array_digest, rgb_diagnostics


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)


def ensure_hash_seed(seed):
    """Python hash randomization must be fixed before interpreter startup."""
    value = str(seed % (2**32))
    if os.environ.get('PYTHONHASHSEED') != value:
        os.execve(sys.executable, [sys.executable, '-m', 'sim.run_agent', *sys.argv[1:]],
                  {**os.environ, 'PYTHONHASHSEED': value})


def canonical_metadata(metadata):
    value = deepcopy(metadata)
    # Pinned kitchen_object_utils constructs these memberships by iterating
    # OBJ_GROUPS, whose insertion order includes iteration over all_types (set).
    # Membership order has no physical meaning; all other list ordering matters.
    for cfg in value.get('object_cfgs', []):
        if isinstance(cfg, dict) and isinstance(cfg.get('info'), dict):
            groups = cfg['info'].get('groups_containing_sampled_obj')
            if isinstance(groups, list) and all(isinstance(group, str) for group in groups):
                cfg['info']['groups_containing_sampled_obj'] = sorted(groups)
    return value


def metadata_digest(metadata):
    return hashlib.sha256(json.dumps(canonical_metadata(metadata), sort_keys=True,
                                     default=numpy_json).encode()).hexdigest()


def choose_layout_style(pairs, excluded_layouts, excluded_styles, seed):
    candidates = sorted((int(layout), int(style)) for layout, style in pairs
                        if layout not in excluded_layouts and style not in excluded_styles)
    if not candidates:
        raise ValueError('No valid scene pairs remain for this task and split')
    return candidates[int(np.random.default_rng(seed).integers(len(candidates)))]


def reset_scene(env, seed, reference=None):
    """Reset normally, then initialize all conditions with the same spawn pose.

    RoboCasa samples the mobile base using collision rejection. Record the
    accepted spawn explicitly rather than relying only on a seed. Replay it through
    RoboCasa's reset API, including its normal controller reset and settling.
    No simulator state is overwritten and the final fingerprint still must match.
    """
    env.reset(seed=seed)
    kitchen = env.unwrapped.env
    metadata = kitchen.get_ep_meta()
    source = metadata if reference is None else reference.get('initial_metadata')
    if source is None:
        raise ValueError('Reference scene lacks initial_metadata; regenerate the baseline with the current runner')
    spawn = {key: deepcopy(source[key]) for key in ('init_robot_base_pos', 'init_robot_base_ori')}
    # Supply only spawn fields: replaying object_cfgs would change model-building
    # RNG consumption, and copying language/fixture metadata could hide mismatch.
    kitchen.set_ep_meta(spawn)
    seed_process(seed)
    return env.reset(seed=seed)


def scene_fingerprint(env, observation, camera_keys):
    kitchen = env.unwrapped.env
    metadata = kitchen.get_ep_meta()
    xml = kitchen.sim.model.get_xml()
    state = np.asarray(kitchen.sim.get_state().flatten(), dtype=np.float64)
    identity = {
        'layout_id': metadata['layout_id'], 'style_id': metadata['style_id'],
        'goal': str(observation['annotation.human.task_description']),
        'metadata_sha256': metadata_digest(metadata),
        'model_xml_sha256': hashlib.sha256(xml.encode()).hexdigest(),
        'initial_state_sha256': array_digest(np.round(state, decimals=8)),
    }
    return {'identity': identity, 'fingerprint_version': 2,
            'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
            'initial_metadata': json.loads(json.dumps(metadata, default=numpy_json)),
            'initial_state': state.tolist(),
            'initial_images': rgb_diagnostics({key: observation[key] for key in camera_keys}),
            'note': 'Privileged evaluation fingerprint only; never provided to Muse or GR00T.'}


def verify_scene(actual, expected):
    differences = [key for key, value in expected['identity'].items() if actual['identity'].get(key) != value]
    normalized = False
    if ('metadata_sha256' in differences and 'initial_metadata' in actual and 'initial_metadata' in expected
            and metadata_digest(actual['initial_metadata']) == metadata_digest(expected['initial_metadata'])):
        # References from v1 hashed raw membership order. Compare their actual
        # metadata under the same narrow normalization, never just waive the hash.
        differences.remove('metadata_sha256')
        normalized = True
    if differences:
        details = []
        before, after = expected.get('initial_metadata', {}), actual.get('initial_metadata', {})
        for key in sorted(before.keys() | after.keys()):
            if before.get(key) != after.get(key):
                details.append(f'metadata.{key}')
        if 'initial_state' in expected and 'initial_state' in actual:
            before_state, after_state = np.asarray(expected['initial_state']), np.asarray(actual['initial_state'])
            if before_state.shape == after_state.shape:
                delta = np.abs(before_state - after_state)
                details.append(f'state_max_abs_delta={float(delta.max(initial=0)):.9g}')
                details.append(f'state_changed_indices={np.flatnonzero(delta > 1e-8)[:12].tolist()}')
            else:
                details.append(f'state_shape={before_state.shape}->{after_state.shape}')
        raise ValueError(f'Initial scene mismatch ({differences}; {details}); '
                         'comparison aborted before policy execution. See both episode scene.json files for values.')
    return {'matched': True, 'metadata_membership_order_normalized': normalized,
            'reference_fingerprint_version': expected.get('fingerprint_version', 1)}


def numpy_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Nonserializable scene metadata: {type(value).__name__}')
