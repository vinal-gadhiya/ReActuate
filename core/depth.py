"""Aligned visible-surface depth for Muse; never a policy input or contact signal."""

import numpy as np


def depth_to_metres(normalized, near, far):
    """MuJoCo depth-buffer conversion, as in robosuite.utils.camera_utils."""
    if not np.isfinite(near) or not np.isfinite(far) or not 0 < near < far:
        raise ValueError('Invalid depth camera clipping planes')
    depth = np.asarray(normalized, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or not all(depth.shape):
        raise ValueError('Expected a nonempty HxW or HxWx1 depth buffer')
    valid = np.isfinite(depth) & (depth >= 0) & (depth <= 1)
    if not valid.any():
        raise ValueError('Depth buffer has no valid pixels')
    safe = np.where(valid, depth, 0)
    metres = near / (1.0 - safe * (1.0 - near / far))
    return np.where(valid, metres, np.nan).astype(np.float32)


def visualize_depth(metres, max_metres):
    """Fixed scale across cameras and time; near=white, far=black, invalid=magenta."""
    if not np.isfinite(max_metres) or max_metres <= 0:
        raise ValueError('Depth display range must be finite and positive')
    valid = np.isfinite(metres) & (metres > 0)
    scaled = np.clip(np.where(valid, metres, max_metres) / max_metres, 0, 1)
    grey = np.rint((1 - scaled) * 255).astype(np.uint8)
    image = np.repeat(grey[..., None], 3, axis=-1)
    image[~valid] = (255, 0, 255)
    return image


def supervisor_depth_views(observation, cameras, near, far, max_metres=3.0):
    images, raw = {}, {}
    for camera in cameras:
        key = f'{camera}_depth'
        if key not in observation:
            raise ValueError(f'Requested depth observation is missing: {key}')
        metres = depth_to_metres(observation[key], near, far)
        if metres.shape != observation[camera].shape[:2]:
            raise ValueError(f'Depth is not aligned with RGB dimensions: {camera}')
        images[f'current_depth:{camera}'] = visualize_depth(metres, max_metres)
        raw[key] = metres
    description = {
        'enabled': True, 'unit': 'metres', 'display_range_metres': [0, max_metres],
        'near_clip_metres': near, 'far_clip_metres': far,
        'legend': f'White = 0 m; grey = intermediate depth; black = {max_metres:g} m or farther; '
                  'magenta = invalid depth. The same fixed scale is used for every view and request.',
        'alignment': 'current_depth:<camera> is pixel-aligned with current:<camera>; '
                     'before_depth:<camera> is aligned with before:<camera>. Images are already upright.',
        'meaning': 'Distance along the camera optical axis to the first visible surface. '
                   'This is not object identity, contact, containment, grasp, or a gripper-to-object distance. '
                   'Use RGB for identity and temporal RGB/depth for separation and retention.',
    }
    return images, raw, description
