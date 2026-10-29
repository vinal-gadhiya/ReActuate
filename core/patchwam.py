"""NumPy-only bridge between the common RoboCasa wire format and PatchWAM.

Layout verified against the official CloseCabinet LeRobot meta/modality.json in
nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos at
fadf4e714efd130813bc4b93188df2e19a2de86c. The upstream APPolicy owns image
stitching, state normalization, prompt formatting, inference and denormalization.
"""

import numpy as np

from core.policy_http import PROTOCOL_VERSION

SOURCE_REF = '7783ae2447ef26c10b2b0470753dfad6be85202e'
CHECKPOINT_REPO = 'QZWang/robocasa-human300-step559660'
CHECKPOINT_REF = '9df2fa292952d4d10ef61083c732862c880b1d07'
CHECKPOINT_SHA256 = '7877dd2546d7e2fe489aab79f84476266724b7392d48fe2f2861e9752cd4dccb'
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
STATE_LAYOUT = (('base_position', 3), ('base_rotation', 4),
                ('end_effector_position_relative', 3), ('end_effector_rotation_relative', 4),
                ('gripper_qpos', 2))
ACTION_LAYOUT = (('base_motion', 4), ('control_mode', 1), ('end_effector_position', 3),
                 ('end_effector_rotation', 3), ('gripper_close', 1))
LANGUAGE = 'annotation.human.task_description'


class PatchWAMAdapter:
    def __init__(self, policy, provenance=None):
        if (policy.action_horizon != 16 or policy.state_dim != 16
                or tuple(policy.image_keys) != CAMERAS):
            raise ValueError('PatchWAM policy does not match the released RoboCasa checkpoint interface')
        self.policy = policy
        self.metadata = {
            'protocol_version': PROTOCOL_VERSION, 'policy': 'patchwam',
            'data_config': 'panda_omron', 'embodiment': 'PandaOmron',
            'camera_width': 256, 'camera_height': 256, 'action_horizon': 16,
            'recommended_action_steps': 16, 'source_ref': SOURCE_REF,
            'checkpoint_repo': CHECKPOINT_REPO, 'checkpoint_revision': CHECKPOINT_REF,
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'modalities': {
                'video': {'modality_keys': [f'video.{key}' for key in CAMERAS], 'delta_indices': [0]},
                'state': {'modality_keys': [f'state.{key}' for key, _ in STATE_LAYOUT], 'delta_indices': [0]},
                'language': {'modality_keys': [LANGUAGE], 'delta_indices': [0]},
                'action': {'modality_keys': [f'action.{key}' for key, _ in ACTION_LAYOUT],
                           'delta_indices': list(range(16))},
            },
            'provenance': provenance or {},
            'action_mapping': 'Dataset order split by named fields, without clipping, thresholding or control overrides.',
        }

    def predict(self, observations):
        images = {}
        for key in CAMERAS:
            image = np.asarray(observations[f'video.{key}'])
            if image.shape != (1, 256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f'Invalid current RGB observation: {key}')
            images[key] = np.ascontiguousarray(image[0])
        parts = []
        for key, width in STATE_LAYOUT:
            value = np.asarray(observations[f'state.{key}'])
            if value.shape != (1, width) or not np.isfinite(value).all():
                raise ValueError(f'Invalid robot state: {key}')
            parts.append(value[0].astype(np.float32))
        language = np.asarray(observations[LANGUAGE])
        if language.shape != (1,) or language.dtype.kind not in ('U', 'S') or not str(language[0]).strip():
            raise ValueError('Expected one nonempty instruction')
        actions = np.asarray(self.policy.act(images, np.concatenate(parts), str(language[0])), dtype=np.float32)
        if actions.shape != (16, 12) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid PatchWAM action chunk: {actions.shape}; expected finite (16, 12)')
        result, offset = {}, 0
        for key, width in ACTION_LAYOUT:
            result[f'action.{key}'] = actions[:, offset:offset + width].copy()
            offset += width
        return result
