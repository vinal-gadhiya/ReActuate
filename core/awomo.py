"""NumPy wire bridge to the pinned official Awomo-0.5I RoboCasa policy.

The upstream policy owns state80, normalization, H264/canvas preprocessing,
sampling, clipping/discretization and its convert_action-order action rows.
"""

import re
import numpy as np

from core.patchwam import CAMERAS, STATE_LAYOUT, ACTION_LAYOUT, LANGUAGE
from core.policy_http import PROTOCOL_VERSION

SOURCE_REF = 'b11d82f7ee198955d0e9d26b3ebeed1dafea03b4'
CHECKPOINT_REPO = 'Auwomo/Awomo-0.5I-RoboCasa365'
CHECKPOINT_REF = 'd02e3560a9e5e649bb0718a97b9e4b923df59594'
CHECKSUMS = {'model.pt': 'e745faf5f75ab0428f00755739374e67d7600724445e1c25973474ef0ca951be',
             'awomo_config.json': 'bed652745ff23c1f43c446337f0da78615bbf3ef8e24a5268e3f36ee62784b3e'}
CONTEXT_KEYS = ('__episode_id', '__control_step', '__episode_seed', '__frame_steps', '__camera_frames')


class AwomoHistory:
    """Real 5 Hz sensor samples between 16-step predictions; never fabricated frames."""

    def __init__(self, episode_id, seed):
        self.episode_id, self.seed = episode_id, seed
        self.pending = []
        self.last_prediction = None

    def observe(self, step, observation):
        if step % 4 == 0:
            if self.pending and step <= self.pending[-1][0]:
                raise ValueError('Awomo camera samples must advance in time')
            self.pending.append((step, np.stack([observation[f'video.{key}'].copy() for key in CAMERAS])))

    def payload(self, step):
        expected = [0] if self.last_prediction is None else list(range(self.last_prediction + 4, self.last_prediction + 17, 4))
        if [at for at, _ in self.pending] != expected or step != expected[-1]:
            raise ValueError('Awomo requires real frames every 4 steps and predictions every 16 steps')
        return {'__episode_id': np.asarray(self.episode_id), '__episode_seed': np.asarray(self.seed),
                '__control_step': np.asarray(step), '__frame_steps': np.asarray(expected, dtype=np.int64),
                '__camera_frames': np.stack([frame for _, frame in self.pending])}

    def predicted(self, step):
        self.last_prediction = step
        self.pending.clear()

    def refresh(self, step, observation):
        if self.pending and self.pending[-1][0] == step:
            self.pending[-1] = (step, np.stack([observation[f'video.{key}'].copy() for key in CAMERAS]))


class AwomoAdapter:
    def __init__(self, policy, provenance=None):
        self.policy = policy
        self.session = None
        self.last_step = None
        self.replan = -1
        self.failed_session = None
        self.metadata = {
            'protocol_version': PROTOCOL_VERSION, 'policy': 'awomo-0.5i',
            'data_config': 'panda_omron', 'embodiment': 'PandaOmron',
            'camera_width': 256, 'camera_height': 256, 'action_horizon': 16,
            'recommended_action_steps': 16, 'required_action_steps': 16,
            'source_ref': SOURCE_REF, 'checkpoint_repo': CHECKPOINT_REPO, 'checkpoint_revision': CHECKPOINT_REF,
            'observation_history': {'stride': 4, 'prediction_interval': 16, 'scope': 'episode'},
            'request_context_keys': list(CONTEXT_KEYS),
            'modalities': {
                'video': {'modality_keys': [f'video.{key}' for key in CAMERAS], 'delta_indices': [0]},
                'state': {'modality_keys': [f'state.{key}' for key, _ in STATE_LAYOUT], 'delta_indices': [0]},
                'language': {'modality_keys': [LANGUAGE], 'delta_indices': [0]},
                'action': {'modality_keys': [f'action.{key}' for key, _ in ACTION_LAYOUT],
                           'delta_indices': list(range(16))}},
            'instruction_updates': 'Latest verbatim instruction per prediction; retain visual history across subtask changes.',
            'action_mapping': 'Official convert_action-order output rearranged into named fields; no additional control overrides.',
            'provenance': provenance or {},
        }

    def predict(self, observations):
        session = np.asarray(observations['__episode_id'])
        if session.ndim != 0 or session.dtype.kind not in ('U', 'S') or not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', str(session.item())):
            raise ValueError('Invalid Awomo episode ID')
        session = str(session.item())
        scalar = {}
        for key in ('__control_step', '__episode_seed'):
            value = np.asarray(observations[key])
            if value.ndim != 0 or value.dtype.kind not in ('i', 'u') or not 0 <= int(value) < 2**31:
                raise ValueError(f'Invalid Awomo integer: {key}')
            scalar[key] = int(value)
        step = scalar['__control_step']
        new_episode = session != self.session
        if (new_episode and step != 0) or (not new_episode and (session == self.failed_session or step != self.last_step + 16)):
            raise ValueError('Unexpected Awomo session/timeline; start a new episode after failure')
        frame_steps = np.asarray(observations['__frame_steps'])
        expected = [0] if new_episode else list(range(self.last_step + 4, self.last_step + 17, 4))
        if frame_steps.dtype.kind not in ('i', 'u') or frame_steps.tolist() != expected:
            raise ValueError('Missing, stale or out-of-order Awomo history samples')
        frames = np.asarray(observations['__camera_frames'])
        if frames.dtype != np.uint8 or frames.shape != (len(expected), 3, 256, 256, 3):
            raise ValueError('Awomo camera_frames must be uint8 [T,3,256,256,3]')
        for index, camera in enumerate(CAMERAS):
            current = np.asarray(observations[f'video.{camera}'])
            if current.shape != (1, 256, 256, 3) or current.dtype != np.uint8 or not np.array_equal(current[0], frames[-1, index]):
                raise ValueError('Awomo current and historical camera endpoints disagree')
        parts = []
        for key, width in STATE_LAYOUT:
            value = np.asarray(observations[f'state.{key}'])
            if value.shape != (1, width) or not np.isfinite(value).all():
                raise ValueError(f'Invalid Awomo state: {key}')
            parts.append(value[0].astype(np.float32))
        language = np.asarray(observations[LANGUAGE])
        if language.shape != (1,) or language.dtype.kind not in ('U', 'S') or not str(language[0]).strip():
            raise ValueError('Awomo requires one nonempty instruction')
        instruction = str(language[0])
        if new_episode:
            self.policy.episode = None
            prompt = getattr(self.policy, 'prompt', None)
            if prompt is not None:
                prompt._cache.clear()
            self.session, self.replan, self.failed_session = session, -1, None
        elif self.policy.episode is not None:
            # Upstream otherwise latches the first instruction for the episode.
            self.policy.episode['instruction'] = instruction
        request = {'session_id': session, 'replan_index': self.replan + 1,
                   'seed': (scalar['__episode_seed'] * 10000 + self.replan + 1) % (2**32),
                   'instruction': instruction, 'observation_steps': expected,
                   'camera_frames': frames, 'state': np.concatenate(parts)}
        try:
            actions = np.asarray(self.policy.predict(request)['actions'], dtype=np.float32)
            if actions.shape != (16, 12) or not np.isfinite(actions).all():
                raise ValueError('Awomo returned an invalid action chunk')
        except Exception:
            self.failed_session = session
            raise
        self.last_step, self.replan = step, self.replan + 1
        # Upstream order: eef xyz, eef rotation, gripper, base xyz, torso, mode.
        canonical = actions[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
        result, offset = {}, 0
        for name, width in ACTION_LAYOUT:
            result[f'action.{name}'] = canonical[:, offset:offset + width].copy()
            offset += width
        return result
