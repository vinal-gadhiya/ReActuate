"""Bounded action arbitration driven by public visual assessments, not object state."""

from copy import deepcopy

import numpy as np


ACTION_SHAPES = {
    'action.end_effector_position': (3,), 'action.end_effector_rotation': (3,),
    'action.gripper_close': (1,), 'action.base_motion': (4,), 'action.control_mode': (1,),
}


class TransportControl:
    def __init__(self):
        self.held_object = ''
        self.phase = 'unknown'
        self.release_ready = False
        self.loss_detected = False
        self.overrides = {}

    def update(self, assessment, entries):
        self.phase = assessment['phase']
        grasp = assessment['grasp']
        if grasp == 'lost' and self.held_object:
            self.loss_detected = True
        if grasp == 'supported' and self.phase in ('acquire', 'carry', 'locate_destination', 'place'):
            # Preserve the latch across wording changes; only an explicit release
            # can clear it. This label is not an object-ID tracking guarantee.
            if not self.held_object:
                self.held_object = assessment['held_object'].strip()
        # Neither an occlusion nor an empty-gripper claim silently releases the latch.
        destination = assessment['destination'].strip().casefold()
        self.release_ready = bool(self.held_object and grasp == 'supported' and self.phase == 'place'
                                  and destination and any(
            entry['name'].strip().casefold() == destination and entry['status'] == 'visible'
            for entry in entries))

    def authorize(self, name, arguments):
        if self.loss_detected:
            raise ValueError('Object loss reported; stop this prevention-only trial, do not attempt floor recovery')
        if name == 'set_gripper' and not arguments['closed']:
            if not self.held_object and self.phase == 'access':
                return  # Handle release is distinct from cargo acquisition.
            if not self.release_ready:
                raise ValueError('Release blocked: direct opening during acquisition/uncertainty is disabled; '
                                 'let GR00T manage acquisition. Cargo release requires a supported grasp, '
                                 'place phase, and a visible named destination')
        if name == 'move_vertical' and not (self.held_object and self.phase == 'place' and self.release_ready):
            raise ValueError('Vertical placement requires a latched object and current placement evidence')

    def authorize_program(self, commands):
        if self.held_object and len(commands) > 1 and any(
                name == 'set_gripper' and not arguments['closed'] for name, arguments in commands):
            raise ValueError('Release of a latched object must be the only command after a fresh placement observation')
        for name, arguments in commands:
            self.authorize(name, arguments)

    def filter(self, action, authorized_release=False):
        result = {key: np.asarray(value).copy() for key, value in action.items()}
        if self.held_object and not authorized_release:
            if not np.all(result['action.gripper_close'] == 1):
                self.overrides['gripper_open_blocked'] = self.overrides.get('gripper_open_blocked', 0) + 1
            result['action.gripper_close'][:] = 1
        if self.phase == 'place':
            if np.any(result['action.base_motion']) or np.any(result['action.control_mode']):
                self.overrides['placement_base_stopped'] = self.overrides.get('placement_base_stopped', 0) + 1
            result['action.base_motion'][:] = 0
            result['action.control_mode'][:] = 0
            if np.any(result['action.end_effector_rotation']):
                self.overrides['placement_rotation_stopped'] = self.overrides.get('placement_rotation_stopped', 0) + 1
            result['action.end_effector_rotation'][:] = 0
        return result

    def released(self):
        self.held_object = ''
        self.release_ready = False

    def context(self):
        return deepcopy({'held_object_latch': self.held_object, 'phase': self.phase,
                         'release_ready': self.release_ready, 'loss_detected': self.loss_detected,
                         'action_overrides': self.overrides,
                         'note': 'Visual claims gate commands; this is not contact sensing or a guarantee against slips.'})
