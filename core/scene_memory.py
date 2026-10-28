"""Episode-local visual claims and bounded information-gathering state; no simulator internals."""

from copy import deepcopy
import re


def record_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


ASSESSMENT_SCHEMA = record_schema({
    'phase': {'type': 'string', 'enum': ['access', 'acquire', 'carry', 'locate_destination', 'place', 'recover', 'unknown']},
    'held_object': {'type': 'string'},
    'grasp': {'type': 'string', 'enum': ['supported', 'uncertain', 'empty', 'lost']},
    'destination': {'type': 'string'}, 'information_needed': {'type': 'string'}, 'evidence': {'type': 'string'},
})
MEMORY_ENTRY_SCHEMA = record_schema({
    'name': {'type': 'string'}, 'kind': {'type': 'string', 'enum': ['object', 'fixture']},
    'location': {'type': 'string'},
    'status': {'type': 'string', 'enum': ['visible', 'occluded', 'held', 'placed', 'lost', 'uncertain']},
    'evidence': {'type': 'string'},
})
PERCEPTION_SCHEMA = record_schema({
    'target': {'type': 'string'}, 'camera': {'type': 'string'},
    'mode': {'type': 'string', 'enum': ['detect', 'track']},
})


def validate_scene_fields(decision):
    assessment = decision['assessment']
    if not isinstance(assessment, dict) or set(assessment) != set(ASSESSMENT_SCHEMA['properties']):
        raise ValueError('Invalid transfer assessment')
    for key, specification in ASSESSMENT_SCHEMA['properties'].items():
        value = assessment[key]
        if not isinstance(value, str) or len(value) > 1000 or ('enum' in specification and value not in specification['enum']):
            raise ValueError(f'Invalid assessment {key}')
    if not assessment['evidence'].strip() or (assessment['grasp'] == 'supported' and not assessment['held_object'].strip()):
        raise ValueError('Grasp assessment needs evidence and supported grasps need object identity')
    updates = decision['memory_updates']
    if not isinstance(updates, list) or len(updates) > 12:
        raise ValueError('At most 12 memory updates per decision')
    for entry in updates:
        if not isinstance(entry, dict) or set(entry) != set(MEMORY_ENTRY_SCHEMA['properties']):
            raise ValueError('Invalid memory update')
        for key, specification in MEMORY_ENTRY_SCHEMA['properties'].items():
            value = entry[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 500 or (
                    'enum' in specification and value not in specification['enum']):
                raise ValueError(f'Invalid memory {key}')
    perception = decision['perception']
    if (not isinstance(perception, dict) or set(perception) != {'target', 'camera', 'mode'}
            or perception['mode'] not in ('detect', 'track')
            or any(not isinstance(perception[key], str) or len(perception[key]) > 160 for key in ('target', 'camera'))):
        raise ValueError('Invalid next perception query')


class SceneMemory:
    def __init__(self):
        self.entries = {}
        self.assessment = None
        self.assessments = []
        self.perception = None
        self.navigation_bursts = 0
        self.last_navigation = None
        self.stall_streak = 0
        self.last_instruction = None

    def update(self, decision, step, request):
        validate_scene_fields(decision)
        claim = {'step': step, 'request': request, **deepcopy(decision['assessment'])}
        self.assessment = claim
        self.assessments = (self.assessments + [claim])[-12:]
        for entry in decision['memory_updates']:
            key = (entry['kind'], entry['name'].casefold().strip())
            old = self.entries.get(key, {})
            last_seen = step if entry['status'] == 'visible' else old.get('last_seen_step')
            self.entries[key] = {**deepcopy(entry), 'observation_step': step, 'request': request,
                                 'last_seen_step': last_seen}
        # Old records remain time-stamped claims, not current ground truth.
        if len(self.entries) > 48:
            oldest = sorted(self.entries, key=lambda key: self.entries[key]['observation_step'])
            for key in oldest[:len(self.entries) - 48]:
                del self.entries[key]
        self.perception = deepcopy(decision['perception'])

    def context(self):
        return {'entries': list(self.entries.values()), 'assessment': self.assessment,
                'recent_assessments': self.assessments, 'next_perception': self.perception,
                'navigation_bursts': self.navigation_bursts, 'last_navigation': self.last_navigation,
                'stationary_bursts': self.stall_streak,
                'interpretation': 'Time-stamped visual/model claims, not simulator truth. Reobserve moved, held, '
                                  'occluded, or stale objects. A historical box is not a current 3D location.'}

    def authorize_navigation(self, fixture, limit):
        if not re.search(r'\b(sink|fridge|refrigerator|oven|microwave|dishwasher|coffee machine|cabinet|'
                         r'stove|hood|toaster|electric kettle|stand mixer)\b', fixture.casefold()):
            raise ValueError('Navigation target must name a catalog-supported fixture, not a position or object')
        record = self.entries.get(('fixture', fixture.casefold().strip()))
        if record is None or record['status'] not in ('visible', 'occluded'):
            raise ValueError('Navigation requires a fixture recorded with visual evidence')
        if not self.assessment or not self.assessment['information_needed'].strip():
            raise ValueError('Navigation requires an explicit information need')
        if self.navigation_bursts >= limit:
            raise ValueError('Navigation burst budget exhausted')
        if self.last_navigation == fixture.casefold().strip() and self.stall_streak >= 2:
            raise ValueError('Repeated navigation to the same fixture is stalled')

    def note_motion(self, instruction, diagnostics):
        stationary = diagnostics.get('stationary', False)
        self.stall_streak = self.stall_streak + 1 if stationary else 0
        self.last_instruction = instruction
