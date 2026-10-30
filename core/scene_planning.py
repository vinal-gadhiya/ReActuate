"""Public visual survey required before generating a new task plan."""

from copy import deepcopy

from core.agent import PLAN_SCHEMA, validate_plan


def record(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


ENTITY_SCHEMA = record({
    'name': {'type': 'string'},
    'kind': {'type': 'string', 'enum': ['object', 'fixture']},
    'visibility': {'type': 'string', 'enum': ['visible', 'uncertain', 'not_seen']},
    'location': {'type': 'string'},
    'access': {'type': 'string', 'enum': ['apparently_reachable', 'needs_approach', 'blocked', 'unknown']},
    'cameras': {'type': 'array', 'items': {'type': 'string'}},
    'evidence': {'type': 'string'},
})
SURVEY_SCHEMA = record({
    'entities': {'type': 'array', 'items': ENTITY_SCHEMA},
    'ordering_constraints': {'type': 'string'},
    'first_action_rationale': {'type': 'string'},
    'route_rationale': {'type': 'string'},
})
GROUNDED_PLAN_SCHEMA = record({
    'scene_assessment': SURVEY_SCHEMA,
    'steps': deepcopy(PLAN_SCHEMA['properties']['steps']),
})


def validate_scene_assessment(survey, cameras=None):
    if not isinstance(survey, dict) or set(survey) != set(SURVEY_SCHEMA['properties']):
        raise ValueError('Plan requires a scene assessment before selecting subtasks')
    for key in ('ordering_constraints', 'first_action_rationale', 'route_rationale'):
        if not isinstance(survey[key], str) or not 0 < len(survey[key].strip()) <= 1600:
            raise ValueError(f'Invalid scene assessment {key}')
    entities = survey['entities']
    if not isinstance(entities, list) or not 1 <= len(entities) <= 16:
        raise ValueError('Scene assessment requires 1–16 task-relevant entities')
    names = set()
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != set(ENTITY_SCHEMA['properties']):
            raise ValueError('Invalid scene entity fields')
        for key, spec in ENTITY_SCHEMA['properties'].items():
            if key == 'cameras':
                continue
            value = entity[key]
            if (not isinstance(value, str) or not 0 < len(value.strip()) <= 500
                    or ('enum' in spec and value not in spec['enum'])):
                raise ValueError(f'Invalid scene entity {key}')
        name = (entity['kind'], entity['name'].strip().casefold())
        if name in names:
            raise ValueError('Scene entity names must be unique within each kind')
        names.add(name)
        views = entity['cameras']
        if (not isinstance(views, list) or len(views) > 6
                or any(not isinstance(view, str) or not 0 < len(view) <= 160 for view in views)
                or len(set(views)) != len(views)):
            raise ValueError('Invalid scene entity cameras')
        if cameras is not None and any(view not in cameras for view in views):
            raise ValueError('Scene evidence must reference supplied current cameras')
        if entity['visibility'] == 'visible' and not views:
            raise ValueError('Visible scene entities require camera evidence')
        if entity['visibility'] != 'visible' and entity['access'] == 'apparently_reachable':
            raise ValueError('Unconfirmed objects cannot be declared apparently reachable')
    return survey


def validate_grounded_plan(result, cameras):
    if not isinstance(result, dict) or set(result) != {'scene_assessment', 'steps'}:
        raise ValueError('Generated plan requires scene_assessment and steps')
    validate_scene_assessment(result['scene_assessment'], cameras)
    return validate_plan({'steps': result['steps']})


def load_plan(value):
    """Accept legacy operator plans and new plans carrying a visual survey."""
    if not isinstance(value, dict) or set(value) not in ({'steps'}, {'steps', 'scene_assessment'}):
        raise ValueError('Invalid saved plan fields')
    survey = value.get('scene_assessment')
    if survey is not None:
        validate_scene_assessment(survey)
    return validate_plan({'steps': value['steps']}), deepcopy(survey)


def inventory_updates(survey):
    """Convert current survey claims to semantic memory, without grasp claims."""
    validate_scene_assessment(survey)
    return [{'name': entity['name'], 'kind': entity['kind'], 'location': entity['location'],
             'status': 'visible' if entity['visibility'] == 'visible' else 'uncertain',
             'evidence': entity['evidence']} for entity in survey['entities']]
