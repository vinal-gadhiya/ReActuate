"""Public visual survey required before generating a new task plan."""

from copy import deepcopy
import re

from core.agent import PLAN_SCHEMA, validate_plan
from core.goals import REQUIREMENT_SCHEMA, GoalLedger


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
    'requirements': {'type': 'array', 'items': REQUIREMENT_SCHEMA},
    'steps': deepcopy(PLAN_SCHEMA['properties']['steps']),
})


class PlanValidationError(ValueError):
    """A model plan needs correction; the request itself was well formed."""


def grounded_plan_schema(cameras):
    schema = deepcopy(GROUNDED_PLAN_SCHEMA)
    schema['properties']['scene_assessment']['properties']['entities']['items']['properties']['cameras']['items']['enum'] = sorted(cameras)
    return schema


def normalize_plan_cameras(result, cameras):
    """Resolve only unambiguous aliases of supplied current views, never before views."""
    result = deepcopy(result)
    aliases = {}
    for camera in cameras:
        if not camera.startswith('current:'):
            continue
        base = camera[len('current:'):]
        short = base.removeprefix('video.')
        short = re.sub(r'^robot\d+_', '', short)
        variants = {camera, base, base.removeprefix('video.'), short}
        if short.startswith('agentview_'):
            variants.add(short[len('agentview_'):])
        if short == 'eye_in_hand':
            variants.add('wrist')
        variants |= {'current:' + name for name in variants if not name.startswith('current:')}
        for name in variants:
            aliases.setdefault(name.casefold(), set()).add(camera)
    survey = result.get('scene_assessment') if isinstance(result, dict) else None
    entities = survey.get('entities', []) if isinstance(survey, dict) else []
    if not isinstance(entities, list):
        return result
    for entity in entities:
        views = entity.get('cameras') if isinstance(entity, dict) else None
        if not isinstance(views, list) or any(not isinstance(view, str) for view in views):
            continue
        if len(set(views)) != len(views):
            raise PlanValidationError('Duplicate scene entity camera references')
        normalized = []
        for view in views:
            matches = {view.strip()} if view.strip() in cameras else aliases.get(view.strip().casefold(), set())
            if len(matches) != 1:
                raise PlanValidationError(f'Unknown or ambiguous current camera {view!r}; use exactly one of {sorted(cameras)}')
            canonical = next(iter(matches))
            if canonical not in normalized:
                normalized.append(canonical)
        entity['cameras'] = normalized
    return result


def request_grounded_plan(ask, context, cameras, max_attempts=3):
    """Retry invalid plans without motion; ask owns API budgets and audit records."""
    feedback = None
    for _ in range(max_attempts):
        try:
            result = ask('plan', {**context, 'plan_validation_feedback': feedback,
                                  'allowed_current_cameras': sorted(cameras)})
            if result is None:
                raise RuntimeError('API budget exhausted before a valid initial plan was available')
            result = normalize_plan_cameras(result, cameras)
            validate_grounded_plan(result, cameras)
            return result
        except ValueError as error:
            feedback = str(error)
    raise PlanValidationError(f'No valid initial plan after {max_attempts} attempts: {feedback}')


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
    if not isinstance(result, dict) or set(result) not in ({'scene_assessment', 'steps'}, {'scene_assessment', 'requirements', 'steps'}):
        raise ValueError('Generated plan requires scene_assessment and steps')
    validate_scene_assessment(result['scene_assessment'], cameras)
    if 'requirements' in result:
        GoalLedger('', result['requirements']).validate_window(result['steps'])
    return validate_plan({'steps': result['steps']})


def load_plan(value):
    """Accept legacy operator plans and new plans carrying a visual survey."""
    if (not isinstance(value, dict) or 'steps' not in value
            or set(value) - {'steps', 'scene_assessment', 'requirements'}):
        raise ValueError('Invalid saved plan fields')
    survey = value.get('scene_assessment')
    if survey is not None:
        validate_scene_assessment(survey)
    if 'requirements' in value:
        GoalLedger('', value['requirements']).validate_window(value['steps'])
    return validate_plan({'steps': value['steps']}), deepcopy(survey)


def inventory_updates(survey):
    """Convert current survey claims to semantic memory, without grasp claims."""
    validate_scene_assessment(survey)
    return [{'name': entity['name'], 'kind': entity['kind'], 'location': entity['location'],
             'status': 'visible' if entity['visibility'] == 'visible' else 'uncertain',
             'evidence': entity['evidence']} for entity in survey['entities']]
