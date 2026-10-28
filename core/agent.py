"""Plans and bounded robot programs shared by the agent and simulator."""

import ast
from core.scene_memory import ASSESSMENT_SCHEMA, MEMORY_ENTRY_SCHEMA, PERCEPTION_SCHEMA, validate_scene_fields

STEP_PROPERTIES = {
    'id': {'type': 'string'},
    'instruction': {'type': 'string'},
    'target': {'type': 'string'},
    'success_criteria': {'type': 'string'},
    'max_steps': {'type': 'integer'},
}
STEP_SCHEMA = {
    'type': 'object', 'properties': STEP_PROPERTIES,
    'required': list(STEP_PROPERTIES), 'additionalProperties': False,
}
PLAN_SCHEMA = {
    'type': 'object', 'properties': {'steps': {'type': 'array', 'items': STEP_SCHEMA}},
    'required': ['steps'], 'additionalProperties': False,
}
DECISION_SCHEMA = {
    'type': 'object',
    'properties': {
        'status': {'type': 'string', 'enum': ['continue', 'complete', 'replan', 'stop', 'inspect']},
        'evidence': {'type': 'string'},
        'program': {'type': 'string'},
        'replacement_steps': {'type': 'array', 'items': STEP_SCHEMA},
        'assessment': ASSESSMENT_SCHEMA,
        'memory_updates': {'type': 'array', 'items': MEMORY_ENTRY_SCHEMA},
        'perception': PERCEPTION_SCHEMA,
        'handoff': {
            'type': 'object',
            'properties': {
                'readiness': {'type': 'string', 'enum': ['ready', 'blocked', 'uncertain', 'not_applicable']},
                'evidence': {'type': 'string'},
            },
            'required': ['readiness', 'evidence'], 'additionalProperties': False,
        },
    },
    'required': ['status', 'evidence', 'program', 'replacement_steps', 'handoff', 'assessment', 'memory_updates', 'perception'],
    'additionalProperties': False,
}


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'steps'}:
        raise ValueError('A plan must contain only steps')
    steps = plan['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 10:
        raise ValueError('A plan needs 1–10 short subtasks')
    identifiers = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) != set(STEP_PROPERTIES):
            raise ValueError('Invalid subtask fields')
        for key in ('id', 'instruction', 'target', 'success_criteria'):
            if not isinstance(step[key], str) or not 0 < len(step[key].strip()) <= 1000:
                raise ValueError(f'Invalid subtask {key}')
        if step['id'] in identifiers:
            raise ValueError('Subtask IDs must be unique')
        identifiers.add(step['id'])
        if type(step['max_steps']) is not int or not 1 <= step['max_steps'] <= 1000:
            raise ValueError('Subtask step budget must be 1–1000')
    return steps


def parse_program(source, max_steps=80, allow_gripper=False):
    """Parse literal calls; generated text is never eval/exec'd."""
    if not isinstance(source, str) or len(source) > 6000:
        raise ValueError('Invalid robot program')
    tree = ast.parse(source)
    if not 1 <= len(tree.body) <= 3:
        raise ValueError('A program needs 1–3 tool calls')
    commands = []
    total = 0
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError('Only direct tool calls are allowed')
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.args or any(key.arg is None for key in call.keywords):
            raise ValueError('Use named tools and literal keyword arguments')
        name = call.func.id
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg in arguments or not isinstance(keyword.value, ast.Constant):
                raise ValueError('Arguments must be unique literals')
            arguments[keyword.arg] = keyword.value.value
        if name == 'run_groot':
            if set(arguments) != {'instruction', 'steps'}:
                raise ValueError('run_groot requires instruction and steps')
            if not isinstance(arguments['instruction'], str) or not 0 < len(arguments['instruction'].strip()) <= 1000:
                raise ValueError('GR00T needs a short instruction')
        elif name == 'navigate':
            if (set(arguments) != {'fixture', 'steps'} or not isinstance(arguments['fixture'], str)
                    or not 0 < len(arguments['fixture'].strip()) <= 120):
                raise ValueError('navigate requires a named fixture and steps')
        elif name == 'set_gripper' and allow_gripper:
            if set(arguments) != {'closed', 'steps'} or type(arguments['closed']) is not bool:
                raise ValueError('set_gripper requires boolean closed and steps')
            if type(arguments['steps']) is not int or not 1 <= arguments['steps'] <= 10:
                raise ValueError('Gripper operation is limited to 10 control steps')
        else:
            raise ValueError(f'Tool unavailable: {name}')
        steps = arguments.get('steps')
        if type(steps) is not int or not 1 <= steps <= max_steps:
            raise ValueError('Invalid tool step budget')
        total += steps
        commands.append((name, arguments))
    if total > max_steps:
        raise ValueError('Program exceeds the decision step budget')
    return commands


def validate_decision(decision, max_steps=80, allow_gripper=False):
    if not isinstance(decision, dict) or set(decision) != set(DECISION_SCHEMA['properties']):
        raise ValueError('Invalid decision fields')
    status = decision['status']
    if status not in ('continue', 'complete', 'replan', 'stop', 'inspect'):
        raise ValueError('Invalid decision status')
    if not isinstance(decision['evidence'], str) or not 0 < len(decision['evidence'].strip()) <= 4000:
        raise ValueError('A decision needs observable evidence or an explicit uncertainty statement')
    if not isinstance(decision['program'], str) or not isinstance(decision['replacement_steps'], list):
        raise ValueError('Invalid program or replacement plan')
    validate_scene_fields(decision)
    handoff = decision['handoff']
    if (not isinstance(handoff, dict) or set(handoff) != {'readiness', 'evidence'}
            or handoff['readiness'] not in ('ready', 'blocked', 'uncertain', 'not_applicable')
            or not isinstance(handoff['evidence'], str) or not 0 < len(handoff['evidence'].strip()) <= 2000):
        raise ValueError('A decision needs handoff readiness and supporting evidence')
    if status == 'continue':
        parse_program(decision['program'], max_steps, allow_gripper)
    elif decision['program'].strip():
        raise ValueError('Only continue decisions may execute motion')
    if status == 'replan':
        validate_plan({'steps': decision['replacement_steps']})
    elif decision['replacement_steps']:
        raise ValueError('Only replan decisions may replace remaining subtasks')
    return decision
