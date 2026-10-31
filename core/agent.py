"""Plans and bounded robot programs shared by the agent and simulator."""

import ast
import math
import re
from core.scene_memory import ASSESSMENT_SCHEMA, MEMORY_ENTRY_SCHEMA, PERCEPTION_SCHEMA, validate_scene_fields
from core.goals import ACTION_SCHEMA, GOAL_UPDATE_SCHEMA, ATTEMPT_REPORT_SCHEMA, RECOVERY_SCHEMA

STEP_PROPERTIES = {
    'id': {'type': 'string'},
    'instruction': {'type': 'string'},
    'target': {'type': 'string'},
    'success_criteria': {'type': 'string'},
    'max_steps': {'type': 'integer'},
    'goal_ids': {'type': 'array', 'items': {'type': 'string'}},
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
        'goal_updates': {'type': 'array', 'items': GOAL_UPDATE_SCHEMA},
        'attempt_reports': {'type': 'array', 'items': ATTEMPT_REPORT_SCHEMA},
        'action': {'anyOf': [ACTION_SCHEMA, {'type': 'null'}]},
        'recovery': {'anyOf': [RECOVERY_SCHEMA, {'type': 'null'}]},
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
DECISION_SCHEMA['required'] = list(DECISION_SCHEMA['properties'])
LEGACY_STEP_FIELDS = set(STEP_PROPERTIES) - {'goal_ids'}
LEGACY_DECISION_FIELDS = set(DECISION_SCHEMA['properties']) - {'goal_updates', 'attempt_reports', 'action', 'recovery'}


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'steps'}:
        raise ValueError('A plan must contain only steps')
    steps = plan['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 10:
        raise ValueError('A plan needs 1–10 short subtasks')
    identifiers = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) not in (set(STEP_PROPERTIES), LEGACY_STEP_FIELDS):
            raise ValueError('Invalid subtask fields')
        for key in ('id', 'instruction', 'target', 'success_criteria'):
            if not isinstance(step[key], str) or not 0 < len(step[key].strip()) <= 1000:
                raise ValueError(f'Invalid subtask {key}')
        if step['id'] in identifiers:
            raise ValueError('Subtask IDs must be unique')
        identifiers.add(step['id'])
        if type(step['max_steps']) is not int or not 1 <= step['max_steps'] <= 1000:
            raise ValueError('Subtask step budget must be 1–1000')
        if 'goal_ids' in step and (not isinstance(step['goal_ids'], list) or not step['goal_ids']
                                  or any(not isinstance(item, str) or not item.strip() for item in step['goal_ids'])
                                  or len(step['goal_ids']) != len(set(step['goal_ids']))):
            raise ValueError('Execution stages need nonempty unique goal_ids')
    return steps


def parse_program(source, max_steps=80, allow_gripper=False, allow_vertical=False):
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
            value = keyword.value
            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub) and isinstance(value.operand, ast.Constant):
                if type(value.operand.value) not in (int, float):
                    raise ValueError('Only numeric negative literals are allowed')
                literal = -value.operand.value
            elif isinstance(value, ast.Constant):
                literal = value.value
            else:
                raise ValueError('Arguments must be unique literals')
            if keyword.arg in arguments:
                raise ValueError('Arguments must be unique literals')
            arguments[keyword.arg] = literal
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
        elif name == 'move_vertical' and allow_vertical:
            if (set(arguments) != {'distance', 'steps'} or type(arguments['distance']) not in (int, float)
                    or not math.isfinite(arguments['distance']) or not 0 < abs(arguments['distance']) <= 0.02):
                raise ValueError('move_vertical requires a finite signed distance of at most 0.02 metres')
            if type(arguments['steps']) is not int or not 1 <= arguments['steps'] <= 10:
                raise ValueError('Vertical placement is limited to 10 control steps')
        else:
            raise ValueError(f'Tool unavailable: {name}')
        steps = arguments.get('steps')
        if type(steps) is not int or not 1 <= steps <= max_steps:
            raise ValueError('Invalid tool step budget')
        total += steps
        commands.append((name, arguments))
    if total > max_steps:
        raise ValueError('Program exceeds the decision step budget')
    if allow_vertical and any(name == 'move_vertical' for name, _ in commands) and any(
            name == 'set_gripper' and not arguments['closed'] for name, arguments in commands):
        raise ValueError('Observe after vertical motion before requesting release in a separate decision')
    return commands


def navigation_fixture(name, arguments):
    """Identify navigation for accounting without changing the policy instruction."""
    if name == 'navigate':
        return arguments['fixture']
    if name != 'run_groot' or not re.match(r'^\s*navigate\b', arguments['instruction'], re.IGNORECASE):
        return None
    match = re.fullmatch(r'\s*navigate\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 -]*?)\.?\s*',
                         arguments['instruction'], re.IGNORECASE)
    if match is None:
        raise ValueError('Use one simple instruction: Navigate to the <location>.')
    return match[1].strip()


def validate_stage_program(commands):
    """Structural boundary only. Conciseness and wording are policy guidance."""
    if len(commands) != 1:
        raise ValueError('Execute one simple stage per decision, then observe before the next stage')


def validate_placement_destination(commands, planned_instruction):
    """Preserve an explicitly planned support/receptacle during execution."""
    pattern = r'\b(?:place|put)\s+.+?\s+(on|onto|in|into|inside|under|over|beside)\s+(.+?)\s*[.!?]?$'
    expected = re.search(pattern, planned_instruction, re.IGNORECASE)
    if expected is None:
        return

    def destination(match):
        relation = {'onto': 'on', 'into': 'in', 'inside': 'in'}.get(match[1].casefold(), match[1].casefold())
        name = re.sub(r'^(?:the|a|an)\s+', '', match[2].strip().rstrip('.!?').casefold())
        return relation, ' '.join(name.split())

    expected_relation, expected_name = destination(expected)
    for name, arguments in commands:
        actual = re.search(pattern, arguments['instruction'], re.IGNORECASE) if name == 'run_groot' else None
        if actual is None:
            continue
        relation, actual_name = destination(actual)
        if relation != expected_relation or not (actual_name == expected_name or actual_name.endswith(' ' + expected_name)):
            raise ValueError('Placement destination differs from the active goal. Preserve its support/receptacle; '
                             'do not substitute a nearby surface. Replan explicitly if the goal needs repair.')


def validate_decision(decision, max_steps=80, allow_gripper=False, allow_vertical=False):
    if not isinstance(decision, dict) or set(decision) not in (set(DECISION_SCHEMA['properties']), LEGACY_DECISION_FIELDS):
        raise ValueError('Invalid decision fields')
    status = decision['status']
    if status not in ('continue', 'complete', 'replan', 'stop', 'inspect'):
        raise ValueError('Invalid decision status')
    if not isinstance(decision['evidence'], str) or not 0 < len(decision['evidence'].strip()) <= 4000:
        raise ValueError('A decision needs observable evidence or an explicit uncertainty statement')
    if not isinstance(decision['program'], str) or not isinstance(decision['replacement_steps'], list):
        raise ValueError('Invalid program or replacement plan')
    validate_scene_fields(decision)
    if set(decision) == set(DECISION_SCHEMA['properties']):
        if not isinstance(decision['goal_updates'], list) or not isinstance(decision['attempt_reports'], list):
            raise ValueError('Goal updates and attempt reports must be arrays')
        if status == 'continue' and decision['action'] is None:
            raise ValueError('Executable actions need semantic metadata and explicit goal references on their stage')
    handoff = decision['handoff']
    if (not isinstance(handoff, dict) or set(handoff) != {'readiness', 'evidence'}
            or handoff['readiness'] not in ('ready', 'blocked', 'uncertain', 'not_applicable')
            or not isinstance(handoff['evidence'], str) or not 0 < len(handoff['evidence'].strip()) <= 2000):
        raise ValueError('A decision needs handoff readiness and supporting evidence')
    if status == 'continue':
        parse_program(decision['program'], max_steps, allow_gripper, allow_vertical)
    elif decision['program'].strip() and not (status == 'stop' and decision['program'].strip().casefold() == 'stop'):
        raise ValueError('Only continue decisions may execute motion')
    if status == 'replan':
        validate_plan({'steps': decision['replacement_steps']})
    elif decision['replacement_steps']:
        raise ValueError('Only replan decisions may replace remaining subtasks')
    return decision
