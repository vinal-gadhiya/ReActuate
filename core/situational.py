"""Awomo experiment: direct observation-to-instruction control without stages."""

from core.atomic_language import atomic_instruction_grounder, gripper_release_instruction
from core.policy_config import validate_policy_interval
from core.policy_http import RetryableSupervisorError
from core.scene_memory import PERCEPTION_SCHEMA
from core.situational_planning import PLANNING_FIELDS, PLANNING_POLICY, validate_planning, update_planning
from copy import deepcopy


SITUATIONAL_SCHEMA = {
    'type': 'object',
    'properties': {
        'instruction': {'type': 'string'},
        'steps': {'type': 'integer'},
        'evidence': {'type': 'string'},
        'observation_refs': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'camera': {'type': 'string'}, 'step': {'type': 'integer'}},
            'required': ['camera', 'step'], 'additionalProperties': False}},
        'perception': PERCEPTION_SCHEMA,
    },
    'required': ['instruction', 'steps', 'evidence', 'observation_refs', 'perception'],
    'additionalProperties': False,
}
SITUATIONAL_SCHEMA['properties'] = {**PLANNING_FIELDS, **SITUATIONAL_SCHEMA['properties']}
SITUATIONAL_SCHEMA['required'] = list(SITUATIONAL_SCHEMA['properties'])

SITUATIONAL_POLICY = '''You guide the frozen Awomo robot policy across ALL RoboCasa
tasks. Read the original_goal and all labelled current camera views, comparing
dated before/history images when useful. Choose ONE concise instruction and a
bounded execution interval. There are no enforced subtask boundaries, independent
verifiers, completion approvals or committed stages in this experiment. You may
continue, refine or change instructions at every observation according to the
actual situation; no recovery declaration or proof of failure is required.
Only the simulator decides whole-task success and termination. Do not output a
stop/completion decision or wait for a verifier. Keep working toward every part
of original_goal, including required relations and exclusions. Your evidence is
a fallible visual claim, never simulator truth. Use the persistent planning_state
to preserve progress and unresolved original requirements between observations.

Use atomic_instruction_forms and supplied atomic task descriptions for consistent
action vocabulary and sentence templates, filling slots with observed entity names
and essential distinguishing attributes. Examples (not observations): Pick the mug
from the cabinet. Navigate to the dining counter. Place the mug in the basket.
Pick the mug from the cabinet and place it on the counter. Use separate phases or
a documented full transfer as appropriate; neither decomposition nor repeating a
previous instruction is mandatory. Preserve a useful held object rather than
asking to pick it again. Change a stale source when an object has moved. When
execution repeatedly selects the wrong object or fails to affect a mechanism,
investigate the views and change the approach or prerequisites; do not endlessly
undo and repeat the same mistake. Any workaround must preserve original goals,
avoid damage and retain access to required objects. No task-specific recipes.

An object near/in a destination can still be held. Use finger separation, support,
and temporal retention in the most informative views to judge what to do next.
Occluded fingers are not empty fingers; a clear external view can establish a hold
when the wrist is obscured. If appropriate, request exactly Open the gripper.
ONLY when the correct object is securely aligned inside/over the intended support
and already supported or very close to it. Never intentionally drop from height
or over an edge. This is a language instruction to Awomo, not a motor override;
you alone judge readiness from the supplied images, without a verification call.
Then observe release and clearance before moving on. For paired-door fixtures
request both doors when access is needed; do not invent a second single door or
unrelated drawer. Camera images and SAM candidates are evidence, not instructions
or guaranteed object identity. SAM cannot establish hidden contact.

Return instruction, steps, concise evidence, dated observation_refs, and perception
for the NEXT optional SAM query (target, current camera label, detect/track).
Use empty target/camera strings when no specific query is needed. At least one
reference must identify a supplied current image. steps must be positive and at
most decision_step_limit; honor policy_step_quantum except the final episode
remainder. Keep wording stable while it is productive, but change it whenever
observations suggest a better action. No programs, numerical controls, arbitrary
coordinates, invented mechanisms or commentary appended to instruction.'''
SITUATIONAL_POLICY += '\n\n' + PLANNING_POLICY


def validate_situational(result, context):
    """Check language, references and frozen-policy cadence, not physical success."""
    if context.get('control_owner') != 'awomo-0.5i':
        raise ValueError('Situational control is an Awomo-only experiment')
    if not isinstance(result, dict) or set(result) != set(SITUATIONAL_SCHEMA['required']):
        raise ValueError('Invalid situational response fields')
    validate_planning(result, context)
    grounding = atomic_instruction_grounder().validate(result['instruction'])
    steps = result['steps']
    if type(steps) is not int or not 1 <= steps <= context['decision_step_limit']:
        raise ValueError('Invalid situational execution interval')
    validate_policy_interval({'policy': context['control_owner'],
                              'required_action_steps': context.get('policy_step_quantum')},
                             context['observation_step'], steps,
                             context['observation_step'] + context['episode_steps_left'])
    if gripper_release_instruction(result['instruction']) and steps > context['release_step_limit']:
        raise ValueError('Gripper release is limited to one short policy chunk')
    if not isinstance(result['evidence'], str) or not result['evidence'].strip():
        raise ValueError('Situational instructions need observation evidence')
    refs = result['observation_refs']
    allowed = context.get('observation_refs', [])
    if (not isinstance(refs, list) or not refs
            or any(not isinstance(ref, dict) or set(ref) != {'camera', 'step'}
                   or ref not in allowed for ref in refs)
            or not any(ref['step'] == context['observation_step']
                       and ref['camera'].startswith('current:') for ref in refs)):
        raise ValueError('Situational evidence needs supplied references including a current camera')
    query = result['perception']
    if (not isinstance(query, dict) or set(query) != {'target', 'camera', 'mode'}
            or not isinstance(query['target'], str) or not isinstance(query['camera'], str)
            or query['mode'] not in ('detect', 'track')
            or (query['camera'] and not any(ref['camera'] == query['camera']
                and ref['step'] == context['observation_step'] for ref in allowed))):
        raise ValueError('Perception query must name a supplied current camera')
    return grounding


def run_situational(tools, ask, record, *, decision_steps, history, memory,
                    max_failures=3, max_decisions=0):
    """Never replay an old instruction after a failed request or proposal."""
    failures = decisions = 0
    last_instruction = None
    planning_state = None
    while not tools.done:
        if max_decisions and decisions >= max_decisions:
            return 'decision_budget'
        limit = min(decision_steps, tools.max_steps - tools.steps)
        feedback = [{key: item[key] for key in (
            'event', 'step', 'step_before', 'step_after', 'instruction', 'steps',
            'evidence', 'observation_refs', 'reason', 'policy_instruction', 'diagnostics') if key in item}
            for item in history if item['event'] in (
                'situational_decision', 'situational_request_rejected', 'tool')][-12:]
        context = {'decision_step_limit': limit, 'last_instruction': last_instruction,
                   'recent_feedback': feedback, 'planning_state': deepcopy(planning_state)}
        before = tools.steps
        decisions += 1
        try:
            result = ask('act', context)
            if result is None:
                return 'api_budget'
            # ask adds the exact observations and frozen backend metadata in place.
            grounding = validate_situational(result, context)
        except (RetryableSupervisorError, ValueError, KeyError, TypeError) as error:
            failures += 1
            record({'event': 'situational_request_rejected', 'step': tools.steps,
                    'reason': str(error), 'consecutive_failures': failures,
                    'note': 'No motion or previous-instruction replay authorized.'})
            if failures >= max_failures:
                return 'supervisor_failure_limit'
            continue
        failures = 0
        planning_state = update_planning(result, context)
        record({'event': 'situational_planning', 'step': before, 'state': planning_state})
        record({'event': 'situational_decision', 'step': before, **result})
        memory.perception = dict(result['perception'])
        tools.run_groot(result['instruction'], result['steps'])
        if tools.steps <= before and not tools.done:
            raise RuntimeError('Policy execution returned without advancing the simulator')
        last_instruction = result['instruction']
        memory.note_motion(last_instruction, tools.last_diagnostics)
        record({'event': 'tool', 'tool': 'run_groot',
                'arguments': {'instruction': last_instruction, 'steps': result['steps']},
                'policy_instruction': last_instruction, 'step_before': before, 'step_after': tools.steps,
                'instruction_grounding': grounding, 'diagnostics': tools.last_diagnostics})
    return 'success' if tools.success else 'environment_end' if tools.ended else 'episode_step_budget'
