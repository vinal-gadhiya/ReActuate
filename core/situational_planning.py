"""Persistent public task interpretation and revisable Muse progress claims.

These records neither verify physical success nor authorize stage transitions.
"""

from copy import deepcopy


REFERENCE_SCHEMA = {
    'type': 'object', 'properties': {'camera': {'type': 'string'}, 'step': {'type': 'integer'}},
    'required': ['camera', 'step'], 'additionalProperties': False,
}
REQUIREMENT_SCHEMA = {
    'type': 'object', 'properties': {
        'id': {'type': 'string'}, 'description': {'type': 'string'},
        'success_condition': {'type': 'string'}, 'reference_frame': {'type': 'string'},
    },
    'required': ['id', 'description', 'success_condition', 'reference_frame'],
    'additionalProperties': False,
}
TASK_MODEL_SCHEMA = {
    'type': 'object', 'properties': {
        'success_picture': {'type': 'string'},
        'requirements': {'type': 'array', 'items': REQUIREMENT_SCHEMA},
    },
    'required': ['success_picture', 'requirements'], 'additionalProperties': False,
}
PROGRESS_SCHEMA = {
    'type': 'object', 'properties': {
        'requirement_id': {'type': 'string'},
        'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'satisfied', 'uncertain']},
        'evidence': {'type': 'string'},
        'observation_refs': {'type': 'array', 'items': REFERENCE_SCHEMA},
    },
    'required': ['requirement_id', 'status', 'evidence', 'observation_refs'], 'additionalProperties': False,
}
SCENE_FACT_SCHEMA = {
    'type': 'object', 'properties': {
        'id': {'type': 'string'}, 'description': {'type': 'string'},
        'basis': {'type': 'string', 'enum': ['observed', 'inferred', 'unknown']},
        'observation_refs': {'type': 'array', 'items': REFERENCE_SCHEMA},
    },
    'required': ['id', 'description', 'basis', 'observation_refs'], 'additionalProperties': False,
}
PLANNING_FIELDS = {
    'task_model': {'anyOf': [TASK_MODEL_SCHEMA, {'type': 'null'}]},
    'scene_updates': {'type': 'array', 'items': SCENE_FACT_SCHEMA},
    'open_questions': {'type': 'array', 'items': {'type': 'string'}},
    'progress_updates': {'type': 'array', 'items': PROGRESS_SCHEMA},
    'next_plan': {'type': 'array', 'items': {'type': 'string'}},
    'plan_rationale': {'type': 'string'},
}

PLANNING_POLICY = '''Before the FIRST physical action, explicitly interpret the
whole original_goal, survey all current views, and describe what successful
completion would look like in task_model.success_picture. Extract every explicit
final requirement into a stable ID, description, observable success_condition,
and reference_frame. Preserve object identities, quantities, groups, sources,
destinations, left/right relations, dependencies and exclusions stated in the
task. For spatial arrangements identify the named fixture and viewpoint defining
left/right, not an unlabelled camera-relative guess. If the reference frame or
inventory is unknown, say so and plan to investigate it; do not invent it.
Distinguish "all" required objects from a single visible example. A neighboring
compartment may contain required items; include a search when the task or scene
supports that possibility, without inventing unseen inventory or extra goals.
Requirements describe final outcomes, not helper actions or compulsory stages.
The original_goal remains authoritative if your interpretation is incomplete.

Build an explicit scene understanding BEFORE the first physical action. Use
scene_updates for task-relevant object identities/locations, candidate sources,
destination surfaces/regions, access and mechanism states, held-object state,
visibility and spatial reference frames. Each fact has a stable id, description,
basis (observed, inferred, unknown) and current camera/step references. Do not
turn likely contents into observed inventory. Mark hidden/ambiguous state unknown
and list the important unanswered questions in open_questions. An initial scene
survey must include at least one fact, even if the useful fact is that a needed
area is occluded. Do not demand perfect knowledge of hidden compartments before
acting: when necessary, make opening, travel or a better observation part of the
plan. Prefer an action that resolves a consequential uncertainty while advancing
the task, rather than manipulating a guessed target or declaring a guessed result.
At later observations send only changed/new scene facts; prior facts survive with
their original evidence and date, so assess staleness and contradictions explicitly.
Replace a stale fact by its same id with new evidence. Supply the updated complete
open_questions list; remove questions when resolved. Inference remains inference.

In the FIRST response provide task_model and a concise next_plan before the
instruction. This public plan is an actionable summary, not a request for private
reasoning. At later observations use task_model=null: the harness preserves the
initial definitions separately from your mutable plan. Always check the entire
planning_state checklist, including completed claims and work still pending.
Return progress_updates for changed requirements only, with current camera/step
evidence. Use pending, in_progress, satisfied or uncertain. An unseen outcome is
not automatically undone; preserve dated supported progress, and revise a prior
claim if newer images contradict it. Acquisition alone does not satisfy a final
placement; proximity does not prove support, release or a required spatial relation.
Do not mark a helper action as a new final requirement. Unchanged entries survive
when omitted. If you discover an omission in the initial interpretation, explain
it in plan_rationale and include the missing work in next_plan; never silently
change the original task or forget its requirements.

Return an updated next_plan and brief plan_rationale each time, preserving useful
progress and adapting order, sources, access, travel and workaround strategy to
the current scene. Plans are advisory and do not lock the next instruction.
If all requirements appear satisfied but the simulator has not ended, investigate
uncertain or missing task conditions rather than stop or blindly replay finished
work. Progress is your own fallible claim; no separate verifier or subtask boundary
approval is introduced. Keep entries concise to limit inference overhead.'''


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_planning(result, context):
    """Validate identities and dated evidence, never the truth of an outcome."""
    previous = context.get('planning_state')
    task_model = result['task_model']
    if previous is None:
        if (not isinstance(task_model, dict) or set(task_model) != {'success_picture', 'requirements'}
                or not _text(task_model['success_picture'])
                or not isinstance(task_model['requirements'], list) or not task_model['requirements']):
            raise ValueError('Initial action needs an explicit success picture and original requirements')
        identifiers = set()
        for item in task_model['requirements']:
            if (not isinstance(item, dict) or set(item) != set(REQUIREMENT_SCHEMA['required'])
                    or any(not _text(item[key]) for key in REQUIREMENT_SCHEMA['required'])
                    or item['id'] in identifiers):
                raise ValueError('Original requirements need unique stable IDs and concrete success conditions')
            identifiers.add(item['id'])
    else:
        if task_model is not None:
            raise ValueError('Original task interpretation is preserved; revise the execution plan instead')
        identifiers = {item['id'] for item in previous['task_model']['requirements']}
    plan = result['next_plan']
    if not isinstance(plan, list) or not plan or any(not _text(item) for item in plan):
        raise ValueError('A concise next execution plan is required')
    if not _text(result['plan_rationale']):
        raise ValueError('Plan updates need a brief rationale')
    scene = result['scene_updates']
    if not isinstance(scene, list) or (previous is None and not scene):
        raise ValueError('Initial planning needs an explicit observed or uncertain scene survey')
    seen_facts = set()
    for fact in scene:
        if (not isinstance(fact, dict) or set(fact) != set(SCENE_FACT_SCHEMA['required'])
                or not _text(fact['id']) or not _text(fact['description'])
                or fact['basis'] not in ('observed', 'inferred', 'unknown') or fact['id'] in seen_facts):
            raise ValueError('Scene facts need unique IDs, descriptions and explicit evidence basis')
        _current_refs(fact['observation_refs'], context)
        seen_facts.add(fact['id'])
    if (not isinstance(result['open_questions'], list)
            or any(not _text(question) for question in result['open_questions'])):
        raise ValueError('Open questions must list the remaining task-relevant uncertainties')
    updates = result['progress_updates']
    if not isinstance(updates, list):
        raise ValueError('Progress updates must be a list')
    seen = set()
    for item in updates:
        if (not isinstance(item, dict) or set(item) != set(PROGRESS_SCHEMA['required'])
                or not isinstance(item['requirement_id'], str)
                or item['requirement_id'] not in identifiers or item['requirement_id'] in seen
                or item['status'] not in PROGRESS_SCHEMA['properties']['status']['enum']
                or not _text(item['evidence'])):
            raise ValueError('Progress updates must reference original requirement IDs once each')
        _current_refs(item['observation_refs'], context)
        seen.add(item['requirement_id'])


def _current_refs(refs, context):
    if (not isinstance(refs, list) or not refs
            or any(ref not in context['observation_refs'] for ref in refs)
            or not any(ref['step'] == context['observation_step']
                       and ref['camera'].startswith('current:') for ref in refs)):
        raise ValueError('Changed scene/progress claims need supplied current camera/step evidence')


def update_planning(result, context):
    """Persist omitted requirements and claims across the entire episode."""
    validate_planning(result, context)
    previous = context.get('planning_state')
    if previous is None:
        state = {'task_model': deepcopy(result['task_model']), 'progress': [
            {'requirement_id': item['id'], 'status': 'pending',
             'evidence': 'Original requirement; no completion claim yet.', 'observation_refs': []}
            for item in result['task_model']['requirements']]}
    else:
        state = deepcopy(previous)
    facts = {fact['id']: fact for fact in state.get('scene_facts', [])}
    for fact in result['scene_updates']:
        facts[fact['id']] = {**deepcopy(fact), 'observation_step': context['observation_step']}
    state['scene_facts'] = list(facts.values())
    state['open_questions'] = list(result['open_questions'])
    updates = {item['requirement_id']: deepcopy(item) for item in result['progress_updates']}
    state['progress'] = [updates.get(item['requirement_id'], item) for item in state['progress']]
    state.update(next_plan=deepcopy(result['next_plan']), plan_rationale=result['plan_rationale'],
                 observation_step=context['observation_step'],
                 interpretation='Muse progress claims, not simulator labels or stage approvals.')
    return state
