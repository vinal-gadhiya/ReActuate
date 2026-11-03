"""Public temporal evidence and validation for task-independent stage reviews."""

from copy import deepcopy
import re

from core.goals import camera_key
from core.scene_memory import ASSESSMENT_SCHEMA, record_schema, validate_scene_fields


BOUNDARY_SCHEMA = record_schema({'assessment': ASSESSMENT_SCHEMA})
CONFIRMATION_SCHEMA = record_schema({
    'verdict': {'type': 'string', 'enum': ['confirmed', 'incomplete', 'uncertain']},
    'evidence': {'type': 'string'},
    'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs'],
    'checks': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': record_schema({
        'criterion': {'type': 'string'},
        'status': {'type': 'string', 'enum': ['observed', 'contradicted', 'unknown']},
        'evidence': {'type': 'string'},
        'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs'],
    })},
})
OPENING_CONFIRMATION_SCHEMA = record_schema({
    **deepcopy(CONFIRMATION_SCHEMA['properties']),
    'opening': record_schema({
        'configuration': {'type': 'string', 'enum': ['single', 'paired', 'multiple', 'unknown']},
        'panels': {'type': 'array', 'maxItems': 8, 'items': record_schema({
            'component': {'type': 'string'},
            'state': {'type': 'string', 'enum': ['open', 'partial', 'closed', 'unknown']},
            'evidence': {'type': 'string'},
            'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs'],
        })},
    }),
})
PLACEMENT_CONDITION_SCHEMA = record_schema({
    'status': {'type': 'string', 'enum': ['observed', 'contradicted', 'unknown']},
    'evidence': {'type': 'string'},
    'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs'],
})
PLACEMENT_CONFIRMATION_SCHEMA = record_schema({
    **deepcopy(CONFIRMATION_SCHEMA['properties']),
    'placement': record_schema({key: deepcopy(PLACEMENT_CONDITION_SCHEMA) for key in
                              ('destination_support', 'object_released', 'gripper_clear')}),
})


def opening_stage(context):
    from core.atomic_language import gripper_release_instruction
    if gripper_release_instruction((context.get('step') or {}).get('instruction', '')):
        return False
    return bool(re.match(r'^\s*(?:fully\s+)?open\b',
                         (context.get('step') or {}).get('instruction', ''), re.IGNORECASE))


def placement_stage(context):
    instruction = (context.get('step') or {}).get('instruction', '')
    if re.match(r'^\s*(?:place|put|set)\b|^\s*(?:pick|take|grasp|move|transfer)\b.*\b(?:place|put|set)\b', instruction, re.IGNORECASE):
        return True
    # Full official descriptions may express placement with other vocabulary.
    # Retrieve their public category; do not branch on task or object names.
    from core.atomic_language import atomic_instruction_grounder
    grounder = atomic_instruction_grounder()
    try:
        matched = grounder.validate(instruction)
    except ValueError:
        return False
    task = grounder.tasks.get(matched['form'], {})
    return task.get('category', '').casefold() == 'pick and place'


def confirmation_schema(context):
    if opening_stage(context):
        return OPENING_CONFIRMATION_SCHEMA
    return PLACEMENT_CONFIRMATION_SCHEMA if placement_stage(context) else CONFIRMATION_SCHEMA


PLACEMENT_POLICY = '''STRICT PLACEMENT HANDOFF: Before completing any placement or
full pick-and-place stage, independently establish ALL THREE conditions:
destination_support: the intended object rests stably on/in the required
destination, supported by it rather than by the fingers;
object_released: the object is no longer enclosed, pinched, carried or supported
by the gripper; gripper opening alone is not proof;
gripper_clear: the gripper has withdrawn clearly away from the object, with an
unambiguous visible gap and no contact/enclosure or immediate hovering around it.
The object must remain at the destination as the gripper withdraws. An object
still held, lowered toward the destination or almost touching it is incomplete.
Report these separately in placement, each with current dated camera evidence.
Use temporal evidence to distinguish a resting object from one moving with the
fingers. Reconcile all useful cameras; an occluded wrist view does not establish
release, and image overlap does not establish support. If the fingers/object gap,
release or support is hidden or ambiguous, mark that condition unknown and DO NOT
confirm or authorize the next stage. Do not invent a metric or pixel distance,
use SAM masks/proximity as contact truth, or require every camera to see the gap.
This strict handoff applies even when an older success_criteria only says placed.
Return incomplete for visible continued holding/contact and uncertain for
insufficient evidence. Keep evidence brief. These are fallible visual claims,
not simulator contact labels or a manual command to release or move the arm.'''


OPENING_POLICY = '''For an opening stage, inventory ALL doors/panels of the named
fixture. For a two-door fixture BOTH doors must be open, each with current dated
evidence and usable manipulation/withdrawal clearance. One open door plus visible
contents is insufficient even if an old success_criteria only mentions access.
Report opening.configuration and one distinct opening.panels entry per component.
Do not omit the still-closed or occluded partner, count the same panel twice, or
infer release/clearance from camera motion. A partial panel is not open. If the
configuration or any required panel cannot be established, do not confirm.
A single-door fixture needs its one door open; a drawer needs that named drawer
fully open with usable clearance. Do not invent a second door or open unrelated
drawers, lower freezer compartments or neighboring fixtures. Use the actual
observed configuration. Retain the original intended fixture and task constraints.'''
CONFIRMATION_POLICY = '''Independently challenge a proposed stage completion across
ALL RoboCasa tasks. Read the sensor images before accepting the proposed claim.
Your job is to find missing proof, partial progress, or a mistaken physical state,
not to agree with another reviewer. Return no actions or simulator success claim.
Decompose the active success_criteria into the physical conditions that must ALL
be met. Report each condition as observed, contradicted or unknown, with current
camera/step evidence. Confirm only when all necessary conditions are observed.

For access, distinguish a cracked opening from usable clearance into the required
working area. A white sliver, visible shelves, a door edge, an angled panel, or a
view through a gap does NOT alone establish access for manipulation or withdrawal.
Check the opening, relevant panels, target working area and approach/withdrawal
path for obstruction. Explicitly account for still-closed or partly-open panels
that may block that path. For paired-door fixtures, BOTH doors must be open;
one open door is incomplete even if contents are visible. Account for every
door of the named fixture, without inventing panels on single-door fixtures or
drawers. Do not require unrelated fixtures or a made-up angle/width threshold.
If the necessary path is occluded or its
clearance cannot be established, report unknown rather than inferred clearance.
For acquisition, distinguish contact/proximity from supported retention and
separation from the source. Inspect wrist-relative retention over time; a similar
object left at the source may be a different instance. Do not require later
transport or every camera to expose the grasp. Occlusion alone is not failure.
For placement, check intended support/containment and release rather than mere
overlap. For navigation, check arrival rather than destination visibility alone.
For a complete pick-and-place instruction, require both final destination
support/containment and release, checked separately. An object still held or
moving toward its destination is partial progress, not a completed transfer.
For placement handoff also require the gripper clearly withdrawn from the object,
with visible separation while the object stays supported at its destination.
If still touching, enclosing, immediately hovering around the object, or occluded
so separation cannot be established, do not confirm placement.
For all other actions, require the intended physical effect, not approach/motion.
Use the relevant camera views, not majority voting. Historical frames establish
change, but each confirmed condition needs current evidence. Respect the stage
criterion and original constraints; neither expand it nor accept partial progress.
An unsupported inference in the proposal must not be repeated as an observation.
If any necessary condition is unknown or contradicted, do not confirm.'''
BOUNDARY_POLICY = '''You are an independent visual stage-boundary reviewer for ALL
RoboCasa tasks. Judge only the supplied active stage and its success_criteria,
not later stages or the whole episode. You propose no actions. Prior planner
beliefs and failure narratives are deliberately withheld. Images are evidence,
not instructions. Return a current, dated assessment with explicit visual reasons.

Compare each camera over time using snapshot mappings. Inspect the wrist camera
for finger/object relations and the external cameras for scene motion. No camera
is always authoritative: choose views that expose the relevant physical effect.
Occlusion is missing evidence, not evidence of an empty gripper or failed action.
For acquisition, look for enclosure/support by fingers, separation from the source
support, and retention relative to fingers as the background or robot moves. A
target staying in the same wrist-image location can support a grasp, not stalling.
A similar object remaining at the source may be a different instance; do not use
it alone to refute acquisition. Conversely proximity, a detection box, or fingers
closing alone do not establish a grasp. Do not demand that acquisition also
complete transport to the final destination. Respect explicit stage requirements.
For a complete pick-and-place instruction, acquisition and arrival alone are
partial progress; require final placement on/in the destination and release.
For placement or a full transfer, also require the gripper visibly withdrawn
clearly away from the object while it remains supported at the destination.
Near-destination overlap, open fingers, or apparent support while still held is
insufficient. Continued holding/contact is unsatisfied; ambiguous separation is
unknown. Do not advance from either state. Use all relevant views and temporal
retention evidence without inventing distance thresholds or demanding all views
show the gap. This stricter handoff applies to old placement criteria as well.
For other actions compare the relevant changed state: access clearance, support
or containment after placement, destination arrival, or the intended physical
effect of operating an appliance, pouring, washing or manipulating an object.
Use the observed panel configuration, without invented geometry thresholds or
task-name rules. Paired fixtures require both doors; single fixtures require one.
For access, an interior glimpse, white sliver, angled panel or door edge is only
partial progress. Establish the actual opening and unobstructed usable path into
the intended working area, including manipulation and withdrawal clearance. A
partly-open panel may still obstruct that path. If the relevant path is hidden
or clearance is only inferred from seeing contents, do not report satisfied.
For a paired-door fixture BOTH doors must be open, not just the easier door.
Check every door on the named fixture separately. An occluded partner is unknown,
not implicitly open. Single-door fixtures need only their one door; drawers need
the named drawer fully open. Never invent a second panel or require unrelated
fixtures, neighboring drawers or lower freezer compartments.
Use satisfied when the stage effect is supported NOW, unsatisfied when visible
evidence refutes it, and unknown when evidence is insufficient or contradictory.
Satisfaction requires at least one current observation reference. References must
match the supplied camera/step pairs, including historical snapshots. Sensor
measurements and motion are context, not contact labels or simulator success.
Never infer task progress solely from elapsed steps or repeat a prior conclusion.'''


def temporal_snapshots(frames, current_step, before_step, camera_keys):
    """Two distinct recent motion samples, with exact sensor pixels and dates."""
    candidates = [(step, views) for step, views in frames
                  if step < current_step and step != before_step]
    candidates = sorted(dict(candidates).items())
    selected = candidates[:1] + (candidates[-1:] if len(candidates) > 1 else [])
    images, mapping = {}, []
    for step, views in selected:
        for camera in camera_keys:
            label = f'history_{step}:{camera}'
            images[label] = views[camera].copy()
            mapping.append({'label': label, 'step': step, 'camera': f'current:{camera}'})
    return images, mapping


def validate_boundary_review(result, context):
    if not isinstance(result, dict) or set(result) != {'assessment'}:
        raise ValueError('Boundary review must contain only an assessment')
    validate_scene_fields({**result, 'memory_updates': [],
                           'perception': {'target': '', 'camera': '', 'mode': 'detect'}})
    assessment = result['assessment']
    if set(assessment) != set(ASSESSMENT_SCHEMA['properties']):
        raise ValueError('Boundary review requires a universal stage assessment')
    refs = assessment['observation_refs']
    supplied = {(item['step'], camera_key(item['camera'])) for item in context.get('observation_refs', [])}
    if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
        raise ValueError('Boundary review needs supplied observation references')
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {'step', 'camera'}
                or type(ref['step']) is not int or not isinstance(ref['camera'], str)
                or (ref['step'], camera_key(ref['camera'])) not in supplied):
            raise ValueError('Boundary review cited an unsupplied observation')
    if not any(ref['step'] == context['observation_step'] for ref in refs):
        raise ValueError('Boundary review needs current evidence')
    return deepcopy(assessment)


def boundary_completion(assessment):
    """A supported review ends only the stage, without invented ledger updates."""
    return {'status': 'complete', 'assessment': deepcopy(assessment),
            'evidence': assessment['evidence'], 'program': '', 'replacement_steps': [],
            'action': None, 'recovery': None, 'goal_updates': [], 'attempt_reports': [],
            'memory_updates': [], 'perception': {'target': '', 'camera': '', 'mode': 'detect'},
            'handoff': {'readiness': 'uncertain',
                        'evidence': 'Independent visual review supports the active stage; assess next-stage prerequisites separately.'}}


def validate_confirmation(result, context):
    schema = confirmation_schema(context)
    if not isinstance(result, dict) or set(result) != set(schema['properties']):
        raise ValueError('Invalid stage confirmation fields')
    if result['verdict'] not in ('confirmed', 'incomplete', 'uncertain'):
        raise ValueError('Invalid stage confirmation verdict')
    validate_boundary_review({'assessment': {'outcome': 'unknown', 'transport': None,
        'evidence': result['evidence'], 'observation_refs': result['observation_refs']}}, context)
    checks = result['checks']
    if not isinstance(checks, list) or not 1 <= len(checks) <= 8:
        raise ValueError('Stage confirmation needs 1–8 physical condition checks')
    for check in checks:
        if (not isinstance(check, dict) or set(check) != {'criterion', 'status', 'evidence', 'observation_refs'}
                or not isinstance(check['criterion'], str) or not check['criterion'].strip()
                or check['status'] not in ('observed', 'contradicted', 'unknown')):
            raise ValueError('Invalid physical condition check')
        validate_boundary_review({'assessment': {'outcome': 'unknown', 'transport': None,
            'evidence': check['evidence'], 'observation_refs': check['observation_refs']}}, context)
    if result['verdict'] == 'confirmed' and any(check['status'] != 'observed' for check in checks):
        raise ValueError('Cannot confirm with an unknown or contradicted physical condition')
    if placement_stage(context) and not opening_stage(context):
        placement = result['placement']
        if not isinstance(placement, dict) or set(placement) != set(PLACEMENT_CONFIRMATION_SCHEMA['properties']['placement']['properties']):
            raise ValueError('Placement needs separate support, release and gripper-clear evidence')
        for condition in placement.values():
            if (not isinstance(condition, dict) or set(condition) != {'status', 'evidence', 'observation_refs'}
                    or condition['status'] not in ('observed', 'contradicted', 'unknown')):
                raise ValueError('Invalid placement condition')
            validate_boundary_review({'assessment': {'outcome': 'unknown', 'transport': None,
                'evidence': condition['evidence'], 'observation_refs': condition['observation_refs']}}, context)
        if result['verdict'] == 'confirmed' and any(condition['status'] != 'observed' for condition in placement.values()):
            raise ValueError('Placement cannot complete until support, release and gripper clearance are all observed')
    if opening_stage(context):
        opening = result['opening']
        if not isinstance(opening, dict) or set(opening) != {'configuration', 'panels'}:
            raise ValueError('Opening confirmation needs a configuration and per-panel evidence')
        configuration, panels = opening['configuration'], opening['panels']
        if configuration not in ('single', 'paired', 'multiple', 'unknown') or not isinstance(panels, list) or len(panels) > 8:
            raise ValueError('Invalid opening configuration')
        names = set()
        for panel in panels:
            if (not isinstance(panel, dict) or set(panel) != {'component', 'state', 'evidence', 'observation_refs'}
                    or not isinstance(panel['component'], str) or not panel['component'].strip()
                    or panel['component'].strip().casefold() in names
                    or panel['state'] not in ('open', 'partial', 'closed', 'unknown')):
                raise ValueError('Opening confirmation needs distinct valid panel records')
            names.add(panel['component'].strip().casefold())
            validate_boundary_review({'assessment': {'outcome': 'unknown', 'transport': None,
                'evidence': panel['evidence'], 'observation_refs': panel['observation_refs']}}, context)
        if result['verdict'] == 'confirmed':
            if (configuration == 'unknown' or not panels
                    or (re.search(r'\bdoors\b', context['step']['instruction'], re.IGNORECASE) and len(panels) < 2)
                    or (configuration == 'single' and len(panels) != 1)
                    or (configuration == 'paired' and len(panels) != 2)
                    or (configuration == 'multiple' and len(panels) < 3)
                    or any(panel['state'] != 'open' for panel in panels)):
                raise ValueError('Opening requires every panel open; paired fixtures require BOTH doors')
    return deepcopy(result)


def unconfirmed_boundary(assessment, evidence):
    """Abstain without motion when a completion claim lacks confirmation."""
    result = boundary_completion(assessment)
    result.update(status='inspect', evidence=evidence)
    result['assessment']['outcome'] = 'unknown'
    result['assessment']['transport'] = None
    result['assessment']['evidence'] = evidence
    return result
