"""Evidence gate for a policy-language release intervention; no motor control."""

from core.boundary_review import validate_boundary_review, placement_stage
from core.scene_memory import record_schema, ASSESSMENT_SCHEMA


CONDITION = record_schema({'status': {'type': 'string', 'enum': ['observed', 'contradicted', 'unknown']},
    'evidence': {'type': 'string'}, 'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs']})
RELEASE_SCHEMA = record_schema({'verdict': {'type': 'string', 'enum': ['confirmed', 'incomplete', 'uncertain']},
    'evidence': {'type': 'string'}, 'observation_refs': ASSESSMENT_SCHEMA['properties']['observation_refs'],
    'supported_target_hold': CONDITION, 'destination_alignment': CONDITION, 'low_release_height': CONDITION})
RELEASE_POLICY = '''Independently judge whether a short Open the gripper language
request is appropriate NOW during the supplied pending placement. Inspect ALL
useful current/before/dated camera views. Confirm only when all three conditions
are observed: supported_target_hold: the intended object is actually held, not
a distractor; destination_alignment: it is over or inside the correct required
support/receptacle, with no edge/rim/obstruction or wrong destination under it;
low_release_height: it is already supported or visibly very close to the intended
support surface, so release will not drop it from height. Being somewhere above
the destination, image overlap, SAM masks or an occluded wrist view is insufficient.
If height, alignment, identity or hold is ambiguous, return uncertain; if visibly
wrong or too high return incomplete. Never invent metric/pixel thresholds or
contact truth. Cite supplied current camera/step evidence for EACH condition.
No actions, no task completion claim. Keep evidence brief.'''


def validate_release(result, context):
    if (not placement_stage(context) or not isinstance(result, dict)
            or set(result) != set(RELEASE_SCHEMA['properties'])
            or result['verdict'] not in ('confirmed', 'incomplete', 'uncertain')):
        raise ValueError('Invalid placement release review')
    def evidence(value):
        validate_boundary_review({'assessment': {'outcome': 'unknown', 'transport': None,
            'evidence': value['evidence'], 'observation_refs': value['observation_refs']}}, context)
    evidence(result)
    for name in ('supported_target_hold', 'destination_alignment', 'low_release_height'):
        condition = result[name]
        if (not isinstance(condition, dict) or set(condition) != {'status', 'evidence', 'observation_refs'}
                or condition['status'] not in ('observed', 'contradicted', 'unknown')):
            raise ValueError('Invalid release readiness condition')
        evidence(condition)
        if result['verdict'] == 'confirmed' and condition['status'] != 'observed':
            raise ValueError('Release requires observed hold, alignment and low height')
    return result
