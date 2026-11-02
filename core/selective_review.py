"""Task-independent review scheduling; never decides a physical outcome."""

from copy import deepcopy

from core.goals import normalized
from core.scene_memory import transport_assessment


TRANSITION_POLICY = '''Independently verify the physical claims that justify a
proposed intervention or goal-status change. Return only the requested verdict,
dated observation_refs and brief condition checks; no actions or success score.
The proposal, previous transport and requirement statuses are fallible claims.
Check each verification_reason against all relevant current/before/dated views.
Verify actual loss, identity change, wrong target, changed prerequisite or claimed
requirement state. For recovery also check that the proposed changed condition
addresses the observed cause and preserves original constraints. Do not require
the active stage to be complete merely to allow a justified prerequisite repair.
For execution_failure, check dated executed-attempt evidence of the failed effect
and the reason a changed strategy or decomposition is necessary. Burst counts,
ordinary approach motion and productive partial progress alone are insufficient.
Occlusion is missing evidence, not loss. Positive evidence from one useful view
is not refuted by another occluded view. A similar item at the source may be a
different instance. Compare identity and wrist-relative retention across time.
For fixtures, controls, lids, racks, navigation or pouring, use the intended
physical effect, not gripper state, contact or motion. Each confirmed condition
needs supplied current evidence. Confirm only if ALL consequential claims are
supported; otherwise report incomplete or uncertain. Keep each check to one
short sentence. Never invent simulator truth, geometry or task-specific rules.'''


def previous_transport(context):
    memory = context.get('scene_memory') or {}
    claims = [memory.get('assessment'), *reversed(memory.get('recent_assessments', []))]
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        transport = transport_assessment(claim)
        if transport.get('grasp'):
            return {**deepcopy(transport), 'observation_step': claim.get('step')}
    return {}


def review_reasons(proposal, context):
    reasons = []
    assessment = proposal.get('assessment') or {}
    if proposal.get('status') == 'complete' or assessment.get('outcome') == 'satisfied':
        reasons.append('stage_completion')
    recovery = proposal.get('recovery') or {}
    if recovery.get('necessity') not in (None, 'not_needed'):
        reasons.append('necessary_intervention')
    before, after = previous_transport(context), transport_assessment(assessment)
    if after.get('grasp') == 'supported' and before.get('grasp') != 'supported':
        reasons.append('new_acquisition_claim')
    if before.get('grasp') == 'supported':
        if after.get('grasp') in ('empty', 'lost'):
            reasons.append('grasp_loss_or_release')
        elif (after.get('grasp') == 'supported'
              and normalized(after.get('held_object', '')) != normalized(before.get('held_object', ''))):
            reasons.append('held_identity_change')
    if any(entry.get('status') == 'lost' for entry in proposal.get('memory_updates', [])):
        reasons.append('object_loss_claim')
    old_status = {item['id']: item.get('status') for item in
                  (context.get('goal_ledger') or {}).get('requirements', [])}
    if any(item.get('status') in ('satisfied', 'unsatisfied')
           and item.get('status') != old_status.get(item.get('goal_id'))
           for item in proposal.get('goal_updates', [])):
        reasons.append('original_goal_status_change')
    if any(item.get('outcome') in ('intended_object_acquired', 'other_required_object_acquired',
                                   'unrelated_object_acquired', 'prerequisite_changed')
           for item in proposal.get('attempt_reports', [])):
        reasons.append('consequential_attempt_claim')
    return reasons



