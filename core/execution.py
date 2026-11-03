"""Task-independent commitment to an active policy instruction."""

from copy import deepcopy

from core.goals import normalized


SWITCH_REASONS = {'blocked_access', 'wrong_target', 'object_loss', 'unsafe_state',
                  'goal_contradiction', 'useful_required_object_held', 'execution_failure', 'controlled_release', 'reference_refinement'}


def supervision_budget(max_decisions, max_api_calls, decisions_used, api_calls_used,
                       episode_steps_left, decision_steps):
    """Separate optional episode caps from the bounded wire-request allowance."""
    decisions_left = max(0, max_decisions - decisions_used) if max_decisions else None
    total_calls_left = max(0, max_api_calls - api_calls_used) if max_api_calls else None
    request_calls = 200 if total_calls_left is None else min(200, total_calls_left)
    motion_bound = episode_steps_left
    for remaining in (decisions_left, total_calls_left):
        if remaining is not None:
            motion_bound = min(motion_bound, remaining * decision_steps)
    return decisions_left, request_calls, motion_bound


def completed_stage_decision(decision, current_step, ledger=None):
    """Separate a supported stage outcome from a model's next-action proposal.

    No proposed motion is executed by this conversion. Completion is still a
    visual model claim; observation validation does not establish physical truth.
    """
    if not isinstance(decision, dict):
        return None
    assessment = decision.get('assessment')
    if not isinstance(assessment, dict) or assessment.get('outcome') != 'satisfied':
        return None
    evidence, refs = assessment.get('evidence'), assessment.get('observation_refs')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Stage completion requires observable evidence')
    if ledger is not None:
        ledger.validate_refs(refs, current_step)
    elif (not isinstance(refs, list) or not refs or any(
            not isinstance(ref, dict) or set(ref) != {'step', 'camera'}
            or type(ref['step']) is not int or not 0 <= ref['step'] <= current_step
            or not isinstance(ref['camera'], str) or not ref['camera'].strip() for ref in refs)):
        raise ValueError('Stage completion requires dated observation references')
    if not any(ref['step'] == current_step for ref in refs):
        raise ValueError('Stage completion needs current observation evidence')
    result = deepcopy(decision)
    result.update(status='complete', program='', replacement_steps=[])
    for field in ('action', 'recovery'):
        if field in result:
            result[field] = None
    return result


def validate_stage_switch(committed, proposed, decision, current_step, ledger=None, allow_refinement=False):
    if normalized(committed) == normalized(proposed):
        return
    from core.atomic_language import gripper_release_instruction
    from core.boundary_review import placement_stage
    if gripper_release_instruction(proposed):
        if (not placement_stage({'step': {'instruction': committed}})
                or (decision.get('assessment', {}).get('transport') or {}).get('grasp') != 'supported'):
            raise ValueError('Controlled release requires pending placement and a supported held-object claim')
    recovery = decision.get('recovery')
    if isinstance(recovery, dict) and recovery.get('necessity') == 'reference_refinement' and not allow_refinement:
        raise ValueError('Reference refinement is available only in relational instruction mode')
    if (not isinstance(recovery, dict) or recovery.get('necessity') not in SWITCH_REASONS
            or not isinstance(recovery.get('description'), str) or not recovery['description'].strip()):
        raise ValueError('Active stage is not verified complete. A different instruction requires current '
                         'evidence of a necessary intervention; a few failed bursts or different tool alias are insufficient')
    assessment = decision['assessment']
    if assessment.get('outcome') == 'satisfied':
        raise ValueError('Complete the verified stage before switching to the next one')
    refs = recovery.get('observation_refs')
    if ledger is not None:
        ledger.validate_refs(refs, current_step)
    if not isinstance(refs, list) or not refs or not any(ref.get('step') == current_step for ref in refs):
        raise ValueError('Instruction switch requires current observation evidence of its necessity')
    if recovery['necessity'] == 'useful_required_object_held':
        transport = assessment.get('transport') or {}
        if transport.get('grasp') != 'supported' or not transport.get('held_object', '').strip():
            raise ValueError('Opportunistic reordering requires a supported held-object assessment')
