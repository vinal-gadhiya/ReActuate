"""Task-independent commitment to an active policy instruction."""

from copy import deepcopy

from core.goals import normalized


SWITCH_REASONS = {'blocked_access', 'wrong_target', 'object_loss', 'unsafe_state',
                  'goal_contradiction', 'useful_required_object_held'}


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


def validate_stage_switch(committed, proposed, decision, current_step, ledger=None):
    if normalized(committed) == normalized(proposed):
        return
    recovery = decision.get('recovery')
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
