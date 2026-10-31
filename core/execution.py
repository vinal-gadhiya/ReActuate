"""Task-independent commitment to an active policy instruction."""

from core.goals import normalized


SWITCH_REASONS = {'blocked_access', 'wrong_target', 'object_loss', 'unsafe_state',
                  'goal_contradiction', 'useful_required_object_held'}


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
