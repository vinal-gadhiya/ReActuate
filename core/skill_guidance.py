"""Small shared reasoning illustrations, not task-specific execution scripts."""


RECOVERY_GUIDANCE = '''Use the active physical success criterion for EVERY skill.
Distinguish productive approach from failure; two policy bursts alone are not
two failed attempts. Diagnose an observed cause before changing strategy:
wrong entity/component, blocked access, missing prerequisite, occupied workspace,
poor view or stale state. A fixture/control can change while grasp/phase stays
constant. Reconcile all cameras; occlusion does not refute clear positive evidence.
After meaningful repeated failure, propose the least disruptive change that
addresses its cause, using an explicit supported recovery and atomic-catalog
wording. Options include a useful view at a known fixture, a missing prerequisite,
clearing an observed obstruction, reordering eligible work or permitted temporary
staging. These are options, not compulsory recipes. Preserve original goals,
dependencies and useful progress; restore temporary changes where required.
For acquisition prefer a known observed source and an essential distinguishing
modifier. Never invent a source, color, control or destination. Repeated verified
wrong-object acquisitions can justify moving a distractor to a suitable observed
support, then retrying the target; repeated views of one wrong grasp are not
separate acquisitions. No task-specific controller rules or manual motion tools
exist. Use brief evidence once, cite dates, and avoid duplicating explanations.
If verification_feedback is supplied, resolve its missing physical evidence or
correct its disputed claim. Do not repeatedly claim completion from unchanged
images or redo an acquisition just because one camera is occluded.'''


def reasoning_example(context):
    """One bounded illustration chosen by declared action vocabulary, never task ID."""
    text = (context.get('step') or {}).get('instruction', '').casefold().strip()
    if text.startswith(('pick ', 'place ', 'put ')):
        return {'situation': 'One camera is occluded, but another shows an object moving with the fingers.',
                'reasoning': 'Keep retention uncertain or supported as evidence warrants; obtain a useful view '
                             'before declaring loss. Use an observed source for a justified retry.',
                'scope': 'Illustration only; choose the actual action and wording from current evidence and atomic references.'}
    return {'situation': 'A requested state change is absent; an unmet access or setup condition is visible.',
            'reasoning': 'Verify that prerequisite, repair only the relevant condition, then retry and check '
                         'the intended effect. Contact, unchanged gripper state and elapsed steps do not decide success.',
            'scope': 'Applies to fixtures, controls, lids, racks, navigation and other skills; not a fixed recipe.'}
